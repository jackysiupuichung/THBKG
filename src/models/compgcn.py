"""CompGCN (Vashishth et al. 2020) for heterogeneous link prediction.

Composition-based multi-relational GCN. Each relation has a learnable embedding;
each layer composes a node embedding with its incoming-edge relation embedding
via a composition function (subtraction by default, optionally multiplication
or circular correlation), aggregates over neighbors, and updates with three
direction-typed weight matrices (incoming, outgoing, self-loop).

Inputs follow the same heterogeneous contract as the other models in this
package — node features per type, edge_index per relation. Internally we
homogenize: per-type linear projection to hidden_dim, then a single graph with
relation ids.

Reference: Vashishth et al. "Composition-based Multi-Relational Graph
Convolutional Networks", ICLR 2020.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from .decoder import Decoder


NodeType = str
EdgeType = Tuple[str, str, str]


def _circular_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Real-FFT based circular correlation: ifft(conj(fft(a)) * fft(b)).real
    fa = torch.fft.rfft(a, dim=-1)
    fb = torch.fft.rfft(b, dim=-1)
    return torch.fft.irfft(torch.conj(fa) * fb, n=a.size(-1), dim=-1)


def _compose(h: torch.Tensor, r: torch.Tensor, op: str) -> torch.Tensor:
    if op == "sub":
        return h - r
    if op == "mult":
        return h * r
    if op == "corr":
        return _circular_correlation(h, r)
    raise ValueError(f"Unknown composition op '{op}'")


class CompGCNLayer(MessagePassing):
    """Single CompGCN layer with direction-typed weights and relation update."""

    def __init__(self, in_channels: int, out_channels: int, num_relations: int,
                 composition: str = "sub", dropout: float = 0.1):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations  # *original* relations; we double for inverses internally.
        self.composition = composition
        self.dropout = dropout

        # Three direction-typed weights: original (in), inverse (out), self-loop.
        self.W_in = nn.Linear(in_channels, out_channels, bias=False)
        self.W_out = nn.Linear(in_channels, out_channels, bias=False)
        self.W_loop = nn.Linear(in_channels, out_channels, bias=False)

        # Relation transformation (shared across types).
        self.W_rel = nn.Linear(in_channels, out_channels, bias=False)

        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self,
        x: torch.Tensor,                # [N, in_channels]
        rel_emb: torch.Tensor,          # [2 * R, in_channels]  (R original + R inverse)
        edge_index: torch.Tensor,       # [2, E_total]  (original + inverse edges concatenated)
        edge_type: torch.Tensor,        # [E_total]      (ids in [0, 2R))
    ):
        E = edge_index.size(1)
        E_orig = E // 2  # first half are original, second half inverses (by construction in caller)

        # Edge "direction" mask: 0 for original (use W_in), 1 for inverse (use W_out).
        edge_dir = torch.zeros(E, dtype=torch.long, device=x.device)
        edge_dir[E_orig:] = 1

        # Propagate.
        out = self.propagate(
            edge_index, x=x, rel_emb=rel_emb, edge_type=edge_type, edge_dir=edge_dir,
        )

        # Self-loop term.
        out = out + self.W_loop(x)
        out = out + self.bias

        # Update relation embeddings.
        rel_out = self.W_rel(rel_emb)

        return out, rel_out

    def message(self, x_j, rel_emb, edge_type, edge_dir):
        # x_j: [E, in_channels] — source node features per edge
        # Pick the relation embedding for this edge.
        r_j = rel_emb[edge_type]  # [E, in_channels]
        composed = _compose(x_j, r_j, self.composition)  # [E, in_channels]

        # Apply direction-typed weight: in vs out.
        # Build a single output tensor by indexing per edge.
        msg_in = self.W_in(composed)
        msg_out = self.W_out(composed)
        # Mix using edge_dir mask: where edge_dir==0 use msg_in, else msg_out.
        mask = edge_dir.view(-1, 1).to(msg_in.dtype)
        return msg_in * (1 - mask) + msg_out * mask


class CompGCN(nn.Module):
    def __init__(
        self,
        in_channels: Dict[NodeType, int],
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        dropout: float = 0.1,
        composition: str = "sub",
    ):
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.num_relations = len(self.edge_types)
        self.composition = composition
        self.dropout = dropout

        self._etype_to_id = {et: i for i, et in enumerate(self.edge_types)}

        # Per-type input projection to shared hidden_dim.
        self.lin_dict = nn.ModuleDict({
            nt: nn.Linear(in_channels[nt], hidden_dim) for nt in self.node_types
        })

        # Relation embeddings: 2 * R (original + inverse), each hidden_dim.
        self.rel_init = nn.Parameter(torch.empty(2 * self.num_relations, hidden_dim))
        nn.init.xavier_uniform_(self.rel_init)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            in_c = hidden_dim
            out_c = hidden_dim if layer < num_layers - 1 else out_dim
            self.layers.append(
                CompGCNLayer(in_c, out_c, self.num_relations,
                             composition=composition, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(out_c))

        self.decoder = Decoder(out_dim)

    def _homogenize(self, x_dict, edge_index_dict):
        projected = {nt: self.lin_dict[nt](x_dict[nt]) for nt in self.node_types if nt in x_dict}

        offsets = {}
        running = 0
        x_parts = []
        for nt in self.node_types:
            if nt not in projected:
                continue
            n = projected[nt].size(0)
            offsets[nt] = (running, running + n)
            x_parts.append(projected[nt])
            running += n
        x_cat = torch.cat(x_parts, dim=0)

        # Original directional edges.
        ei_parts = []
        et_parts = []
        for et, ei in edge_index_dict.items():
            if et not in self._etype_to_id:
                continue
            src_t, _, dst_t = et
            if src_t not in offsets or dst_t not in offsets:
                continue
            src_off, dst_off = offsets[src_t][0], offsets[dst_t][0]
            remapped = torch.stack([ei[0] + src_off, ei[1] + dst_off], dim=0)
            ei_parts.append(remapped)
            et_parts.append(
                torch.full((ei.size(1),), self._etype_to_id[et],
                           dtype=torch.long, device=ei.device)
            )
        if ei_parts:
            edge_index_orig = torch.cat(ei_parts, dim=1)
            edge_type_orig = torch.cat(et_parts, dim=0)
        else:
            edge_index_orig = torch.zeros((2, 0), dtype=torch.long, device=x_cat.device)
            edge_type_orig = torch.zeros((0,), dtype=torch.long, device=x_cat.device)

        # Inverse edges: swap row/col, shift relation id by num_relations.
        edge_index_inv = torch.stack([edge_index_orig[1], edge_index_orig[0]], dim=0)
        edge_type_inv = edge_type_orig + self.num_relations

        edge_index = torch.cat([edge_index_orig, edge_index_inv], dim=1)
        edge_type = torch.cat([edge_type_orig, edge_type_inv], dim=0)

        return x_cat, edge_index, edge_type, offsets

    def encode(self, x_dict, edge_index_dict):
        x, edge_index, edge_type, offsets = self._homogenize(x_dict, edge_index_dict)
        rel = self.rel_init

        for i, layer in enumerate(self.layers):
            x, rel = layer(x, rel, edge_index, edge_type)
            x = self.norms[i](x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return {nt: x[start:end] for nt, (start, end) in offsets.items()}

    def forward(
        self,
        x_dict,
        edge_index_dict,
        edge_label_index,
        src_type,
        dst_type,
        edge_time_dict=None,
        edge_feat_dict=None,
        edge_label_time=None,
        **kwargs,
    ):
        z_dict = self.encode(x_dict, edge_index_dict)
        z_src = z_dict[src_type][edge_label_index[0]]
        z_dst = z_dict[dst_type][edge_label_index[1]]
        return self.decoder(z_src, z_dst)

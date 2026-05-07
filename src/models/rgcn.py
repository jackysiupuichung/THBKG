"""Relational Graph Convolutional Network (R-GCN; Schlichtkrull et al. 2018).

Canonical R-GCN with basis decomposition. Heterogeneous input is projected
per-node-type to a shared hidden dimension, then homogenized into a single
graph with `edge_type` indices for `RGCNConv`. Output embeddings are split
back per-node-type for the link-prediction decoder.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

from .decoder import Decoder


NodeType = str
EdgeType = Tuple[str, str, str]


class RGCN(nn.Module):
    def __init__(
        self,
        in_channels: Dict[NodeType, int],
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        dropout: float = 0.1,
        num_bases: int | None = None,
    ):
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.num_relations = len(self.edge_types)
        self.dropout = dropout

        # Stable indexing: relation id == position in self.edge_types.
        self._etype_to_id = {et: i for i, et in enumerate(self.edge_types)}
        # Stable node-type index for split-after-conv.
        self._ntype_to_id = {nt: i for i, nt in enumerate(self.node_types)}

        # Per-type input projection to shared hidden_dim (so we can concatenate).
        self.lin_dict = nn.ModuleDict({
            nt: nn.Linear(in_channels[nt], hidden_dim) for nt in self.node_types
        })

        if num_bases is None:
            num_bases = max(2, self.num_relations // 2)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            in_c = hidden_dim
            out_c = hidden_dim if layer < num_layers - 1 else out_dim
            self.convs.append(
                RGCNConv(
                    in_channels=in_c,
                    out_channels=out_c,
                    num_relations=self.num_relations,
                    num_bases=num_bases,
                )
            )
            self.norms.append(nn.LayerNorm(out_c))

        self.decoder = Decoder(out_dim)

    def _homogenize(
        self,
        x_dict: Dict[NodeType, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[NodeType, Tuple[int, int]]]:
        """Concatenate node features into one tensor and remap edges to global ids.

        Returns
        -------
        x_cat : [N_total, hidden_dim]
        edge_index : [2, E_total] in global id space
        edge_type : [E_total]
        offsets : {node_type: (start, end)} for splitting the output back.
        """
        # Project each node type to hidden_dim so we can stack.
        projected = {nt: self.lin_dict[nt](x_dict[nt]) for nt in self.node_types if nt in x_dict}

        # Compute offsets in the order of self.node_types (stable across calls).
        offsets: Dict[NodeType, Tuple[int, int]] = {}
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

        # Remap edges to global node ids and stamp with relation id.
        ei_parts = []
        et_parts = []
        for et, ei in edge_index_dict.items():
            if et not in self._etype_to_id:
                continue
            src_t, _, dst_t = et
            if src_t not in offsets or dst_t not in offsets:
                continue
            src_off = offsets[src_t][0]
            dst_off = offsets[dst_t][0]
            remapped = torch.stack([ei[0] + src_off, ei[1] + dst_off], dim=0)
            ei_parts.append(remapped)
            et_parts.append(
                torch.full((ei.size(1),), self._etype_to_id[et],
                           dtype=torch.long, device=ei.device)
            )
        if ei_parts:
            edge_index = torch.cat(ei_parts, dim=1)
            edge_type = torch.cat(et_parts, dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=x_cat.device)
            edge_type = torch.zeros((0,), dtype=torch.long, device=x_cat.device)

        return x_cat, edge_index, edge_type, offsets

    def encode(
        self,
        x_dict: Dict[NodeType, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
    ) -> Dict[NodeType, torch.Tensor]:
        x, edge_index, edge_type, offsets = self._homogenize(x_dict, edge_index_dict)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            x = self.norms[i](x)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)

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

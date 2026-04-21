import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, HeteroConv, Linear
from .decoder import Decoder

class GATv2(nn.Module):
    """
    Static GATv2 model wrapped in HeteroConv.
    """
    def __init__(self, hidden_dim, out_dim, num_heads, num_layers=2, metadata=None, dropout=0.1, edge_feat_dim=0):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.edge_feat_dim = edge_feat_dim

        # Per-node-type input projection (like HGT)
        self.lin_dict = nn.ModuleDict({
            nt: Linear(-1, hidden_dim) for nt in self.node_types
        })

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: GATv2Conv(
                    -1, hidden_dim, heads=num_heads, dropout=dropout,
                    add_self_loops=False,
                    concat=False,  # average heads -> output stays at hidden_dim
                    edge_dim=edge_feat_dim if edge_feat_dim > 0 else None,
                )
                for edge_type in self.edge_types
            }, aggr='sum')
            self.convs.append(conv)
            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(hidden_dim) for nt in self.node_types
            }))

        self.decoder = Decoder(hidden_dim)

    def forward(
        self,
        x_dict,
        edge_index_dict,
        edge_label_index=None,
        src_type=None,
        dst_type=None,
        edge_time_dict=None,   # ignored in static GAT
        edge_feat_dict=None,
        **kwargs
    ):
        # 1. Message Passing (Encode)
        x_dict = self.encode(x_dict, edge_index_dict, edge_feat_dict=edge_feat_dict)

        # 2. Link Prediction (Decode)
        if edge_label_index is not None and src_type is not None:
            return self.decode(x_dict[src_type], x_dict[dst_type], edge_label_index)

        return x_dict

    def encode(self, x_dict, edge_index_dict, edge_feat_dict=None):
        # Project each node type to hidden_dim
        x_dict = {nt: self.lin_dict[nt](x).relu() for nt, x in x_dict.items() if nt in self.lin_dict}

        for i, conv in enumerate(self.convs):
            if self.edge_feat_dim > 0 and edge_feat_dict is not None:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_feat_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict)
            x_dict = {nt: torch.nan_to_num(self.norms[i][nt](x.relu())) for nt, x in x_dict.items() if nt in self.norms[i]}
        return x_dict

    def decode(self, z_src, z_dst, edge_label_index=None):
        if edge_label_index is not None:
            row, col = edge_label_index
            return self.decoder(z_src[row], z_dst[col])
        else:
            raise NotImplementedError("Full matrix decoding not yet implemented for GATv2")

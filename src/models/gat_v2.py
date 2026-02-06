import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, HeteroConv, Linear
from .decoder import DualHeadDecoder

class GATv2(nn.Module):
    """
    Static GATv2 model wrapped in HeteroConv.
    """
    def __init__(self, hidden_dim, out_dim, num_heads, num_layers=2, metadata=None, dropout=0.1):
        super().__init__()
        self.node_types, self.edge_types = metadata
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: GATv2Conv(-1, hidden_dim, heads=num_heads, dropout=dropout, add_self_loops=False)
                for edge_type in self.edge_types
            }, aggr='sum')
            self.convs.append(conv)
            
        # Dual heads for Multi-Task Probabilistic Learning
        self.decoder = DualHeadDecoder(hidden_dim)

    def forward(
        self, 
        x_dict, 
        edge_index_dict, 
        edge_label_index=None, 
        src_type=None, 
        dst_type=None, 
        edge_time_dict=None, # Ignored in static GAT
        **kwargs
    ):
        # 1. Message Passing (Encode)
        x_dict = self.encode(x_dict, edge_index_dict)
        
        # 2. Link Prediction (Decode)
        if edge_label_index is not None and src_type is not None:
            return self.decode(x_dict[src_type], x_dict[dst_type], edge_label_index)
            
        return x_dict

    def encode(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict

    def decode(self, z_src, z_dst, edge_label_index=None):
        if edge_label_index is not None:
            # Multi-head decoding on specific edges
            row, col = edge_label_index
            
            # Get node embeddings
            emb_src = z_src[row]
            emb_dst = z_dst[col]
            
            return self.decoder(emb_src, emb_dst)
        else:
            # Evaluation/Inference (returning embeddings or full matrix not easily supported with dual heads yet)
            # For now, we assume this usage is always paired with edge_label_index in our training loop.
            raise NotImplementedError("Full matrix decoding not yet implemented for dual-head GATv2")



import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, HeteroConv, Linear

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
        # Head A: Existence (Binary Discovery)
        self.lin_exist = Linear(-1, 1)
        
        # Head B: Probability (Calibrated Strength)
        self.lin_prob = Linear(-1, 1)

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
            
            # Head A: Existence (Binary)
            # Project to single scalar then dot product? 
            # Or concat and project?
            # Standard GAT link prediction typically assumes dot product of embeddings.
            # But here we want separate heads.
            # Option 1: Separate linear projections for the embeddings, then dot product.
            # Option 2: Hadamard product -> Linear.
            # To be consistent with "Linear(-1, 1)" definitions above, we apply linear transform to the interaction?
            # Actually, standard GAE does dot product.
            # If we want specific heads, we can project embeddings first or project the hadamard product.
            # Let's use Hadamard product -> Linear as it's more expressive for separate heads sharing the same backbone.
            
            # Hadamard product (element-wise multiplication) representing the edge interaction
            edge_feat = emb_src * emb_dst
            
            logits_exist = self.lin_exist(edge_feat).squeeze(-1)
            logits_prob = self.lin_prob(edge_feat).squeeze(-1)
            
            return {
                'logits_exist': logits_exist,
                'logits_prob': logits_prob
            }
        else:
            # Evaluation/Inference (returning embeddings or full matrix not easily supported with dual heads yet)
            # For now, we assume this usage is always paired with edge_label_index in our training loop.
            raise NotImplementedError("Full matrix decoding not yet implemented for dual-head GATv2")



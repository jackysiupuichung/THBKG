import torch
import torch.nn as nn
from torch.nn import Linear

class DualHeadDecoder(nn.Module):
    """
    Dual-head decoder for Multi-Task Link Prediction.
    
    Head A: Existence (Binary)
    Head B: Probability (Regression)
    
    Uses Hadamard product of node embeddings followed by separate linear projections.
    """
    def __init__(self, in_channels=-1):
        super().__init__()
        self.lin_exist = Linear(in_channels, 1)
        self.lin_prob = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        # Hadamard product
        # Ensure dimensions match (broadcasting if necessary, but usually aligned here)
        edge_feat = z_src * z_dst
        
        logits_exist = self.lin_exist(edge_feat).squeeze(-1)
        logits_prob = self.lin_prob(edge_feat).squeeze(-1)
        
        return {
            'logits_exist': logits_exist,
            'logits_prob': logits_prob
        }

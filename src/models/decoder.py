import torch
import torch.nn as nn
from torch.nn import Linear


class Decoder(nn.Module):
    """
    MLP decoder for link ranking.

    Concatenates source and destination node embeddings, passes through a
    reverse-pyramid MLP, and outputs a single unbounded ranking score (logit).

    Architecture: [2*in_channels] -> [in_channels] -> [in_channels//2] -> [1]
    """
    def __init__(self, in_channels=-1, dropout=0.1, time_dim: int = 0):
        super().__init__()
        self.time_dim = time_dim

        mlp_in = 2 * in_channels + time_dim
        self.mlp = nn.Sequential(
            Linear(mlp_in, in_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(in_channels // 2, 1),
        )

    def forward(self, z_src, z_dst, t_emb=None):
        if t_emb is not None:
            edge_feat = torch.cat([z_src, z_dst, t_emb], dim=-1)
        else:
            edge_feat = torch.cat([z_src, z_dst], dim=-1)
        return self.mlp(edge_feat).squeeze(-1)

import torch
from torch import nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, hidden_dims=[128, 64],
                 user_emb=None, item_emb=None):
        super().__init__()

        # user embedding
        self.user_emb = nn.Embedding(num_users, embed_dim)
        if user_emb is not None:
            if isinstance(user_emb, torch.Tensor):
                self.user_emb.weight.data.copy_(user_emb)
            else:
                raise ValueError("user_emb must be a torch.Tensor")

        # item embedding
        self.item_emb = nn.Embedding(num_items, embed_dim)
        if item_emb is not None:
            if isinstance(item_emb, torch.Tensor):
                self.item_emb.weight.data.copy_(item_emb)
            else:
                raise ValueError("item_emb must be a torch.Tensor")

        # MLP layers
        layers = []
        input_dim = embed_dim * 2
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user, item):
        u = self.user_emb(user)
        v = self.item_emb(item)
        x = torch.cat([u, v], dim=-1)
        return self.mlp(x)

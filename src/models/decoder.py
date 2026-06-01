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


class BilinearDecoder(nn.Module):
    """Bilinear decoder: score(t, d) = z_t^T W z_d + bias.

    Designed to break the disease-collapse failure mode of the MLP decoder.
    By construction the score is multiplicative in both endpoints — there
    is no way to write it as `f(d) + g(t)` for arbitrary functions, so the
    model cannot output a near-constant score for all targets sharing a
    disease.

    A `time_dim` channel can be added via a separate linear head; the time
    embedding contributes additively (it's not a node).

    Parameters
    ----------
    in_channels : hidden dim of the encoder outputs.
    dropout : applied to z_src and z_dst before the bilinear product.
    time_dim : optional dim of the time embedding; 0 disables.
    rank : if 0, use a full `[in_channels, in_channels]` matrix.
           If r > 0, use a low-rank factorisation W = U V^T with
           U, V ∈ R^{in_channels × r}, which reduces parameters from
           in_channels^2 to 2 * in_channels * r (useful for regularisation).
    """

    def __init__(self, in_channels=-1, dropout=0.1, time_dim: int = 0, rank: int = 0):
        super().__init__()
        assert in_channels > 0, "in_channels must be set"
        self.in_channels = in_channels
        self.time_dim = time_dim
        self.rank = rank
        self.dropout = nn.Dropout(dropout)

        if rank > 0:
            self.U = nn.Parameter(torch.empty(in_channels, rank))
            self.V = nn.Parameter(torch.empty(in_channels, rank))
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)
        else:
            self.W = nn.Parameter(torch.empty(in_channels, in_channels))
            nn.init.xavier_uniform_(self.W)

        self.bias = nn.Parameter(torch.zeros(1))

        if time_dim > 0:
            # Time-only contribution. Kept additive: this is the one
            # exception to "fully multiplicative" — needed because time
            # is a per-edge property, not a node.
            self.time_head = Linear(time_dim, 1)
        else:
            self.time_head = None

    def forward(self, z_src, z_dst, t_emb=None):
        z_src = self.dropout(z_src)
        z_dst = self.dropout(z_dst)
        if self.rank > 0:
            # score = (z_t U) · (z_d V)
            score = (z_src @ self.U * (z_dst @ self.V)).sum(dim=-1)
        else:
            # score = z_t^T W z_d, computed as (z_t W) · z_d
            score = (z_src @ self.W * z_dst).sum(dim=-1)
        score = score + self.bias
        if t_emb is not None and self.time_head is not None:
            score = score + self.time_head(t_emb).squeeze(-1)
        return score


def build_decoder(kind: str = "mlp", **kwargs) -> nn.Module:
    """Factory: kind ∈ {"mlp", "bilinear", "bilinear_lr<R>"}.

    Examples:
        build_decoder("mlp", in_channels=128, dropout=0.1)
        build_decoder("bilinear", in_channels=128, dropout=0.1)
        build_decoder("bilinear_lr16", in_channels=128, dropout=0.1)  # rank=16
    """
    kind = (kind or "mlp").lower()
    if kind == "mlp":
        return Decoder(**kwargs)
    if kind == "bilinear":
        return BilinearDecoder(rank=0, **kwargs)
    if kind.startswith("bilinear_lr"):
        try:
            r = int(kind[len("bilinear_lr"):])
        except ValueError:
            raise ValueError(f"could not parse rank from {kind!r}")
        return BilinearDecoder(rank=r, **kwargs)
    raise ValueError(f"unknown decoder kind {kind!r}")

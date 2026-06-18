"""Differentiable per-edge mask for PaGE-Link-style explanation (SCAFFOLD).

PaGE-Link stage 1 learns a soft mask m_e in [0,1] over the subgraph edges by
optimising the masked subgraph's ability to reproduce the model's link logit.
This module provides:

  * EdgeMask        -- learnable per-edge-type logits, sigmoid -> m_e.
  * inject into the EAHGT HGTConv message passing WITHOUT forking the conv, by
    multiplying m_e into the same per-edge channel the conv already uses for
    edge_attr (see src/models/hgt_conv_rte.py, where alpha is scaled by a
    per-edge scalar). We pass m_e as an extra multiplicative edge_feat so the
    gradient flows to the mask while the model weights stay frozen.

See note/explain_pagelink.md. STATUS: scaffold — signatures + docstrings only.
"""

from __future__ import annotations

from typing import Dict, Tuple

EdgeType = Tuple[str, str, str]


class EdgeMask:
    """Learnable per-edge mask over a single pair's subgraph.

    Holds one logit tensor per edge type (shape [E_t]); ``values()`` returns
    sigmoid(logits) as the multiplicative mask m_e. Only these logits require
    grad — the explained model is frozen.
    """

    def __init__(self, edge_counts: Dict[EdgeType, int], init: float = 0.0):
        # TODO(impl): self.logits = {et: torch.full((n,), init, requires_grad=True)
        #             for et, n in edge_counts.items()}
        raise NotImplementedError("scaffold: EdgeMask params (see note)")

    def values(self) -> Dict[EdgeType, "object"]:
        """{edge_type: sigmoid(logit) tensor in [0,1]}."""
        raise NotImplementedError("scaffold")

    def regularisation(self, size_coeff: float, entropy_coeff: float) -> "object":
        """GNNExplainer-style penalty: size (sum m_e) + entropy (per-edge
        -m log m - (1-m) log(1-m)), to drive the mask sparse and near-binary."""
        raise NotImplementedError("scaffold")


def masked_forward(model, batch, mask: EdgeMask, edge_feat_cols, edge_time_dict):
    """Run model.forward on the subgraph with m_e multiplied into message
    passing, returning the query-edge logit.

    Injection: combine the existing edge_feat_dict with the mask so the conv's
    per-edge scalar becomes edge_attr * m_e. Call signature mirrors
    explain_advancement.py (src_type='target', dst_type='disease', ...).

    TODO(impl): build the masked edge_feat_dict; single forward; return logit.
    """
    raise NotImplementedError("scaffold: mask injection + forward (see note)")

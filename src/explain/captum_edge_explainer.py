"""Captum IntegratedGradients attribution for HGT advancement link prediction.

Attributes against TWO classes of inputs in a single IG pass:
  1. Per-edge-type edge features ``[E_t, edge_feat_dim]`` — the model's
     stored (score, novelty) signal — with a zero-feature baseline.
     Answers: "how much do this edge's (score, novelty) values contribute?"
  2. Per-node-type node features ``[N_t, F_t]`` (x_dict) — with a
     zero-feature baseline. Answers: "how much do this node's intrinsic
     features contribute?"

Both share a common IG framework: interpolate linearly from baseline to
real input in n_steps, average gradient × input-delta. By driving both at
once captum returns attributions in a consistent units system (delta in
the same model logit space).

Why not PyG's ``CaptumExplainer``? Its heterogeneous + link-prediction
path is fragile (see PyG discussions #7738 / #7963). Calling captum
directly against a thin adapter is cleaner: we own which tensors are
leaves, the [N,1]→[N,2] reshape, and the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from captum.attr import IntegratedGradients

EdgeType = Tuple[str, str, str]
ADV_ETYPE: EdgeType = ("target", "advancement", "disease")


@dataclass
class PairAttribution:
    """IG attributions for one (target, disease) query.

    edge_feat_attr: ``{edge_type: tensor[E_t, edge_feat_dim]}`` for edge
        types that carry features. Signed floats.
    node_feat_attr: ``{node_type: tensor[N_t, F_t]}`` for every node type.
        Signed floats.
    edge_index_dict: matching edge indices for every edge type, [2, E_t].
    logit: model logit on the queried (target, disease) edge.
    """
    edge_feat_attr: Dict[EdgeType, torch.Tensor]
    node_feat_attr: Dict[str, torch.Tensor]
    edge_index_dict: Dict[EdgeType, torch.Tensor]
    logit: float


class _BatchForwardAdapter(torch.nn.Module):
    """Differentiable wrapper around HGTLinkPredictor.

    Captum hands inputs in a flat positional tuple ``(edge_feats..., node_xs...)``
    in fixed order, all shaped with a leading batch dim of size B (the IG
    interpolation chunk). We loop over B and run the model once per step,
    stacking the queried logit so captum sees ``[B, 2]``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        batch,
        feat_edge_types: List[EdgeType],
        node_types: List[str],
        edge_time_dict: Optional[Dict[EdgeType, torch.Tensor]],
        adv_etype: EdgeType = ADV_ETYPE,
    ):
        super().__init__()
        self.model = model
        self.batch = batch
        self.feat_edge_types = list(feat_edge_types)
        self.node_types = list(node_types)
        self.edge_time_dict = edge_time_dict
        self.adv_etype = adv_etype
        self._n_edge_inputs = len(self.feat_edge_types)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        # Split the positional tuple back into edge-feature inputs and
        # node-feature inputs (same order they were passed in).
        feat_tensors = inputs[: self._n_edge_inputs]
        x_tensors = inputs[self._n_edge_inputs :]

        B = inputs[0].size(0)
        edge_label_index = self.batch[self.adv_etype].edge_label_index
        edge_label_time = getattr(self.batch[self.adv_etype], "edge_label_time", None)

        logits = []
        for b in range(B):
            edge_feat_dict = {
                et: feat[b] for et, feat in zip(self.feat_edge_types, feat_tensors)
            }
            x_dict = {
                nt: x[b] for nt, x in zip(self.node_types, x_tensors)
            }
            out = self.model(
                x_dict,
                self.batch.edge_index_dict,
                edge_label_index,
                src_type=self.adv_etype[0],
                dst_type=self.adv_etype[2],
                edge_time_dict=self.edge_time_dict,
                edge_feat_dict=edge_feat_dict,
                edge_label_time=edge_label_time,
            )
            logits.append(out[0])
        logit = torch.stack(logits, dim=0)        # [B]
        return torch.stack([-logit, logit], dim=-1)  # [B, 2]


def _gather_feat_inputs(
    batch,
    edge_feat_cols: List[int],
    adv_etype: EdgeType,
) -> Tuple[List[EdgeType], List[torch.Tensor]]:
    """Per-edge-type slice of edge_attr[:, edge_feat_cols] as a captum leaf,
    shaped ``[1, E_t, len(edge_feat_cols)]``."""
    feat_etypes: List[EdgeType] = []
    feats: List[torch.Tensor] = []
    for et in batch.edge_types:
        if et == adv_etype:
            continue
        store = batch[et]
        ea = getattr(store, "edge_attr", None)
        if ea is None:
            continue
        feat = ea[:, edge_feat_cols].detach().float().clone().unsqueeze(0)
        feat = feat.requires_grad_(True)
        feat_etypes.append(et)
        feats.append(feat)
    return feat_etypes, feats


def _gather_node_inputs(
    batch,
) -> Tuple[List[str], List[torch.Tensor]]:
    """Per-node-type x as a captum leaf, shaped ``[1, N_t, F_t]``."""
    node_types: List[str] = []
    xs: List[torch.Tensor] = []
    for nt in batch.node_types:
        store = batch[nt]
        x = getattr(store, "x", None)
        if x is None:
            continue
        leaf = x.detach().float().clone().unsqueeze(0).requires_grad_(True)
        node_types.append(nt)
        xs.append(leaf)
    return node_types, xs


def integrated_gradients_for_pair(
    model: torch.nn.Module,
    batch,
    edge_feat_cols: List[int],
    edge_time_dict: Optional[Dict[EdgeType, torch.Tensor]],
    n_steps: int = 32,
    adv_etype: EdgeType = ADV_ETYPE,
) -> PairAttribution:
    """Run IntegratedGradients on a single (target, disease) query, against
    both per-edge-type edge features AND per-node-type node features.

    ``batch`` must be a single subgraph from ``LinkNeighborLoader``
    (batch_size=1; ``batch[ADV_ETYPE].edge_label_index`` is [2, 1]).

    Baseline = zeros for every attributed tensor — "no signal" reference.
    """
    model.eval()

    feat_etypes, feat_inputs = _gather_feat_inputs(batch, edge_feat_cols, adv_etype)
    if not feat_inputs:
        raise RuntimeError("No edge types with edge_attr — nothing to attribute.")
    # Drop empty edge-type tensors. They confuse captum's batching (which
    # treats inputs[0].shape[0] as num_examples → ZeroDivisionError) and
    # contribute no information.
    nonempty = [(et, t) for et, t in zip(feat_etypes, feat_inputs) if t.size(1) > 0]
    if not nonempty:
        raise RuntimeError("All featured edge types are empty in this batch.")
    feat_etypes = [et for et, _ in nonempty]
    feat_inputs = [t for _, t in nonempty]

    node_types, node_inputs = _gather_node_inputs(batch)
    # Also drop empty node-type tensors (a sampled subgraph might miss e.g.
    # `reactome` entirely).
    nonempty_n = [(nt, t) for nt, t in zip(node_types, node_inputs) if t.size(1) > 0]
    node_types = [nt for nt, _ in nonempty_n]
    node_inputs = [t for _, t in nonempty_n]

    adapter = _BatchForwardAdapter(
        model=model,
        batch=batch,
        feat_edge_types=feat_etypes,
        node_types=node_types,
        edge_time_dict=edge_time_dict,
        adv_etype=adv_etype,
    )

    all_inputs = tuple(feat_inputs) + tuple(node_inputs)

    # Sanity logit (no gradients needed)
    with torch.no_grad():
        ref_logits = adapter(*all_inputs)  # [1, 2]
    logit_val = float(ref_logits[0, 1].item())

    ig = IntegratedGradients(adapter)
    baselines = tuple(torch.zeros_like(t) for t in all_inputs)

    attrs = ig.attribute(
        inputs=all_inputs,
        baselines=baselines,
        target=1,
        n_steps=n_steps,
    )
    if isinstance(attrs, torch.Tensor):
        attrs = (attrs,)

    # Split attribution tuple back into edge vs node groups in the same
    # order they were appended above.
    n_edge = len(feat_inputs)
    edge_attrs = attrs[:n_edge]
    node_attrs = attrs[n_edge:]

    edge_feat_attr = {
        et: a.detach().squeeze(0).cpu()
        for et, a in zip(feat_etypes, edge_attrs)
    }
    node_feat_attr = {
        nt: a.detach().squeeze(0).cpu()
        for nt, a in zip(node_types, node_attrs)
    }
    edge_index_dict = {et: ei.detach().cpu() for et, ei in batch.edge_index_dict.items()}

    return PairAttribution(
        edge_feat_attr=edge_feat_attr,
        node_feat_attr=node_feat_attr,
        edge_index_dict=edge_index_dict,
        logit=logit_val,
    )

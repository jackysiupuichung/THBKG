"""Per-pair edge-attribution tables.

NOTE on scope: this module deliberately exposes *instance-level* attributions
only. An earlier version produced a population-level `relation_importance`
table via mean-|IG| over all edges, which the literature flags as a poor
global summary (Integrated Gradients sum to f(x)−f(baseline) per instance,
so naive averaging mixes "magnitude of prediction" with "share of
attribution" and is biased by sample size per edge type). See
https://arxiv.org/abs/2404.13910 (Integrated Gradient Correlation).

If you need a population-level rollup, design it explicitly (e.g.
attribution-share normalised within each pair, or IGC against the per-pair
logit) — don't put it back here as an unflagged mean.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

EdgeType = Tuple[str, str, str]


def per_pair_nodes_df(
    target_id: str,
    disease_id: str,
    node_feat_attr: Dict[str, torch.Tensor],
    n_id_dict: Dict[str, "np.ndarray"],
    logit: float,
) -> pd.DataFrame:
    """One row per node in the queried subgraph.

    Per-node scalar attribution = sum over feature dimensions (signed),
    plus its absolute-value sibling. Use ``ig_node_total`` for ranking
    (the absolute version is the standard for "node importance"). The
    raw [N_t, F_t] tensor is kept inside the PairAttribution object for
    callers who need feature-dim level inspection.
    """
    rows = []
    for nt, attr in node_feat_attr.items():
        a = attr.cpu().numpy()                       # [N_t, F_t]
        n_id = n_id_dict.get(nt)
        signed_total = a.sum(axis=1)                 # [N_t]
        abs_total = np.abs(a).sum(axis=1)            # [N_t]
        for i in range(a.shape[0]):
            rows.append({
                "target_id": target_id,
                "disease_id": disease_id,
                "node_type": nt,
                "node_local_idx": i,
                "node_global_idx": int(n_id[i]) if n_id is not None else i,
                "ig_node_signed": float(signed_total[i]),
                "ig_node_abs":    float(abs_total[i]),
                "logit": logit,
            })
    return pd.DataFrame(rows)


def per_pair_edges_df(
    target_id: str,
    disease_id: str,
    edge_index_dict: Dict[EdgeType, torch.Tensor],
    edge_feat_attr: Dict[EdgeType, torch.Tensor],
    attention: Dict[EdgeType, torch.Tensor],
    logit: float,
    feat_col_names: List[str],
) -> pd.DataFrame:
    """One row per edge in the queried subgraph.

    feat_col_names: human-readable names for the edge_feat_cols (e.g.
    ['ig_score', 'ig_novelty'] when edge_feat_cols=[0,1]).
    """
    rows = []
    for et, ei in edge_index_dict.items():
        ei = ei.cpu().numpy()
        n = ei.shape[1]
        attn = attention.get(et)
        attn_arr = attn.cpu().numpy() if attn is not None else np.full(n, np.nan)
        ig = edge_feat_attr.get(et)
        if ig is not None:
            ig_arr = ig.cpu().numpy()       # [n, F]
            ig_total = ig_arr.sum(axis=1)
        else:
            ig_arr = None
            ig_total = np.full(n, np.nan)

        for i in range(n):
            row = {
                "target_id": target_id,
                "disease_id": disease_id,
                "edge_type": "::".join(et),
                "src": int(ei[0, i]),
                "dst": int(ei[1, i]),
                "ig_total": float(ig_total[i]),
                "attention": float(attn_arr[i]) if attn_arr[i] == attn_arr[i] else np.nan,
                "logit": logit,
            }
            for j, name in enumerate(feat_col_names):
                row[name] = float(ig_arr[i, j]) if ig_arr is not None else np.nan
            rows.append(row)
    return pd.DataFrame(rows)

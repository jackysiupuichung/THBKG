"""T3 (advancement) metrics: relative success of the top-N pairs per therapeutic area.

Relative success at N (RS@N) is the advancement rate among a model's N
top-scored pairs over the rate among the rest, within one therapeutic area
(Czech et al. 2024). Pairs tied at the cut-off count at their expected share:
if the N-th place falls inside a block of tied scores, each pair of the block
is in the top N with probability (places left) / (block size). So a constant
score has RS = 1 exactly, and the result does not depend on row order.

Reported per primary therapeutic area (labels/primary_therapeutic_areas.json),
as the equally weighted mean over them ("TA mean", "all" excluded), for "all"
pooled, and per entry year (AUROC / average precision), because the positive
rate falls with entry year (later entries have less follow-up).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NS = (10, 50, 100)


def relative_success(scores: np.ndarray, outcome: np.ndarray, n: int) -> float:
    """RS@n with ties at the cut-off counted at their expected share; NaN if undefined."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(outcome, dtype=bool)
    m = scores.size
    if n >= m or y.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    s, yo = scores[order], y[order]
    kth = s[n - 1]
    above = s > kth
    tied = s == kth
    k_above = int(above.sum())
    pos_top = yo[above].sum() + (n - k_above) * yo[tied].sum() / tied.sum()
    rate_top = pos_top / n
    rate_rest = (y.sum() - pos_top) / (m - n)
    return float(rate_top / rate_rest) if rate_rest > 0 else float("nan")


def _auroc_ap(scores, outcome) -> tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(outcome, dtype=bool)
    if y.all() or not y.any():
        return float("nan"), float("nan")
    return float(roc_auc_score(y, scores)), float(average_precision_score(y, scores))


def evaluate_advancement(pred: pd.DataFrame, tas: pd.DataFrame, primary: list[str], ns=NS) -> dict[str, pd.DataFrame]:
    """Metrics for one model on one split.

    pred: target_id, disease_id, transition_year, outcome, score (one row per pair).
    Returns {"by_ta": per area and N, "summary": TA mean and "all" per N,
    "by_year": per entry year}.
    """
    merged = pred.merge(tas[["disease_id", "therapeutic_area_name"]], on="disease_id")
    rows = []
    for ta in primary:
        g = merged[merged["therapeutic_area_name"] == ta]
        auroc, ap = _auroc_ap(g["score"], g["outcome"])
        for n in ns:
            rows.append({"therapeutic_area": ta, "N": n, "n_pairs": len(g), "n_pos": int(g["outcome"].sum()),
                         "RS": relative_success(g["score"].to_numpy(), g["outcome"].to_numpy(), n),
                         "auroc": auroc, "ap": ap})
    by_ta = pd.DataFrame(rows)
    areas = by_ta[by_ta["therapeutic_area"] != "all"]
    summary = pd.concat([
        areas.groupby("N")[["RS", "auroc", "ap"]].mean().assign(scope="TA mean"),
        by_ta[by_ta["therapeutic_area"] == "all"].set_index("N")[["RS", "auroc", "ap"]].assign(scope="all"),
    ]).reset_index()
    years = []
    for y, g in pred.groupby("transition_year"):
        auroc, ap = _auroc_ap(g["score"], g["outcome"])
        years.append({"transition_year": int(y), "n_pairs": len(g), "positive_rate": float(g["outcome"].mean()),
                      "auroc": auroc, "ap": ap})
    return {"by_ta": by_ta, "summary": summary, "by_year": pd.DataFrame(years)}

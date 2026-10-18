"""Slate construction and ranking metrics for the T1 and T2 benchmark tasks.

The unit of evaluation is a *slate*: one ``(disease, decision year)`` ranking
problem over the task's full candidate pool, with no negative sampling. Every
slate gets one vote in the macro average, so heavily trialled indications do
not dominate. Ties are broken pessimistically (positives rank last within a
tie), so a constant score cannot look better than random by index order.

Protocol carried over from ~/tempo_THBKG/thbkg/evaluate.py, generalised from
targets to any candidate type.

Usage::

    labels = build_discovery_labels(release_dir)
    slates = discovery_slates(labels, decision_years("test"), num_targets)
    summary, per_slate = evaluate(slates, lambda s: model_scores(s.year, s.disease_index, s.candidates))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

KS = (10, 20, 50, 100, 200)


@dataclass
class Slate:
    """One (disease, decision year) ranking problem."""

    task: str
    year: int
    disease_index: int
    candidates: np.ndarray     # [C] candidate node indices (targets for T1, molecules for T2)
    positive_mask: np.ndarray  # [C] bool


def _slates(task: str, pairs: pd.DataFrame, cand_col: str, years: Iterable[int],
            universe_for_year: Callable[[int], np.ndarray]) -> list[Slate]:
    """Shared builder: ``pairs`` has disease_index, ``cand_col``, first_trial_year."""
    first = pairs["first_trial_year"].to_numpy()
    out: list[Slate] = []
    for year in years:
        universe = universe_for_year(year)
        pos = pairs[first == year]
        if pos.empty or universe.size == 0:
            continue
        prior = pairs[first < year]
        prior_by_dis = prior.groupby("disease_index")[cand_col].apply(np.asarray).to_dict()
        for d, g in pos.groupby("disease_index"):
            excluded = prior_by_dis.get(d)
            cands = universe if excluded is None else universe[~np.isin(universe, excluded)]
            pmask = np.isin(cands, g[cand_col].to_numpy())
            if pmask.any():  # every positive outside the pool -> nothing to rank
                out.append(Slate(task, int(year), int(d), cands, pmask))
    return out


def discovery_slates(labels: pd.DataFrame, years: Iterable[int], num_targets: int) -> list[Slate]:
    """T1 slates: all targets minus those with a trial for the disease before ``Y``."""
    all_targets = np.arange(num_targets, dtype=np.int64)
    return _slates("discovery", labels, "target_index", years, lambda y: all_targets)


def repurposing_slates(labels: pd.DataFrame, years: Iterable[int]) -> list[Slate]:
    """T2 slates: drugs with any trial before ``Y`` and none for the disease before ``Y``.

    Restricting the pool to drugs with trial history makes "drug seen before"
    hold for negatives as well as positives; otherwise every first-in-human drug
    would be a free negative that a model could rank last without learning
    anything about the disease.
    """
    drug_first = labels.groupby("molecule_index")["drug_first_trial_year"].first()
    ids, first = drug_first.index.to_numpy(np.int64), drug_first.to_numpy()

    def universe(year: int) -> np.ndarray:
        return np.sort(ids[first < year])

    return _slates("repurposing", labels, "molecule_index", years, universe)


def _dcg(gains: np.ndarray) -> float:
    return float((gains / np.log2(np.arange(gains.size) + 2.0)).sum())


def slate_metrics(scores: np.ndarray, positive_mask: np.ndarray, ks=KS) -> dict:
    """precision@K / recall@K / NDCG@K for one slate, with pessimistic ties."""
    scores = np.asarray(scores, dtype=np.float64)
    positive_mask = np.asarray(positive_mask, dtype=bool)
    if scores.shape != positive_mask.shape:
        raise ValueError(f"scores {scores.shape} != candidates {positive_mask.shape}")
    # lexsort: last key is primary -> score descending, then negatives before positives
    order = np.lexsort((positive_mask.astype(np.int8), -scores))
    ranked = positive_mask[order].astype(np.float64)
    n_pos = int(positive_mask.sum())

    out = {}
    for k in ks:
        ke = min(k, ranked.size)
        hits = ranked[:ke].sum()
        out[f"precision@{k}"] = hits / ke
        out[f"recall@{k}"] = hits / n_pos
        out[f"ndcg@{k}"] = _dcg(ranked[:ke]) / _dcg(np.ones(min(n_pos, ke)))
    out["n_pos"] = n_pos
    out["n_cand"] = int(ranked.size)
    return out


def macro_average(per_slate: list[dict], ks=KS) -> dict:
    n = len(per_slate)
    if n == 0:
        raise ValueError("no slates to average")
    agg = {f"{m}@{k}": float(np.mean([s[f"{m}@{k}"] for s in per_slate]))
           for k in ks for m in ("precision", "recall", "ndcg")}
    agg["n_slates"] = n
    agg["mean_n_pos"] = float(np.mean([s["n_pos"] for s in per_slate]))
    agg["mean_n_cand"] = float(np.mean([s["n_cand"] for s in per_slate]))
    return agg


def evaluate(slates: list[Slate], score_fn: Callable[[Slate], np.ndarray], ks=KS) -> tuple[dict, list[dict]]:
    """Score every slate with ``score_fn`` (one score per ``slate.candidates``) and macro-average."""
    per_slate = []
    for s in slates:
        m = slate_metrics(score_fn(s), s.positive_mask, ks)
        m.update(year=s.year, disease_index=s.disease_index)
        per_slate.append(m)
    return macro_average(per_slate, ks), per_slate


def slate_stats(slates: list[Slate]) -> pd.DataFrame:
    """One row per slate (year, disease_index, n_pos, n_cand), without the candidate arrays."""
    return pd.DataFrame({
        "year": [s.year for s in slates],
        "disease_index": [s.disease_index for s in slates],
        "n_pos": [int(s.positive_mask.sum()) for s in slates],
        "n_cand": [int(s.candidates.size) for s in slates],
    })


def expected_random(stats: pd.DataFrame, ks=KS) -> dict:
    """Closed-form expectation of a uniformly random ranking, from ``slate_stats``.

    precision@K = P/C and recall@K = K/C per slate, macro-averaged. A floor to
    read results against: with ~2 positives among ~19,600 targets, absolute
    precision@K is tiny for any model.
    """
    agg = {}
    for k in ks:
        agg[f"precision@{k}"] = float((stats["n_pos"] / stats["n_cand"]).mean())
        agg[f"recall@{k}"] = float((np.minimum(k, stats["n_cand"]) / stats["n_cand"]).mean())
    agg["n_slates"] = len(stats)
    return agg


def slate_summary(stats: pd.DataFrame) -> dict:
    """Size of a split, from ``slate_stats``."""
    if stats.empty:
        return {"n_slates": 0}
    return {
        "n_slates": len(stats),
        "n_diseases": int(stats["disease_index"].nunique()),
        "n_positives": int(stats["n_pos"].sum()),
        "median_pos_per_slate": float(stats["n_pos"].median()),
        "mean_candidates": float(stats["n_cand"].mean()),
        "slates_per_year": {int(y): int(c) for y, c in stats.groupby("year").size().items()},
    }

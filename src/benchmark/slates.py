"""Slate construction and ranking metrics for the T1 discovery and T2 hypothesis tasks.

The unit of evaluation is a *slate*: one ``(disease, decision year)`` ranking
problem over the task's full candidate pool, with no negative sampling. Every
slate gets one vote in the macro average, so heavily trialled indications do
not dominate. Ties are broken pessimistically (positives rank last within a
tie, lowest grade first), so a constant score cannot look better than random by
index order.

Precision and recall treat every positive alike. NDCG uses graded relevance: a
positive's gain is its ``grade`` label column (T1: max outcome phase, see
tasks.py), so ranking a pair that went on to Phase 3 above one that stopped at
Phase 1 scores higher. Without a ``grade`` column every positive has gain 1.

A slate can also carry *strata* (T2): named subsets of its positives. For each,
recall@K counts the stratum's positives in the top K of the full slate, so the
other positives still compete for those places.

Protocol carried over from ~/tempo_THBKG/thbkg/evaluate.py.

Usage (``task_slates`` in tasks.py applies the split years and slate filter;
``evaluate_task`` there runs a whole split one year at a time)::

    labels = build_discovery_labels(release_dir)
    slates = task_slates("discovery", "test", labels, num_targets=num_targets)
    summary, per_slate = evaluate(slates, lambda s: model_scores(s.year, s.disease_index, s.candidates))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

KS = (100, 200, 500)


@dataclass
class Slate:
    """One (disease, decision year) ranking problem."""

    task: str
    year: int
    disease_index: int
    candidates: np.ndarray     # [C] candidate target indices
    positive_mask: np.ndarray  # [C] bool
    gain: np.ndarray           # [C] float: the positive's grade, 0 for negatives
    strata: dict[str, np.ndarray] = field(default_factory=dict)  # name -> [C] bool, True on its positives


def _slates(task: str, pairs: pd.DataFrame, cand_col: str, years: Iterable[int],
            universe_for_year: Callable[[int], np.ndarray], year_col: str = "first_trial_year",
            strata: Iterable[str] = ()) -> list[Slate]:
    """Shared builder: ``pairs`` has disease_index, ``cand_col``, ``year_col``
    and optionally ``grade`` and ``rankable``: a pair with rankable False is
    left out of its year's slate, neither positive nor candidate. ``strata``
    names bool columns of ``pairs``, read on the positives."""
    first = pairs[year_col].to_numpy()
    has_grade = "grade" in pairs.columns
    rankable = pairs["rankable"].to_numpy(bool) if "rankable" in pairs.columns else np.ones(len(pairs), bool)
    strata = list(strata)
    out: list[Slate] = []
    for year in years:
        universe = universe_for_year(year)
        pos = pairs[(first == year) & rankable]
        if pos.empty or universe.size == 0:
            continue
        # earlier pairs, and this year's unrankable ones, leave the pool
        prior = pairs[(first < year) | ((first == year) & ~rankable)]
        prior_by_dis = prior.groupby("disease_index")[cand_col].apply(np.asarray).to_dict()
        for d, g in pos.groupby("disease_index"):
            excluded = prior_by_dis.get(d)
            cands = universe if excluded is None else universe[~np.isin(universe, excluded)]
            pos_ids = g[cand_col].to_numpy()
            pos_gain = g["grade"].to_numpy(np.float64) if has_grade else np.ones(len(g))
            order = np.argsort(pos_ids)
            at = np.searchsorted(pos_ids[order], cands).clip(max=len(order) - 1)
            pmask = pos_ids[order][at] == cands
            if pmask.any():  # every positive outside the pool -> nothing to rank
                gain = np.where(pmask, pos_gain[order][at], 0.0)
                masks = {k: pmask & g[k].to_numpy(bool)[order][at] for k in strata}
                out.append(Slate(task, int(year), int(d), cands, pmask, gain, masks))
    return out


def discovery_slates(labels: pd.DataFrame, years: Iterable[int], num_targets: int) -> list[Slate]:
    """T1 slates: all targets minus those with a trial for the disease before ``Y``."""
    all_targets = np.arange(num_targets, dtype=np.int64)
    return _slates("discovery", labels, "target_index", years, lambda y: all_targets)


def hypothesis_slates(labels: pd.DataFrame, years: Iterable[int], num_targets: int,
                      strata: Iterable[str] = ()) -> list[Slate]:
    """T2 slates: all targets minus those with a direct edge of any relation to the disease before ``Y``."""
    all_targets = np.arange(num_targets, dtype=np.int64)
    return _slates("hypothesis", labels, "target_index", years, lambda y: all_targets,
                   year_col="first_direct_year", strata=strata)


def _dcg(gains: np.ndarray) -> float:
    return float((gains / np.log2(np.arange(gains.size) + 2.0)).sum())


def slate_metrics(scores: np.ndarray, positive_mask: np.ndarray, ks=KS, gains=None, strata=None) -> dict:
    """precision@K / recall@K / NDCG@K for one slate, with pessimistic ties.

    ``gains`` (default: 1 per positive) are the graded relevances used by NDCG;
    precision and recall only use ``positive_mask``. ``strata`` (name -> bool
    mask, a subset of the positives) adds ``<name>.recall@K`` and
    ``<name>.n_pos``; the recall is NaN when the stratum has no positive here.
    """
    scores = np.asarray(scores, dtype=np.float64)
    positive_mask = np.asarray(positive_mask, dtype=bool)
    if scores.shape != positive_mask.shape:
        raise ValueError(f"scores {scores.shape} != candidates {positive_mask.shape}")
    gains = positive_mask.astype(np.float64) if gains is None else np.asarray(gains, dtype=np.float64)
    if gains.shape != positive_mask.shape or ((gains > 0) != positive_mask).any():
        raise ValueError("gains must be > 0 exactly on the positives")
    # lexsort: last key is primary -> score descending, then lowest gain first
    # (negatives, then low-grade positives) within a tie
    order = np.lexsort((gains, -scores))
    ranked = positive_mask[order].astype(np.float64)
    ranked_gain = gains[order]
    ideal = np.sort(gains)[::-1]
    n_pos = int(positive_mask.sum())

    out = {}
    for k in ks:
        ke = min(k, ranked.size)
        hits = ranked[:ke].sum()
        out[f"precision@{k}"] = hits / ke
        out[f"recall@{k}"] = hits / n_pos
        out[f"ndcg@{k}"] = _dcg(ranked_gain[:ke]) / _dcg(ideal[:ke])
    out["n_pos"] = n_pos
    out["n_cand"] = int(ranked.size)
    for name, mask in (strata or {}).items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != positive_mask.shape or (mask & ~positive_mask).any():
            raise ValueError(f"stratum {name!r} must mark a subset of the positives")
        ranked_s, n_s = mask[order], int(mask.sum())
        for k in ks:
            out[f"{name}.recall@{k}"] = ranked_s[:min(k, ranked.size)].sum() / n_s if n_s else np.nan
        out[f"{name}.n_pos"] = n_s
    return out


def macro_average(per_slate: list[dict], ks=KS) -> dict:
    """Mean of each metric over the slates; a stratum's recall over the slates where it has a positive."""
    n = len(per_slate)
    if n == 0:
        raise ValueError("no slates to average")
    agg = {f"{m}@{k}": float(np.mean([s[f"{m}@{k}"] for s in per_slate]))
           for k in ks for m in ("precision", "recall", "ndcg")}
    agg["n_slates"] = n
    agg["mean_n_pos"] = float(np.mean([s["n_pos"] for s in per_slate]))
    agg["mean_n_cand"] = float(np.mean([s["n_cand"] for s in per_slate]))
    for name in [key[:-len(".n_pos")] for key in per_slate[0] if key.endswith(".n_pos")]:
        has = [s for s in per_slate if s[f"{name}.n_pos"] > 0]
        for k in ks:
            agg[f"{name}.recall@{k}"] = float(np.mean([s[f"{name}.recall@{k}"] for s in has])) if has else np.nan
        agg[f"{name}.n_slates"] = len(has)
        agg[f"{name}.n_pos"] = int(sum(s[f"{name}.n_pos"] for s in has))
    return agg


def evaluate(slates: list[Slate], score_fn: Callable[[Slate], np.ndarray], ks=KS) -> tuple[dict, list[dict]]:
    """Score every slate with ``score_fn`` (one score per ``slate.candidates``) and macro-average."""
    per_slate = []
    for s in slates:
        m = slate_metrics(score_fn(s), s.positive_mask, ks, gains=s.gain, strata=s.strata)
        m.update(year=s.year, disease_index=s.disease_index)
        per_slate.append(m)
    return macro_average(per_slate, ks), per_slate


def slate_stats(slates: list[Slate]) -> pd.DataFrame:
    """One row per slate (year, disease_index, n_pos, n_cand, and n_pos.<stratum>), without the candidate arrays."""
    names = list(slates[0].strata) if slates else []
    return pd.DataFrame({
        "year": [s.year for s in slates],
        "disease_index": [s.disease_index for s in slates],
        "n_pos": [int(s.positive_mask.sum()) for s in slates],
        "n_cand": [int(s.candidates.size) for s in slates],
        **{f"n_pos.{k}": [int(s.strata[k].sum()) for s in slates] for k in names},
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
    out = {
        "n_slates": len(stats),
        "n_diseases": int(stats["disease_index"].nunique()),
        "n_positives": int(stats["n_pos"].sum()),
        "median_pos_per_slate": float(stats["n_pos"].median()),
        "mean_candidates": float(stats["n_cand"].mean()),
        "slates_per_year": {int(y): int(c) for y, c in stats.groupby("year").size().items()},
    }
    strata = [c for c in stats.columns if c.startswith("n_pos.")]
    if strata:
        out["positives_by_stratum"] = {c[len("n_pos."):]: int(stats[c].sum()) for c in strata}
    return out

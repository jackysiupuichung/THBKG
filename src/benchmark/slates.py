"""Slate construction and ranking metrics for the T1 discovery and T2 hypothesis tasks.

The unit of evaluation is a *slate*: one ``(disease, decision year)`` ranking
problem over the task's full candidate pool, with no negative sampling. Every
slate gets one vote in the macro average, so heavily trialled indications do
not dominate. Ties are broken pessimistically (positives rank last within a
tie, lowest grade first), so a constant score cannot look better than random by
index order.

The primary metric is ``precision@R``, TargetBench's precision at top K with K
set per slate to its number of positives R (R-precision): the share of the
top R candidates that are positives. Unlike precision at a fixed K it can
reach 1 on every slate, and a random ranking gets the slate's positive rate.
``precision@R_per_disease`` pools a disease's slates across the split's years
(hits in each year's top R over all its positives), one vote per disease.

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
    log = PredictionLog()                     # keep the raw predictions
    summary, per_slate = evaluate(slates, lambda s: model_scores(s.year, s.disease_index, s.candidates), log=log)
    log.save(out_dir, "t1_mymodel_test")      # rescore() recomputes any metric, or a stratum, from it
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
    hits_r = int(ranked[:n_pos].sum())  # K = the slate's number of positives
    out["precision@R"] = hits_r / n_pos
    out["hits@R"] = hits_r
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
    agg = {"precision@R": float(np.mean([s["precision@R"] for s in per_slate]))}
    if all("disease_index" in s for s in per_slate):
        d = pd.DataFrame({"d": [s["disease_index"] for s in per_slate], "hits": [s["hits@R"] for s in per_slate],
                          "pos": [s["n_pos"] for s in per_slate]}).groupby("d").sum()
        agg["precision@R_per_disease"] = float((d["hits"] / d["pos"]).mean())
        agg["n_diseases"] = len(d)
    agg.update({f"{m}@{k}": float(np.mean([s[f"{m}@{k}"] for s in per_slate]))
                for k in ks for m in ("precision", "recall", "ndcg")})
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


class PredictionLog:
    """The raw predictions of a slate evaluation, small enough to keep for every run.

    A T2 split holds ~750M candidate scores, too many to store. What is kept per
    slate: each positive's rank in the full evaluation order (score descending,
    ties as in ``slate_metrics``: negatives, then lower-gain positives, first),
    its score and gain, and the ``top_k`` candidates. A stratum, weighting or
    exclusion decided later only selects or reweights positives, so their ranks
    are enough to recompute every metric of this module with ``rescore``; the
    top candidates are for inspection. ``save`` writes
    ``<prefix>_positives.parquet``, ``<prefix>_slates.parquet`` and
    ``<prefix>_top<k>.parquet``.
    """

    def __init__(self, top_k: int = 100):
        self.top_k = top_k
        self._pos: list[pd.DataFrame] = []
        self._slates: list[tuple] = []
        self._top: list[pd.DataFrame] = []

    def add(self, s: Slate, scores: np.ndarray) -> None:
        """Record one scored slate."""
        scores = np.asarray(scores, dtype=np.float64)
        order = np.lexsort((s.gain, -scores))  # the order slate_metrics ranks in
        rank = np.empty(order.size, dtype=np.int64)
        rank[order] = np.arange(1, order.size + 1)
        pos = np.flatnonzero(s.positive_mask)
        self._pos.append(pd.DataFrame({"year": s.year, "disease_index": s.disease_index,
                                       "target_index": s.candidates[pos], "score": scores[pos],
                                       "gain": s.gain[pos], "rank": rank[pos]}))
        self._slates.append((s.year, s.disease_index, int(order.size), int(pos.size)))
        top = order[:self.top_k]
        self._top.append(pd.DataFrame({"year": s.year, "disease_index": s.disease_index,
                                       "rank": np.arange(1, top.size + 1), "target_index": s.candidates[top],
                                       "score": scores[top], "positive": s.positive_mask[top]}))

    def positives(self) -> pd.DataFrame:
        """year, disease_index, target_index, score, gain, rank: one row per positive."""
        return pd.concat(self._pos, ignore_index=True)

    def slates(self) -> pd.DataFrame:
        """year, disease_index, n_cand, n_pos: one row per slate."""
        return pd.DataFrame(self._slates, columns=["year", "disease_index", "n_cand", "n_pos"])

    def top(self) -> pd.DataFrame:
        """year, disease_index, rank, target_index, score, positive: the top_k candidates of each slate."""
        return pd.concat(self._top, ignore_index=True)

    def save(self, out_dir, prefix: str) -> None:
        """Write the three tables as parquet files in ``out_dir``."""
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.positives().to_parquet(out / f"{prefix}_positives.parquet", index=False)
        self.slates().to_parquet(out / f"{prefix}_slates.parquet", index=False)
        self.top().to_parquet(out / f"{prefix}_top{self.top_k}.parquet", index=False)


def rescore(positives: pd.DataFrame, slates: pd.DataFrame, ks=KS) -> tuple[dict, list[dict]]:
    """``evaluate``'s metrics recomputed from a ``PredictionLog``, for its positives or any subset of them.

    Filter ``positives`` to a stratum to score it: the other positives keep the
    places they took, since every rank was taken in the full slate, and R and the
    NDCG ideal become the stratum's. Slates left without a positive are dropped.
    On the full set it equals ``evaluate`` (``run_cpu_baselines.py`` checks this).
    """
    n_cand = {(int(y), int(d)): int(c) for y, d, c in slates[["year", "disease_index", "n_cand"]].itertuples(index=False)}
    per_slate = []
    for _, g in positives.groupby(["year", "disease_index"], sort=True):
        year, d = int(g["year"].to_numpy()[0]), int(g["disease_index"].to_numpy()[0])
        r, gain, c = g["rank"].to_numpy(), g["gain"].to_numpy(np.float64), n_cand[(year, d)]
        n_pos = r.size
        ideal = np.sort(gain)[::-1]
        m: dict = {}
        for k in ks:
            ke = min(k, c)
            hits = int((r <= ke).sum())
            m[f"precision@{k}"] = hits / ke
            m[f"recall@{k}"] = hits / n_pos
            top = r <= ke
            dcg = float((gain[top] / np.log2(r[top] + 1.0)).sum())
            m[f"ndcg@{k}"] = dcg / _dcg(ideal[:ke])
        hits_r = int((r <= n_pos).sum())
        m.update({"precision@R": hits_r / n_pos, "hits@R": hits_r, "n_pos": n_pos, "n_cand": c,
                  "year": int(year), "disease_index": int(d)})
        per_slate.append(m)
    return macro_average(per_slate, ks), per_slate


def evaluate(slates: list[Slate], score_fn: Callable[[Slate], np.ndarray], ks=KS,
             log: PredictionLog | None = None) -> tuple[dict, list[dict]]:
    """Score every slate with ``score_fn`` (one score per ``slate.candidates``) and macro-average; with
    ``log``, also record the raw predictions."""
    per_slate = []
    for s in slates:
        scores = score_fn(s)
        m = slate_metrics(scores, s.positive_mask, ks, gains=s.gain, strata=s.strata)
        m.update(year=s.year, disease_index=s.disease_index)
        per_slate.append(m)
        if log is not None:
            log.add(s, scores)
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

    precision@K = P/C (for any K, R included) and recall@K = K/C per slate,
    macro-averaged; per disease, expected hits P^2/C pooled over its slates. A
    floor to read results against: with ~2 positives among ~19,600 targets,
    absolute precision at a fixed K is tiny for any model.
    """
    agg = {"precision@R": float((stats["n_pos"] / stats["n_cand"]).mean())}
    d = stats.assign(hits=stats["n_pos"] ** 2 / stats["n_cand"]).groupby("disease_index")[["hits", "n_pos"]].sum()
    agg["precision@R_per_disease"] = float((d["hits"] / d["n_pos"]).mean())
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

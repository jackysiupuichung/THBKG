"""Label tables for the THBKG benchmark tasks.

Both tasks follow one target-disease pair through the clinic, under one
temporal protocol: a model ranks candidates for a decision year ``Y`` using only
graph evidence dated strictly before ``Y``. T1 splits decision years by
``SPLITS``; T3 splits its labelled pairs by Phase II entry year
(``ADVANCEMENT_SPLITS``) into the same train/val/test periods.

========  ==========================================  =============================
task      positive                                    candidate pool for (disease, Y)
========  ==========================================  =============================
T1        (target, disease) pair whose first trial    every target with no trial for
discovery (any phase) starts in ``Y``                 that disease before ``Y``
T3        Phase II entry in ``Y`` that reaches        the labelled pairs themselves
advance-  Phase III by a fixed horizon, at least      (labels/train|eval.parquet)
ment      3 years after the split's last entry year
========  ==========================================  =============================

T1 labels are read off the graph's own target->disease ``clinical_trial_*``
edges, so a label year is by construction the year the masking rule hides. T3
labels are read off the start-dated trial edge (``build_advancement_labels``):
a pair enters Phase II in the first year a Phase II trial starts while it has
no Phase III+ trial, and is positive if a Phase III+ trial starts by the
horizon of its label file (``ADVANCEMENT_LABEL_FILES``).

T1 positives are graded for NDCG by their max outcome phase: the highest phase
among the pair's finished trials (outcome edges) up to the end of the graph.
Positives with no finished trial yet get ``MIN_POSITIVE_GRADE``. Precision and
recall stay binary. The grade looks past the decision year, so use it only to
score rankings; to train on it, pass ``horizon`` = the year before the split
being evaluated.

T1 test positives need a first trial at Phase 1 or later
(``MIN_FIRST_TRIAL_PHASE``); pairs whose first trials are all early Phase 1 or
of unknown phase are left out of the test slates, neither positive nor
negative. Train and validation keep them as positives.

T1 test slates are kept only for diseases that gain more than 3 new targets
over the whole test period (``MIN_DISEASE_POSITIVES``). Always build slates
through ``task_slates`` so the split years and this filter are applied.

Works directly on a release directory (graph.safetensors + metadata.json +
mappings/), without PyG.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .slates import Slate, discovery_slates

# The graph's last evidence year (config/event_graph_config.yaml
# time_range.last_year): evidence runs to the 26.09 release, September 2026.
LAST_GRAPH_YEAR = 2026
# Last decision year: the last full year. Pinned rather than read off the labels,
# so the partial year never enters the test split; last_label_year() checks the
# labels run to LAST_GRAPH_YEAR.
LAST_DECISION_YEAR = 2025
# Decision years for T1, inclusive.
SPLITS = {"train": (1996, 2015), "val": (2016, 2019), "test": (2020, LAST_DECISION_YEAR)}
# Evidence dated 1985 or earlier is collapsed onto 1985 by the event list
# (config time_range.first_year), so 1985 cannot be a decision year: a "first
# trial in 1985" means "by 1985".
FIRST_GRAPH_YEAR = 1985

# Minimum positives a disease needs over a whole split for its slates to
# count, per (task, split); default 1. T1 test keeps only diseases with more
# than 3 new targets across the test period.
MIN_DISEASE_POSITIVES = {("discovery", "test"): 4}

# Minimum phase weight of a pair's first trials (the strongest trial dated in
# its first-trial year) for it to count as a positive, per (task, split);
# default: any trial. 0.1 = Phase 1; below it are early Phase 1 (0.05) and
# unknown phase (0.01).
MIN_FIRST_TRIAL_PHASE = {("discovery", "test"): 0.1}

# T3 splits by Phase II entry (transition) year: (label file, first, last), with
# None = open. train is labels/train.parquet (outcome horizon = 2015 + window);
# val and test divide labels/eval.parquet, whose outcome horizon is the last
# evidence year, so its latest pairs enter Phase II window years before it.
ADVANCEMENT_SPLITS = {"train": ("train", None, 2015), "val": ("eval", 2016, 2019),
                      "test": ("eval", 2020, None)}

# T3 labels come from the start-dated trial edge, whose score is the highest
# phase weight among the pair's trials started up to its year.
ADVANCEMENT_RELATION = "clinical_trial_ongoing"
# Phase weight -> phase. Compound stages count as their lower phase; unknown
# phase (0.01) is ignored.
WEIGHT_PHASE = {0.05: 1, 0.1: 1, 0.15: 1, 0.2: 2, 0.5: 2, 0.7: 3, 1.0: 4}
ADVANCEMENT_WINDOW = 3  # years every pair gets to reach Phase III
FIRST_ENTRY_YEAR = 1995
# Label file -> (first entry year, last entry year, outcome horizon). Train
# pairs are labelled ADVANCEMENT_WINDOW years after the last train entry year;
# eval pairs at the graph's last year, so the last eval entries also get
# ADVANCEMENT_WINDOW years.
ADVANCEMENT_LABEL_FILES = {
    "train": (FIRST_ENTRY_YEAR, SPLITS["train"][1], SPLITS["train"][1] + ADVANCEMENT_WINDOW),
    "eval": (SPLITS["val"][0], LAST_GRAPH_YEAR - ADVANCEMENT_WINDOW, LAST_GRAPH_YEAR),
}
# T3 results are reported per therapeutic area: the primary ones are those not
# matching this pattern with at least MIN_PRIMARY_TA_PAIRS eval pairs that have
# direct (non-trial) target-disease evidence before their entry year.
SECONDARY_TA_PATTERN = "biological.*process|pregnancy|poisoning|procedure|animal|infectious"
MIN_PRIMARY_TA_PAIRS = 100

T1_RELATION_PREFIX = "clinical_trial"

# Outcome edges (dated when the trial ended) and the factor that undoes Open
# Targets' halving of the score for trials stopped for safety or efficacy.
OUTCOME_RELATIONS = {
    "clinical_trial_positive": 1.0,
    "clinical_trial_Unknown/Operational": 1.0,
    "clinical_trial_unmet_efficacy": 2.0,
    "clinical_trial_adverse_effects": 2.0,
}
# Phase weight (OT clinical-precedence score) -> grade on a phase scale.
# Unknown phase and early Phase 1 share the lowest grade.
PHASE_GRADE = {0.01: 0.5, 0.05: 0.5, 0.1: 1.0, 0.15: 1.5, 0.2: 2.0, 0.5: 2.5, 0.7: 3.0, 1.0: 4.0}
MIN_POSITIVE_GRADE = 0.5  # positives whose trials have not finished yet


def _release_meta(release_dir: Path) -> dict:
    return json.loads((release_dir / "metadata.json").read_text())


def node_ids(release_dir: Path, node_type: str) -> np.ndarray:
    """Node id strings in index order."""
    df = pd.read_parquet(release_dir / "mappings" / f"{node_type}.parquet")
    return df.sort_values("index")["node_id"].astype(str).to_numpy()


def _first_years(release_dir: Path, src: str, dst: str, keep) -> pd.DataFrame:
    """Earliest edge_time per (src index, dst index) over the matching relations."""
    from safetensors import safe_open

    meta = _release_meta(release_dir)
    parts = []
    with safe_open(str(release_dir / "graph.safetensors"), framework="np") as f:
        for e in meta["edge_types"]:
            if e["src"] != src or e["dst"] != dst or not keep(e["rel"]):
                continue
            ei = f.get_tensor(e["tensors"]["edge_index"]["key"])
            t = f.get_tensor(e["tensors"]["edge_time"]["key"])
            parts.append(pd.DataFrame({"src": ei[0], "dst": ei[1], "year": t}))
    if not parts:
        raise ValueError(f"no {src}->{dst} relations matched in {release_dir}")
    df = pd.concat(parts, ignore_index=True)
    return df.groupby(["src", "dst"], as_index=False)["year"].min()


def max_outcome_phase(release_dir: str | Path, horizon: int | None = None) -> pd.DataFrame:
    """Highest phase grade among each (target, disease) pair's finished trials.

    Reads the outcome edges dated up to ``horizon`` (default: all). Returns
    src (target index), dst (disease index), max_outcome_phase.
    """
    from safetensors import safe_open

    release_dir = Path(release_dir)
    meta = _release_meta(release_dir)
    parts = []
    with safe_open(str(release_dir / "graph.safetensors"), framework="np") as f:
        for e in meta["edge_types"]:
            factor = OUTCOME_RELATIONS.get(e["rel"])
            if factor is None or (e["src"], e["dst"]) != ("target", "disease"):
                continue
            ei = f.get_tensor(e["tensors"]["edge_index"]["key"])
            t = f.get_tensor(e["tensors"]["edge_time"]["key"])
            w = f.get_tensor(e["tensors"]["edge_attr"]["key"])[:, 0].astype(np.float64)
            keep = np.ones(len(t), dtype=bool) if horizon is None else t <= horizon
            parts.append(pd.DataFrame({"src": ei[0][keep], "dst": ei[1][keep],
                                       "w": np.round(w[keep] * factor, 3)}))
    if len(parts) != len(OUTCOME_RELATIONS):
        raise ValueError(f"expected {len(OUTCOME_RELATIONS)} outcome relations, found {len(parts)}")
    df = pd.concat(parts, ignore_index=True)
    unknown = sorted(set(df["w"]) - set(PHASE_GRADE))
    if unknown:
        raise ValueError(f"outcome scores with no phase grade: {unknown}")
    df["max_outcome_phase"] = df["w"].map(PHASE_GRADE)
    return df.groupby(["src", "dst"], as_index=False)["max_outcome_phase"].max()


def first_trial_phase(release_dir: str | Path) -> pd.DataFrame:
    """Phase weight of each (target, disease) pair's strongest trial in its first-trial year.

    Reads every target->disease ``clinical_trial_*`` edge, undoing Open Targets'
    halving on the failure outcomes. Returns src, dst, year, first_trial_phase.
    """
    from safetensors import safe_open

    release_dir = Path(release_dir)
    meta = _release_meta(release_dir)
    parts = []
    with safe_open(str(release_dir / "graph.safetensors"), framework="np") as f:
        for e in meta["edge_types"]:
            if (e["src"], e["dst"]) != ("target", "disease") or not e["rel"].startswith(T1_RELATION_PREFIX):
                continue
            ei = f.get_tensor(e["tensors"]["edge_index"]["key"])
            t = f.get_tensor(e["tensors"]["edge_time"]["key"])
            w = f.get_tensor(e["tensors"]["edge_attr"]["key"])[:, 0].astype(np.float64)
            parts.append(pd.DataFrame({"src": ei[0], "dst": ei[1], "year": t,
                                       "w": np.round(w * OUTCOME_RELATIONS.get(e["rel"], 1.0), 3)}))
    df = pd.concat(parts, ignore_index=True)
    unknown = sorted(set(df["w"]) - set(PHASE_GRADE))
    if unknown:
        raise ValueError(f"trial scores with no phase weight: {unknown}")
    first = df.groupby(["src", "dst"])["year"].transform("min")
    return (df[df["year"] == first].groupby(["src", "dst"], as_index=False)
            .agg(year=("year", "first"), first_trial_phase=("w", "max")))


def build_discovery_labels(release_dir: str | Path) -> pd.DataFrame:
    """T1: first trial year of every (target, disease) pair with any trial, plus its grade.

    ``first_trial_phase`` is the phase weight of the pair's strongest trial in
    that year. ``max_outcome_phase`` is NaN while none of the pair's trials has
    finished; ``grade`` fills those with ``MIN_POSITIVE_GRADE``.
    """
    release_dir = Path(release_dir)
    df = _first_years(release_dir, "target", "disease", lambda r: r.startswith(T1_RELATION_PREFIX))
    phase = first_trial_phase(release_dir)
    df = df.merge(phase, on=["src", "dst", "year"], how="left", validate="one_to_one")
    if df["first_trial_phase"].isna().any():
        raise ValueError("pairs without a first-trial phase")
    df = df.merge(max_outcome_phase(release_dir), on=["src", "dst"], how="left")
    tid, did = node_ids(release_dir, "target"), node_ids(release_dir, "disease")
    out = pd.DataFrame({
        "target_id": tid[df["src"]],
        "disease_id": did[df["dst"]],
        "target_index": df["src"].astype(np.int64),
        "disease_index": df["dst"].astype(np.int64),
        "first_trial_year": df["year"].astype(np.int64),
        "first_trial_phase": df["first_trial_phase"].astype(np.float64),
        "max_outcome_phase": df["max_outcome_phase"].astype(np.float64),
        "grade": df["max_outcome_phase"].fillna(MIN_POSITIVE_GRADE).astype(np.float64),
    })
    return out.sort_values(["first_trial_year", "disease_id", "target_id"], ignore_index=True)


def last_label_year(labels: pd.DataFrame) -> int:
    """End of the test split (LAST_DECISION_YEAR), once the labels are checked to
    run to the graph's last year."""
    latest = int(labels["first_trial_year"].max())
    if latest != LAST_GRAPH_YEAR:
        raise ValueError(f"labels run to {latest} but LAST_GRAPH_YEAR is {LAST_GRAPH_YEAR}: "
                         "update it together with the graph's time range")
    return LAST_DECISION_YEAR


def decision_years(split: str, last_year: int | None = None) -> list[int]:
    """Decision years of a T1 split. ``last_year`` resolves an open-ended split."""
    lo, hi = SPLITS[split]
    if hi is None:
        if last_year is None:
            raise ValueError(f"split {split!r} runs to the latest label year; pass last_year")
        hi = last_year
    return list(range(lo, hi + 1))


def task_slates(task: str, split: str, labels: pd.DataFrame, *,
                num_targets: int | None = None, years=None) -> list[Slate]:
    """Slates for one task and split, with the split's years and slate filter applied.

    ``years`` restricts to a subset of the split, e.g. one year at a time to
    bound memory (a T1 slate holds ~19,600 candidates).
    """
    if task != "discovery":
        raise ValueError(f"unknown slate task {task!r} (T3 is evaluated on its labelled pairs)")
    if num_targets is None:
        raise ValueError("discovery slates need num_targets")
    split_years = decision_years(split, last_label_year(labels))
    years = split_years if years is None else [y for y in years if y in split_years]
    min_phase = MIN_FIRST_TRIAL_PHASE.get((task, split))
    if min_phase is not None:
        labels = labels.assign(rankable=labels["first_trial_phase"] >= min_phase)
    slates = discovery_slates(labels, years, num_targets)
    k = MIN_DISEASE_POSITIVES.get((task, split), 1)
    if k > 1:
        # Counted over the whole split, not over ``years``, so a year-by-year
        # build keeps the same slates. Every T1 positive is in its slate's pool.
        in_split = labels["first_trial_year"].isin(split_years)
        if "rankable" in labels.columns:
            in_split &= labels["rankable"]
        per_disease = labels.loc[in_split].groupby("disease_index").size()
        keep = set(per_disease.index[per_disease >= k].tolist())
        slates = [s for s in slates if s.disease_index in keep]
    return slates


def advancement_labels(release_dir: str | Path, split: str) -> pd.DataFrame:
    """T3 labelled pairs of one split: target_id, disease_id, transition_year, outcome."""
    file, lo, hi = ADVANCEMENT_SPLITS[split]
    df = pd.read_parquet(Path(release_dir) / "labels" / f"{file}.parquet")
    y = df["transition_year"]
    keep = pd.Series(True, index=df.index)
    if lo is not None:
        keep &= y >= lo
    if hi is not None:
        keep &= y <= hi
    return df[keep].reset_index(drop=True)


def advancement_years(release_dir: str | Path) -> pd.DataFrame:
    """Phase II entry year and first Phase III+ year of every (target, disease) pair.

    A pair enters Phase II in the first year its highest phase so far is 2; a
    pair whose first Phase II trial starts in the same year as, or after, a
    Phase III+ one never enters (entry_year NaN). Returns src, dst, entry_year,
    phase3_year.
    """
    from safetensors import safe_open

    release_dir = Path(release_dir)
    meta = _release_meta(release_dir)
    e = next((e for e in meta["edge_types"]
              if e["rel"] == ADVANCEMENT_RELATION and (e["src"], e["dst"]) == ("target", "disease")), None)
    if e is None:
        raise ValueError(f"no target->{ADVANCEMENT_RELATION}->disease edges in {release_dir}")
    with safe_open(str(release_dir / "graph.safetensors"), framework="np") as f:
        ei = f.get_tensor(e["tensors"]["edge_index"]["key"])
        t = f.get_tensor(e["tensors"]["edge_time"]["key"])
        w = np.round(f.get_tensor(e["tensors"]["edge_attr"]["key"])[:, 0].astype(np.float64), 3)
    unknown = sorted(set(w.tolist()) - set(WEIGHT_PHASE) - {0.01})
    if unknown:
        raise ValueError(f"{ADVANCEMENT_RELATION} scores with no phase: {unknown}")
    df = pd.DataFrame({"src": ei[0], "dst": ei[1], "year": t, "phase": pd.Series(w).map(WEIGHT_PHASE)})
    df = df.dropna(subset=["phase"]).sort_values(["src", "dst", "year"])
    # the score is already a running maximum; take it again so the rule holds as stated
    df["phase"] = df.groupby(["src", "dst"])["phase"].cummax()
    entry = df[df["phase"] == 2].groupby(["src", "dst"])["year"].min().rename("entry_year")
    phase3 = df[df["phase"] >= 3].groupby(["src", "dst"])["year"].min().rename("phase3_year")
    return pd.concat([entry, phase3], axis=1).reset_index()


def build_advancement_labels(release_dir: str | Path) -> dict[str, pd.DataFrame]:
    """T3 label files (``ADVANCEMENT_LABEL_FILES``): every pair entering Phase II
    within a file's entry years, positive if it reaches Phase III by the file's
    horizon. Columns: target_id, disease_id, target_index, disease_index,
    transition_year, outcome."""
    release_dir = Path(release_dir)
    yrs = advancement_years(release_dir)
    tid, did = node_ids(release_dir, "target"), node_ids(release_dir, "disease")
    out = {}
    for name, (lo, hi, horizon) in ADVANCEMENT_LABEL_FILES.items():
        g = yrs[yrs["entry_year"].between(lo, hi)]
        src, dst = g["src"].to_numpy(np.int64), g["dst"].to_numpy(np.int64)
        out[name] = pd.DataFrame({
            "target_id": tid[src], "disease_id": did[dst],
            "target_index": src, "disease_index": dst,
            "transition_year": g["entry_year"].to_numpy(np.int64),
            "outcome": (g["phase3_year"] <= horizon).to_numpy(),  # NaN (never) -> False
        }).sort_values(["transition_year", "disease_id", "target_id"], ignore_index=True)
    return out


def therapeutic_areas(diseases_dir: str | Path) -> pd.DataFrame:
    """Each disease's Open Targets therapeutic areas, plus ("ALL_0", "all") for
    every disease that has one. Columns: disease_id, therapeutic_area_id,
    therapeutic_area_name."""
    import pyarrow.dataset as ds

    d = ds.dataset(str(diseases_dir), format="parquet").to_table(
        columns=["id", "name", "therapeuticAreas", "ontology"]).to_pandas()
    is_ta = d["ontology"].map(lambda o: bool(o) and bool(o.get("isTherapeuticArea")))
    names = dict(zip(d.loc[is_ta, "id"], d.loc[is_ta, "name"]))
    ex = d[["id", "therapeuticAreas"]].explode("therapeuticAreas").dropna()
    ex = ex[ex["therapeuticAreas"].isin(names)]
    ta = pd.DataFrame({"disease_id": ex["id"].to_numpy(), "therapeutic_area_id": ex["therapeuticAreas"].to_numpy(),
                       "therapeutic_area_name": ex["therapeuticAreas"].map(names).to_numpy()})
    every = pd.DataFrame({"disease_id": ta["disease_id"].unique(), "therapeutic_area_id": "ALL_0",
                          "therapeutic_area_name": "all"})
    return (pd.concat([every, ta], ignore_index=True).drop_duplicates()
            .sort_values(["disease_id", "therapeutic_area_id"], ignore_index=True))


def primary_therapeutic_areas(release_dir: str | Path, eval_labels: pd.DataFrame,
                              tas: pd.DataFrame) -> list[str]:
    """Names of the therapeutic areas T3 is reported on (see SECONDARY_TA_PATTERN)."""
    first = _first_years(Path(release_dir), "target", "disease", lambda r: not r.startswith(T1_RELATION_PREFIX))
    first = first.rename(columns={"src": "target_index", "dst": "disease_index", "year": "first_evidence_year"})
    lab = eval_labels.merge(first, on=["target_index", "disease_index"], how="left")
    lab["has_evidence"] = lab["first_evidence_year"] < lab["transition_year"]  # NaN -> False
    per = lab.merge(tas, on="disease_id").groupby("therapeutic_area_name")["has_evidence"].sum()
    keep = per[(per >= MIN_PRIMARY_TA_PAIRS) & ~per.index.str.contains(SECONDARY_TA_PATTERN)]
    return sorted(keep.index.tolist(), key=lambda n: (n != "all", n))

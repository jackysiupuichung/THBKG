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
labels are external and unchanged.

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

# Decision years for T1, inclusive. ``None`` = the latest year with trial
# evidence in the labels (2025 for the 26.09 graph).
SPLITS = {"train": (1996, 2015), "val": (2016, 2019), "test": (2020, None)}
# Trial evidence dated 1995 or earlier is collapsed onto 1995 by the event list,
# so 1995 cannot be a decision year: a "first trial in 1995" means "by 1995".
FIRST_GRAPH_YEAR = 1995

# Minimum positives a disease needs over a whole split for its slates to
# count, per (task, split); default 1. T1 test keeps only diseases with more
# than 3 new targets across the test period.
MIN_DISEASE_POSITIVES = {("discovery", "test"): 4}

# T3 splits by Phase II entry (transition) year: (label file, first, last), with
# None = open. train is labels/train.parquet (outcome horizon = 2015 + window);
# val and test divide labels/eval.parquet, whose outcome horizon is the last
# evidence year, so its latest pairs enter Phase II window years before it.
ADVANCEMENT_SPLITS = {"train": ("train", None, 2015), "val": ("eval", 2016, 2019),
                      "test": ("eval", 2020, None)}

T1_RELATION_PREFIX = "clinical_trial"


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


def build_discovery_labels(release_dir: str | Path) -> pd.DataFrame:
    """T1: first trial year of every (target, disease) pair with any trial."""
    release_dir = Path(release_dir)
    df = _first_years(release_dir, "target", "disease", lambda r: r.startswith(T1_RELATION_PREFIX))
    tid, did = node_ids(release_dir, "target"), node_ids(release_dir, "disease")
    out = pd.DataFrame({
        "target_id": tid[df["src"]],
        "disease_id": did[df["dst"]],
        "target_index": df["src"].astype(np.int64),
        "disease_index": df["dst"].astype(np.int64),
        "first_trial_year": df["year"].astype(np.int64),
    })
    return out.sort_values(["first_trial_year", "disease_id", "target_id"], ignore_index=True)


def last_label_year(labels: pd.DataFrame) -> int:
    """Latest first-trial year in a T1 label table: the end of the test split."""
    return int(labels["first_trial_year"].max())


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
    slates = discovery_slates(labels, years, num_targets)
    k = MIN_DISEASE_POSITIVES.get((task, split), 1)
    if k > 1:
        # Counted over the whole split, not over ``years``, so a year-by-year
        # build keeps the same slates. Every T1 positive is in its slate's pool.
        in_split = labels["first_trial_year"].isin(split_years)
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

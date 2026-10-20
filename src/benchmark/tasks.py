"""Label tables for the three THBKG benchmark tasks.

All three tasks share one temporal protocol: a model ranks candidates for a
decision year ``Y`` using only graph evidence dated strictly before ``Y``. T1
and T2 split decision years the same way (``SPLITS``); T3 keeps the released
train/eval split of its label files.

========  ==========================================  =============================
task      positive                                    candidate pool for (disease, Y)
========  ==========================================  =============================
T1        (target, disease) pair whose first trial    every target with no trial for
discovery (any phase) starts in ``Y``                 that disease before ``Y``
T2        (drug, disease) pair whose first trial      every drug with a trial for some
repur-    starts in ``Y``, for a drug already         disease before ``Y`` and none for
posing    trialled for another disease before ``Y``   that disease before ``Y``
T3        Phase II entry in ``Y`` that reaches        the labelled pairs themselves
advance-  Phase III by a fixed horizon                (labels/train|eval.parquet)
ment
========  ==========================================  =============================

T1 and T2 labels are read off the graph's own trial edges, so a label year is
by construction the year the masking rule hides: T1 uses the target->disease
``clinical_trial_*`` relations, T2 the molecule->disease ``trialled_for``
relation. T3 labels are external and unchanged.

T1 test slates are kept only when the disease gains more than 3 new targets
that year (``MIN_POSITIVES``). Always build slates through ``task_slates`` so
the split years and this filter are applied.

Works directly on a release directory (graph.safetensors + metadata.json +
mappings/), without PyG.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .slates import Slate, discovery_slates, repurposing_slates

# Decision years for T1 and T2, inclusive. ``None`` = the latest year with
# trial evidence in the labels (2025 for the 26.09 graph).
SPLITS = {"train": (1996, 2015), "val": (2016, 2019), "test": (2020, None)}
# Trial evidence dated 1995 or earlier is collapsed onto 1995 by the event list,
# so 1995 cannot be a decision year: a "first trial in 1995" means "by 1995".
FIRST_GRAPH_YEAR = 1995

# Minimum positives for a slate to count, per (task, split); default 1. With
# 3 or fewer positives one hit moves recall@K by a third or more, so T1 test
# keeps only slates with more than 3 new targets.
MIN_POSITIVES = {("discovery", "test"): 4}

T1_RELATION_PREFIX = "clinical_trial"
T2_RELATION = "trialled_for"


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


def build_repurposing_labels(release_dir: str | Path) -> pd.DataFrame:
    """T2: first trial year of every (drug, disease) pair, plus the drug's first trial year.

    ``drug_first_trial_year`` is the earliest trial for that drug in any disease;
    a pair is a repurposing event iff ``first_trial_year > drug_first_trial_year``.
    """
    release_dir = Path(release_dir)
    df = _first_years(release_dir, "molecule", "disease", lambda r: r == T2_RELATION)
    df["drug_first"] = df.groupby("src")["year"].transform("min")
    mid, did = node_ids(release_dir, "molecule"), node_ids(release_dir, "disease")
    out = pd.DataFrame({
        "molecule_id": mid[df["src"]],
        "disease_id": did[df["dst"]],
        "molecule_index": df["src"].astype(np.int64),
        "disease_index": df["dst"].astype(np.int64),
        "first_trial_year": df["year"].astype(np.int64),
        "drug_first_trial_year": df["drug_first"].astype(np.int64),
    })
    return out.sort_values(["first_trial_year", "disease_id", "molecule_id"], ignore_index=True)


def last_label_year(labels: pd.DataFrame) -> int:
    """Latest first-trial year in a T1/T2 label table: the end of the test split."""
    return int(labels["first_trial_year"].max())


def decision_years(split: str, last_year: int | None = None) -> list[int]:
    """Decision years of a T1/T2 split. ``last_year`` resolves an open-ended split."""
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
    split_years = decision_years(split, last_label_year(labels))
    years = split_years if years is None else [y for y in years if y in split_years]
    if task == "discovery":
        if num_targets is None:
            raise ValueError("discovery slates need num_targets")
        slates = discovery_slates(labels, years, num_targets)
    elif task == "repurposing":
        slates = repurposing_slates(labels, years)
    else:
        raise ValueError(f"unknown task {task!r}")
    k = MIN_POSITIVES.get((task, split), 1)
    return [s for s in slates if int(s.positive_mask.sum()) >= k]

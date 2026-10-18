"""Label tables for the three THBKG benchmark tasks.

All three tasks share one temporal protocol: a model ranks candidates for a
decision year ``Y`` using only graph evidence dated strictly before ``Y``, and
decision years are split the same way for every task.

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

Works directly on a release directory (graph.safetensors + metadata.json +
mappings/), without PyG.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Decision years, inclusive. The T3 eval split (Phase II entry 2016-2021) fixes
# the test window; ending at 2021 also keeps every test year clear of the
# trial-registration lag visible from 2022 on.
SPLITS = {"train": (1996, 2015), "test": (2016, 2021)}
# Trial evidence dated 1995 or earlier is collapsed onto 1995 by the event list,
# so 1995 cannot be a decision year: a "first trial in 1995" means "by 1995".
FIRST_GRAPH_YEAR = 1995

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


def decision_years(split: str) -> list[int]:
    lo, hi = SPLITS[split]
    return list(range(lo, hi + 1))

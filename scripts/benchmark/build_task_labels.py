#!/usr/bin/env python
"""Write the T1 label table into a release directory and summarise both tasks.

    python scripts/benchmark/build_task_labels.py --release-dir $OUT/hf_release/THBKG

Writes
  labels/discovery.parquet    T1: target_id, disease_id, *_index, first_trial_year
  benchmark_tasks.json        splits, slate filters, label counts and slate statistics
                              per task and split, plus the closed-form random floor on
                              val and test

T3 (advancement) labels are the existing labels/train.parquet and eval.parquet;
they are only summarised here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.benchmark.slates import expected_random, slate_stats, slate_summary  # noqa: E402
from src.benchmark.tasks import (  # noqa: E402
    MIN_POSITIVES, SPLITS, build_discovery_labels, decision_years,
    last_label_year, task_slates,
)


def _split_stats(task, split, labels, n_targets) -> pd.DataFrame:
    """Build slates one year at a time and keep only their sizes: a full T1 split
    holds ~19,600 candidates per slate."""
    years = decision_years(split, last_label_year(labels))
    frames = [slate_stats(task_slates(task, split, labels, num_targets=n_targets, years=[y]))
              for y in years]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-dir", required=True, type=Path)
    args = ap.parse_args()
    rd = args.release_dir
    meta = json.loads((rd / "metadata.json").read_text())
    n_targets = meta["node_types"]["target"]["num_nodes"]

    t1 = build_discovery_labels(rd)
    t1.to_parquet(rd / "labels" / "discovery.parquet", index=False)
    print(f"labels/discovery.parquet:   {len(t1):,} (target, disease) pairs")

    last = last_label_year(t1)
    summary = {
        "splits_decision_years": {k: [lo, last if hi is None else hi] for k, (lo, hi) in SPLITS.items()},
        "min_positives_per_slate": {f"{t}.{s}": k for (t, s), k in MIN_POSITIVES.items()},
        "tasks": {},
    }
    for task, labels in (("discovery", t1),):
        summary["tasks"][task] = {}
        for split in SPLITS:
            stats = _split_stats(task, split, labels, n_targets)
            s = slate_summary(stats)
            if split != "train" and not stats.empty:
                s["random_expected"] = expected_random(stats)
            summary["tasks"][task][split] = s
            print(f"{task:12s} {split:5s} slates={s['n_slates']:,} positives={s.get('n_positives', 0):,} "
                  f"mean_candidates={s.get('mean_candidates', 0):,.0f}")

    adv = {}
    for split in ("train", "eval"):
        df = pd.read_parquet(rd / "labels" / f"{split}.parquet")
        adv[split] = {"pairs": len(df), "positive_rate": round(float(df["outcome"].astype(bool).mean()), 4),
                      "transition_years": [int(df["transition_year"].min()), int(df["transition_year"].max())]}
    summary["tasks"]["advancement"] = adv

    (rd / "benchmark_tasks.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item()
                                                        if isinstance(o, np.generic) else str(o)))
    print(f"✅ {rd / 'benchmark_tasks.json'}")


if __name__ == "__main__":
    main()

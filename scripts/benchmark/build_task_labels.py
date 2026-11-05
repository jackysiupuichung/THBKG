#!/usr/bin/env python
"""Write the benchmark label tables into a release directory and summarise them.

    python scripts/benchmark/build_task_labels.py --release-dir $OUT/hf_release/THBKG \
        [--diseases-dir $INPUT_EVIDENCE_DIR/diseases]

Writes
  labels/discovery.parquet    T1: target_id, disease_id, *_index, first_trial_year,
                              first_trial_phase, max_outcome_phase, grade
  labels/hypothesis.parquet   T2, every direct target-disease pair: target_id, disease_id,
                              *_index, first_direct_year, novel (the positives: that first
                              edge is a literature co-mention), and for the strata
                              first_literature_year, first_other_evidence_year,
                              first_trial_year, literature_repeated, target_prior_diseases
  labels/train.parquet        T3, read off the graph's start-dated trial edge:
  labels/eval.parquet           target_id, disease_id, *_index, transition_year, outcome
  labels/therapeutic_areas.parquet          disease -> therapeutic areas (needs
  labels/primary_therapeutic_areas.json     --diseases-dir, default
                                            $INPUT_EVIDENCE_DIR/diseases)
  benchmark_tasks.json        the official metric per task and the seeds to report; T1/T2
                              scored splits (val, test: window, fit_before, the
                              stacked recipe's windows), the recipes, the label relations
                              hidden by recipe "single", slate filters, and per task and
                              split the slate statistics (T2: positives per stratum), the
                              training positives per recipe and the closed-form random
                              floors; T3 label counts and model-selection folds (cv_folds,
                              with pair counts)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.benchmark.results import SEEDS  # noqa: E402
from src.benchmark.slates import expected_random, slate_stats, slate_summary  # noqa: E402
from src.benchmark.tasks import (  # noqa: E402
    ADVANCEMENT_LABEL_FILES, ADVANCEMENT_SPLITS, HYPOTHESIS_STRATA, MIN_DISEASE_POSITIVES, MIN_FIRST_TRIAL_PHASE,
    PRIMARY_METRIC, RECIPES, SPLITS, advancement_fold_labels, advancement_labels, build_advancement_labels, build_discovery_labels,
    build_hypothesis_labels, cv_folds, fit_before, fit_windows, label_edge_types, last_label_year,
    primary_therapeutic_areas, task_slates, therapeutic_areas, training_pairs,
)


def _split_stats(task, split, labels, n_targets) -> tuple[pd.DataFrame, dict]:
    """Build a scored split's slates (one per disease, ~19,600 candidates each) and keep only their sizes,
    plus the grade counts of their positives."""
    slates = task_slates(task, split, labels, num_targets=n_targets)
    grades = [g for s in slates for g in s.gain[s.positive_mask].tolist()]
    counts = pd.Series(grades, dtype=float).value_counts().sort_index()
    return slate_stats(slates), {str(k): int(v) for k, v in counts.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-dir", required=True, type=Path)
    ap.add_argument("--diseases-dir", type=Path,
                    default=(Path(os.environ["INPUT_EVIDENCE_DIR"]) / "diseases"
                             if os.environ.get("INPUT_EVIDENCE_DIR") else None),
                    help="Open Targets disease index, for the therapeutic-area tables")
    args = ap.parse_args()
    rd = args.release_dir
    meta = json.loads((rd / "metadata.json").read_text())
    n_targets = meta["node_types"]["target"]["num_nodes"]

    t1 = build_discovery_labels(rd)
    t1.to_parquet(rd / "labels" / "discovery.parquet", index=False)
    print(f"labels/discovery.parquet:   {len(t1):,} (target, disease) pairs")

    t2 = build_hypothesis_labels(rd)
    t2.to_parquet(rd / "labels" / "hypothesis.parquet", index=False)
    print(f"labels/hypothesis.parquet:  {len(t2):,} directly linked (target, disease) pairs, "
          f"{int(t2['novel'].sum()):,} first linked by a literature co-mention")

    t3 = build_advancement_labels(rd)
    for name, df in t3.items():
        df.to_parquet(rd / "labels" / f"{name}.parquet", index=False)
        lo, hi, horizon = ADVANCEMENT_LABEL_FILES[name]
        print(f"labels/{name}.parquet: {len(df):,} pairs entering Phase II {lo}-{hi}, "
              f"{int(df['outcome'].sum()):,} reach Phase III by {horizon}")
    primary = None
    if args.diseases_dir is not None:
        tas = therapeutic_areas(args.diseases_dir)
        tas.to_parquet(rd / "labels" / "therapeutic_areas.parquet", index=False)
        primary = primary_therapeutic_areas(rd, t3["eval"], tas)
        (rd / "labels" / "primary_therapeutic_areas.json").write_text(json.dumps(primary, indent=2))
        print(f"labels/therapeutic_areas.parquet: {tas['disease_id'].nunique():,} diseases; "
              f"{len(primary)} primary areas")
    else:
        print("no --diseases-dir: therapeutic-area tables not written")

    last = last_label_year(t1)
    edge_types = [(e["src"], e["rel"], e["dst"]) for e in meta["edge_types"]]
    summary: dict[str, Any] = {
        "primary_metric": PRIMARY_METRIC,
        "seeds": SEEDS,  # runs per setting, reported as mean ± std
        # T1/T2 scored windows (first, last event year; the cutoff is the first year): a model scored on one
        # trains only on evidence and labels dated before fit_before, by one of the recipes
        "splits": {k: {"window": list(ws[0]), "fit_before": fit_before(k),
                       "stacked_windows": [list(w) for w in fit_windows(fit_before(k))]} for k, ws in SPLITS.items()},
        "recipes": RECIPES,
        # target->disease relations recipe "single" hides (with their rev_ reverses) for the pairs trained on
        "label_relations": {t: [r for _, r, _ in label_edge_types(t, edge_types)] for t in ("discovery", "hypothesis")},
        "last_graph_year": last,
        "min_positives_per_disease": {f"{t}.{s}": k for (t, s), k in MIN_DISEASE_POSITIVES.items()},
        "min_first_trial_phase": {f"{t}.{s}": v for (t, s), v in MIN_FIRST_TRIAL_PHASE.items()},
        "hypothesis_strata": HYPOTHESIS_STRATA,
        "advancement_label_files": {k: {"entry_years": [lo, hi], "outcome_horizon": h}
                                    for k, (lo, hi, h) in ADVANCEMENT_LABEL_FILES.items()},
        "cv_folds": {"advancement": cv_folds("advancement")},
        "tasks": {},
    }
    for fold in summary["cv_folds"]["advancement"]:
        for part, df in advancement_fold_labels(rd, fold).items():
            fold[f"{part}_pairs"], fold[f"{part}_positives"] = len(df), int(df["outcome"].sum())
        print(f"advancement fold val {fold['val']}: train {fold['train_pairs']:,} pairs "
              f"({fold['train_positives']:,} positive), val {fold['val_pairs']:,} ({fold['val_positives']:,})")
    for task, labels in (("discovery", t1), ("hypothesis", t2)):
        summary["tasks"][task] = {}
        for split in SPLITS:
            stats, grade_counts = _split_stats(task, split, labels, n_targets)
            s = slate_summary(stats)
            if task == "discovery":  # T2 positives are not graded
                s["positive_grades"] = grade_counts
            if not stats.empty:
                s["random_expected"] = expected_random(stats)
            cut = fit_before(split)
            s["training_positives"] = {
                "single": len(training_pairs(task, labels, cut)),
                "stacked": sum(len(training_pairs(task, labels, hi + 1, since=lo)) for lo, hi in fit_windows(cut))}
            summary["tasks"][task][split] = s
            print(f"{task:12s} {split:4s} slates={s['n_slates']:,} positives={s.get('n_positives', 0):,} "
                  f"mean_candidates={s.get('mean_candidates', 0):,.0f} training positives before {cut}: "
                  f"single {s['training_positives']['single']:,}, stacked {s['training_positives']['stacked']:,}")

    adv = {}
    for split, (file, _, _) in ADVANCEMENT_SPLITS.items():
        df = advancement_labels(rd, split)
        adv[split] = {"label_file": f"labels/{file}.parquet", "pairs": len(df),
                      "positives": int(df["outcome"].astype(bool).sum()),
                      "positive_rate": round(float(df["outcome"].astype(bool).mean()), 4),
                      "transition_years": [int(df["transition_year"].min()), int(df["transition_year"].max())]}
        print(f"{'advancement':12s} {split:5s} pairs={len(df):,} positives={adv[split]['positives']:,} "
              f"entry_years={adv[split]['transition_years']}")
    summary["tasks"]["advancement"] = adv

    (rd / "benchmark_tasks.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item()
                                                        if isinstance(o, np.generic) else str(o)))
    print(f"✅ {rd / 'benchmark_tasks.json'}")


if __name__ == "__main__":
    main()

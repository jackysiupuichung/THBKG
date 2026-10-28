#!/usr/bin/env python
"""Run the baselines that need no training on both tasks, for val and test.

    python scripts/benchmark/run_cpu_baselines.py --release-dir $RELEASE_DIR --out-dir $OUT

Methods (src/benchmark/baselines.py):
  random         T1: closed-form expectation; T3: a constant score (RS@N = 1)
  open_targets   Open Targets direct association score at the decision year
  genetics       the same over the genetic-association sources

Writes, in --out-dir:
  t1_metrics.csv          method, split, precision/recall/NDCG@K macro-averaged over slates
  t1_per_slate.parquet    the same per slate (year, disease_index)
  t3_predictions.parquet  method, split, pair, transition_year, outcome, score
  t3_summary.csv          RS@N, AUROC, AP: TA mean and "all", per method and split
  t3_by_ta.csv            per primary therapeutic area
  t3_by_year.csv          per entry year (positive rate, AUROC, AP)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.benchmark.advancement_metrics import evaluate_advancement  # noqa: E402
from src.benchmark.baselines import SourceScores, harmonic_score, source_weights  # noqa: E402
from src.benchmark.slates import evaluate, expected_random, macro_average, slate_stats  # noqa: E402
from src.benchmark.tasks import advancement_labels, decision_years, last_label_year, task_slates  # noqa: E402

SPLITS = ("val", "test")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    rd, out = args.release_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    n_targets = json.loads((rd / "metadata.json").read_text())["node_types"]["target"]["num_nodes"]
    weights, genetic = source_weights()
    scores = SourceScores(rd)
    methods = {
        "open_targets": lambda ps: harmonic_score(ps, weights),
        "genetics": lambda ps: harmonic_score(ps, weights, genetic),
    }
    print(f"per-source scores: {scores.key.size:,} rows, {len(scores.sources)} sources ({time.time() - t0:.0f}s)")

    # ---- T1: rank every candidate target in each (disease, year) slate
    t1 = pd.read_parquet(rd / "labels" / "discovery.parquet")
    t1_rows, per_slate = [], []
    for split in SPLITS:
        stats, per = [], {m: [] for m in methods}
        for y in decision_years(split, last_label_year(t1)):
            slates = task_slates("discovery", split, t1, num_targets=n_targets, years=[y])
            if not slates:
                continue
            stats.append(slate_stats(slates))
            ps = scores.as_of(y)
            ps = ps[ps["disease_index"].isin({s.disease_index for s in slates})]
            for m, fn in methods.items():
                pair = fn(ps)
                by_d = {d: (g["target_index"].to_numpy(), g["score"].to_numpy()) for d, g in pair.groupby("disease_index")}

                def score_fn(s, by_d=by_d):
                    full = np.zeros(n_targets)
                    if s.disease_index in by_d:
                        ti, sc = by_d[s.disease_index]
                        full[ti] = sc
                    return full[s.candidates]

                per[m].extend(evaluate(slates, score_fn)[1])
        t1_rows.append({"method": "random", "split": split, **expected_random(pd.concat(stats))})
        for m, rows in per.items():
            t1_rows.append({"method": m, "split": split, **macro_average(rows)})
            per_slate.extend({"method": m, "split": split, **r} for r in rows)
        print(f"T1 {split}: {len(per['genetics']):,} slates ({time.time() - t0:.0f}s)")
    t1m = pd.DataFrame(t1_rows)
    t1m.to_csv(out / "t1_metrics.csv", index=False)
    pd.DataFrame(per_slate).to_parquet(out / "t1_per_slate.parquet", index=False)

    # ---- T3: score each labelled pair at its Phase II entry year
    tas = pd.read_parquet(rd / "labels" / "therapeutic_areas.parquet")
    primary = json.loads((rd / "labels" / "primary_therapeutic_areas.json").read_text())
    preds = []
    for split in SPLITS:
        lab = advancement_labels(rd, split)
        preds.append(lab.assign(method="random", split=split, score=0.0))
        for y, g in lab.groupby("transition_year"):
            ps = scores.as_of(int(y))
            ps = ps[ps["disease_index"].isin(set(g["disease_index"]))]
            for m, fn in methods.items():
                j = g.merge(fn(ps), on=["target_index", "disease_index"], how="left")
                preds.append(j.assign(method=m, split=split, score=j["score"].fillna(0.0)))
    pred = pd.concat(preds, ignore_index=True)
    pred["outcome"] = pred["outcome"].astype(bool)
    pred[["method", "split", "target_id", "disease_id", "transition_year", "outcome", "score"]].to_parquet(
        out / "t3_predictions.parquet", index=False)
    by_ta, summary, by_year = [], [], []
    for (m, split), g in pred.groupby(["method", "split"], sort=False):
        res = evaluate_advancement(g, tas, primary)
        by_ta.append(res["by_ta"].assign(method=m, split=split))
        summary.append(res["summary"].assign(method=m, split=split))
        by_year.append(res["by_year"].assign(method=m, split=split))
    pd.concat(by_ta).to_csv(out / "t3_by_ta.csv", index=False)
    t3s = pd.concat(summary)
    t3s.to_csv(out / "t3_summary.csv", index=False)
    pd.concat(by_year).to_csv(out / "t3_by_year.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nT1 (macro over slates)")
    cols = [c for c in t1m.columns if c.split("@")[0] in ("precision", "recall", "ndcg")] + ["n_slates"]
    print(t1m[["method", "split"] + cols].round(4).to_string(index=False))
    print("\nT3")
    print(t3s.pivot_table(index=["split", "method", "scope"], columns="N", values="RS").round(2).join(
        t3s.groupby(["split", "method", "scope"])[["auroc", "ap"]].first().round(3)).to_string())
    print(f"\n-> {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyse the val-window sweep results.

Reads epoch_metrics.csv from each (variant, seed) run, computes:
  - Spearman ρ between each candidate val signal and test_rr@50 over epochs
  - selection regret = max(test_rr@50) - test_rr@50[argmax(val signal)]
  - val signal non-zero fraction
  - val-selected epoch vs oracle-best epoch

Aggregates per variant (mean ± std over the 3 seeds) and writes a
markdown summary to progress/2026-05-12/VAL_WINDOW_SWEEP.md plus a CSV
of the raw per-run metrics.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_BASE = Path("/gpfs/scratch/bty414/opentarget_evidences/23.06/runs/val_window_sweep")
OUT_DIR   = REPO_ROOT / "progress" / "2026-05-12"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["v1_2011_15", "v2_2013_15", "v3_2014_15", "v4_2015", "v5_2012_13"]
VARIANT_LABEL = {
    "v1_2011_15": "V1 baseline (2011-15)",
    "v2_2013_15": "V2 medium  (2013-15)",
    "v3_2014_15": "V3 near    (2014-15)",
    "v4_2015":    "V4 adjacent (2015)",
    "v5_2012_13": "V5 mid ctrl (2012-13)",
}
SEEDS = [42, 17, 123]
CANDIDATE_SIGNALS = [
    "val_ndcg@10", "val_ndcg@50", "val_ndcg@100",
    "val_rr@10",   "val_rr@50",   "val_rr@100",
]
TEST_TRUTH = "test_rr@50"


def per_run(df: pd.DataFrame) -> dict:
    truth = df[TEST_TRUTH].values
    peak_rr = float(np.max(truth))
    out = {"n_epochs": len(df), "test_rr50_peak": peak_rr}
    for sig in CANDIDATE_SIGNALS:
        if sig not in df.columns:
            continue
        s = df[sig].values
        if np.all(np.isnan(s)) or np.nanstd(s) == 0:
            out[f"{sig}__rho"] = float("nan")
            out[f"{sig}__regret"] = float("nan")
            out[f"{sig}__nonzero_frac"] = float(np.mean(s != 0))
            continue
        rho, _ = spearmanr(s, truth)
        sel_ep = int(np.nanargmax(s))
        rr_at_sel = float(truth[sel_ep])
        out[f"{sig}__rho"]           = float(rho)
        out[f"{sig}__regret"]        = peak_rr - rr_at_sel
        out[f"{sig}__nonzero_frac"]  = float(np.mean(s != 0))
    return out


def main():
    rows = []
    for var in VARIANTS:
        for seed in SEEDS:
            p = RUNS_BASE / f"{var}_s{seed}" / "epoch_metrics.csv"
            if not p.exists():
                print(f"MISSING {p}")
                continue
            df = pd.read_csv(p)
            metrics = per_run(df)
            metrics["variant"] = var
            metrics["seed"] = seed
            rows.append(metrics)
    if not rows:
        print("No runs found. Submit jobs first.")
        return
    raw = pd.DataFrame(rows)
    raw_csv = OUT_DIR / "val_window_sweep_raw.csv"
    raw.to_csv(raw_csv, index=False)
    print(f"Wrote raw per-run metrics: {raw_csv}")

    # Aggregate per variant: mean ± std over seeds
    headline_cols = [c for c in raw.columns if c.endswith(("__rho", "__regret", "__nonzero_frac"))]
    agg = raw.groupby("variant")[headline_cols + ["test_rr50_peak"]].agg(["mean", "std"])
    agg_csv = OUT_DIR / "val_window_sweep_agg.csv"
    agg.to_csv(agg_csv)
    print(f"Wrote aggregated metrics: {agg_csv}")

    # Markdown summary, headline signal = val_ndcg@50
    lines = ["# Val-Window Sweep — Results", "",
             "Five val-window variants × 3 seeds = 15 P3 EAHGT runs.",
             "Train cutoff fixed at ≤2010; test fixed at ≥2016; only val window varies.",
             "Headline metric: Spearman ρ(val_ndcg@50, test_rr@50) across epochs.",
             "", "## Headline (val_ndcg@50 → test_rr@50)", ""]
    lines.append("| Variant | val years | mean ρ | std ρ | mean regret | std regret | mean non-zero frac |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    sig = "val_ndcg@50"
    for var in VARIANTS:
        sub = raw[raw["variant"] == var]
        if sub.empty:
            continue
        rho_mean   = sub[f"{sig}__rho"].mean()
        rho_std    = sub[f"{sig}__rho"].std()
        reg_mean   = sub[f"{sig}__regret"].mean()
        reg_std    = sub[f"{sig}__regret"].std()
        nz_mean    = sub[f"{sig}__nonzero_frac"].mean()
        lines.append(
            f"| {VARIANT_LABEL[var]} | "
            f"{VARIANT_LABEL[var].split('(')[-1].strip(')')} | "
            f"{rho_mean:+.2f} | {rho_std:.2f} | "
            f"{reg_mean:.2f} | {reg_std:.2f} | {nz_mean:.2f} |"
        )

    lines.append("")
    lines.append("## All candidate val signals, mean ρ across seeds")
    lines.append("")
    lines.append("| Variant | " + " | ".join(CANDIDATE_SIGNALS) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(CANDIDATE_SIGNALS)) + " |")
    for var in VARIANTS:
        sub = raw[raw["variant"] == var]
        if sub.empty:
            continue
        cells = [VARIANT_LABEL[var]]
        for sig in CANDIDATE_SIGNALS:
            col = f"{sig}__rho"
            cells.append(f"{sub[col].mean():+.2f}" if col in sub.columns else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Acceptance criteria")
    lines.append("")
    lines.append("- **V4 mean ρ > 0.5 AND mean regret < 1.0** → adopt narrow val; re-run full ablation under this split.")
    lines.append("- **V4 mean ρ in 0.3–0.5** → try V3 or V2 if better; partial win.")
    lines.append("- **V4 mean ρ < 0.3** → distribution shift not fixable by narrowing val alone; move to train-positive subsampling.")
    lines.append("- **V5 (mid-only control) ρ ≈ V4 ρ** → adjacency isn't the cause; sample size or other factor is driving.")
    lines.append("- **V5 ρ < V4 ρ** → adjacency confirmed as the mechanism.")

    md_path = OUT_DIR / "VAL_WINDOW_SWEEP.md"
    md_path.write_text("\n".join(lines))
    print(f"Wrote summary: {md_path}")


if __name__ == "__main__":
    main()

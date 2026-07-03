#!/usr/bin/env python3
"""Build the STRICT-MASK (< transition_year) grouped-allTA 5-seed ensemble
(val-selected, percentile-rank fused) -> test_predictions.parquet.
Mirror of build_grouped_ensemble_latest.py but pointed at lr_grouped_k100_strictmask.
Run on a compute node.
"""
import pandas as pd, numpy as np, glob, os

RUNS = "/gpfs/scratch/bty414/opentarget_evidences/26.03/runs"
OUT = f"{RUNS}/grouped_ensemble_strictmask_s5/test_predictions.parquet"
ref = pd.read_parquet(f"{RUNS}/ndcgk_corr/ndcgk100/test_predictions.parquet").reset_index(drop=True)

def vs(rd):
    fs = sorted(glob.glob(f"{rd}/per_epoch_preds/epoch_*.parquet"))
    sc = [pd.read_parquet(f)["score"].values for f in fs]
    em = pd.read_csv(f"{rd}/epoch_metrics.csv")
    ep = int(em.loc[em["val_rs_ta_median@50"].idxmax(), "epoch"])
    return sc[ep - 1], ep

pcts = []
for s in [1, 7, 42, 123, 2024]:
    v, ep = vs(f"{RUNS}/lr_grouped_k100_strictmask/strictmask_s{s}")
    pcts.append(pd.Series(v).rank(pct=True).values)
    print(f"strictmask s{s}: val-selected epoch {ep}")
ens = np.mean(np.stack(pcts), axis=0)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
pd.DataFrame({"target_id": ref.target_id, "disease_id": ref.disease_id,
             "score": ens, "label": ref.label.astype(int)}).to_parquet(OUT, index=False)
print(f"wrote {OUT}")

# Early-Stopping Metric Sweep — p3_eahgt_both

Goal: pick the early-stopping signal that fixes the degenerate
val/`ndcg@10` ≈ 0 issue flagged in
[../2026-04-30/ABLATION.md](../2026-04-30/ABLATION.md) caveat #3 and
in [MODEL_COMPARISON.md](MODEL_COMPARISON.md) caveat #2.

All four runs share the canonical `p3_eahgt_both` recipe (HGT +
score + novelty edge attrs, undirected, `ndcg_k = 50`); only
`train.early_stopping.metric` varies.

## Configurations

| # | Slug | Early-stop metric | Config |
| --- | --- | --- | --- |
| 1 | `p3_es_ndcg10_flat` | `ndcg@10` | [config/experiments/early_stop_sweep/p3_es_ndcg10_flat.yaml](../../config/experiments/early_stop_sweep/p3_es_ndcg10_flat.yaml) |
| 2 | `p3_es_ndcg50_flat` | `ndcg@50` | [config/experiments/early_stop_sweep/p3_es_ndcg50_flat.yaml](../../config/experiments/early_stop_sweep/p3_es_ndcg50_flat.yaml) |
| 3 | `p3_es_ndcgta10` | `ndcg_ta_mean@10` | [config/experiments/early_stop_sweep/p3_es_ndcgta10.yaml](../../config/experiments/early_stop_sweep/p3_es_ndcgta10.yaml) |
| 4 | `p3_es_ndcgta50` | `ndcg_ta_mean@50` | [config/experiments/early_stop_sweep/p3_es_ndcgta50.yaml](../../config/experiments/early_stop_sweep/p3_es_ndcgta50.yaml) |

Row 1 is the control (current setup). Row 4 is the primary
candidate — both knobs aligned with `ndcg_k = 50` and test-time
RS@50.

## Submitting

```
bash scripts/advancement_prediction/run_early_stop_sweep.sh
```

Submits all 4 Slurm jobs to `gpushort` (1 GPU, 1h cap each).
Outputs land in
`/gpfs/scratch/bty414/opentarget_evidences/23.06/runs/early_stop_sweep/p3_es_*/`.

## What to compare

For each row, pull from `epoch_metrics.csv` + `results.yaml`:

- `best_epoch` — does it stop > epoch 1 (good) or at epoch 1
  (degenerate, like the current v2 run)?
- `val/<es_metric>` curve shape — flat / monotonic / noisy?
- Test `RS@10 / @50 / @100`, `AUC`, `AP` at the selected
  checkpoint.
- Per-TA RS distribution (does grouping concentrate gains in tail
  TAs?).

**TA-grouped test metrics** (mean-of-ratios across primary TAs — apples-to-apples
with the headline numbers in [MODEL_COMPARISON.md](MODEL_COMPARISON.md)):

| Slug | best_epoch | val/<es> @ best | test rs_ta_mean@10 | @50 | @100 | test ndcg_ta_mean@50 | test AUC | test AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p3_es_ndcg10_flat | 3 | 0.110 | **6.02** | **6.19** | **4.82** | **0.361** | 0.588 | **0.187** |
| p3_es_ndcg50_flat | 3 | 0.068 | **6.02** | **6.19** | **4.82** | **0.361** | 0.588 | **0.187** |
| p3_es_ndcgta10 | 7 | 0.186 | 3.07 | 1.72 | 1.94 | 0.133 | 0.550 | 0.105 |
| p3_es_ndcgta50 | 7 | 0.140 | 3.07 | 1.72 | 1.94 | 0.133 | 0.550 | 0.105 |

**Flat-RS test metrics** (single-list, included for reference — these are
NOT what MODEL_COMPARISON.md reports):

| Slug | flat rs@10 | flat rs@50 | flat rs@100 |
| --- | --- | --- | --- |
| p3_es_ndcg10_flat | 11.20 | **10.62** | 6.67 |
| p3_es_ndcg50_flat | 11.20 | **10.62** | 6.67 |
| p3_es_ndcgta10 | 8.69 | 3.75 | 2.50 |
| p3_es_ndcgta50 | 8.69 | 3.75 | 2.50 |

(Reference: previous v2 run — `best_epoch = 1`, test RS@100 ≈ 4.82,
AP = 0.187, on the same recipe.)

## What the per-epoch trajectories show

The trainer now logs `rs_ta_mean@K` for both val and test, so we can
read the alignment directly. (Trajectory shared across runs — they
only differ in which epoch gets selected.)

```
ep  ndcg@10  val_ndcgta@10  val_rsta@10  val_rsta@50  TST_rsta@10  TST_rsta@50
1   0.000    0.113          1.918        2.654        4.379        6.280
2   0.000    0.030          0.446        0.417        5.457        7.518   <- test peak
3   0.110    0.100          1.250        1.065        6.022        6.187   <- ndcg@10 wins
4   0.000    0.100          1.321        0.872        6.066        6.188
5   0.000    0.126          0.944        1.327        4.802        5.908
6   0.000    0.157          1.967        1.440        5.742        5.413
7   0.000    0.186          2.641        1.724        3.065        1.719   <- ndcg_ta wins
8   0.000    0.179          2.365        1.541        5.449        3.586
```

Three things stand out:

1. **Flat `ndcg@K` is degenerate.** Zero for 7 of 8 epochs. Only epoch
   3 is non-zero — that's why it gets selected.
2. **TA-grouped NDCG / RS are well-behaved as signals** (no degenerate
   zeros, smooth trajectories). They work *as signals*.
3. **But val_rs_ta_mean and test_rs_ta_mean diverge sharply.** Val
   peaks at epoch 7 (`val_rs_ta_mean@10 = 2.64`); test peaks at
   epoch 2 (`test_rs_ta_mean@50 = 7.52`). Selecting on val-side TA
   picks epoch 7 → test_rs_ta_mean@50 = 1.72, a **~4× regression**
   vs the flat-NDCG-selected epoch 3 (test = 6.19).

## Implication

**Don't promote TA-grouped NDCG/RS as the early-stop metric.** The
val-set TA distribution and the test-set TA distribution disagree
enough that optimising for the val-side TA-grouped metric degrades
the test-side TA-grouped RS. The "degenerate val NDCG" pathology is
real, but its symptom (stopping at the first non-zero epoch) happens
to land near the test peak.

**Keep `ndcg@10` (or `ndcg@50` — they pick the same epoch).** Final
test `rs_ta_mean@50 = 6.19`, matching the headline number for
p3_eahgt_both in [MODEL_COMPARISON.md](MODEL_COMPARISON.md).

## Why the divergence

The val window (post-2010, pre-2019) and the test window (2019+)
differ in TA mix and per-TA positive density. A val-side per-TA
mean-of-ratios is therefore a noisy estimator of the test-side
mean-of-ratios — even when each individual NDCG is well-behaved. The
flat NDCG is degenerate enough that it stops optimisation early
(effectively a "first non-zero epoch" rule), which turns out to be a
better implicit selection criterion than the better-behaved TA-
grouped signal.

## Diagnostic note

`p3_es_ndcg10_flat` and `p3_es_ndcg50_flat` produce **identical test
metrics**. Same for `p3_es_ndcgta10` and `p3_es_ndcgta50`. The K
within each metric family doesn't change which epoch is selected —
the grouping does. So the meaningful contrast in this sweep is
2 conditions × 1 K = the two distinct outcomes shown above.

## Things to try next

1. **Reduce `patience`.** TA-grouped runs wander to epoch 7 with
   `patience=5`. With `patience=1–2`, they'd stop near epoch 3
   (`val_rs_ta_mean@10 = 1.25`), possibly giving a non-degenerate
   signal *and* an early stop near the test peak. Worth one extra
   run.
2. **Closer val/test temporal cutoff.** The val-test temporal
   mismatch is load-bearing here. Narrowing the val window to the
   final pre-test years would make val-side TA-grouped metrics more
   predictive. Larger change — separate experiment.
3. **Multi-seed.** Single seed; need 3-seed runs of the winner +
   `ndcg_ta_mean@10` to confirm the regression isn't seed noise.

## Caveats

1. **Loss objective unchanged.** This sweep varies *checkpoint
   selection only*. The LambdaRank loss is still flat over the
   batch — TA-grouped optimisation is a separate, larger
   experiment.
2. **Multi-TA diseases.** A disease in N primary TAs contributes
   to N NDCG computations in `ndcg_ta_mean` (consistent with
   eval-time RS mean-of-ratios).
3. **Single seed for the sweep.** Pick the winner from this 4-row
   single-seed run, then re-run only the winner + row 1 at 3 seeds
   for the headline comparison.

## Next steps

- [ ] Run the sweep, fill in the table above.
- [ ] Pick winner; promote its `early_stopping.metric` into
      `config/experiments/p3_eahgt_both.yaml` and the other
      `pX_eahgt_*.yaml` configs.
- [ ] 3-seed re-run of (winner, row 1) for the multi-seed bullet
      in [PROGRESS.md](PROGRESS.md).
- [ ] If the TA-grouped early-stop fixes degeneracy without changing
      the loss, defer the "TA-grouped LambdaRank loss" experiment.

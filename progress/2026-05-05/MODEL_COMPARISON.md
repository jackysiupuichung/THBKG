# Model Comparison & Ablation — 2026-05-05

This document supersedes [../2026-04-30/ABLATION.md](../2026-04-30/ABLATION.md).
It folds the proposed-model vs. baselines comparison and the per-
component ablation into a single figure set, anchored to the
**undirected_v1** training recipe (canonical hyperparameters from
`config/experiments/advancement_lambdarank_undirected_v1.yaml`).

All runs use the 23.06 graph, the same train/val/test split, and the
same evaluation pipeline (`evaluate_advancement.py`). The only knobs
that vary between rows are the architecture and the edge-feature
subset.

Outputs live in
[advancement_data/results/ablation_v2/](../../advancement_data/results/ablation_v2/)
(in-distribution comparison + ablation) and
[advancement_data/results/external/](../../advancement_data/results/external/)
(undirected aggregation variants v1–v5 + the directed reference).

## Models in the comparison

| Slug | Architecture | Edge feats | RTE | Role |
| --- | --- | --- | --- | --- |
| OTS | OpenTargets associationScore | — | — | IR reference |
| RDG | Restart-Diffusion-Graph (no time, positive) | — | — | IR reference |
| b1_hgt | HGT | — | — | Pure HGT baseline |
| b3_gatv2 | GATv2 | — | — | Attention baseline |
| b6_rgcn | R-GCN | — | — | Relation-conditioned baseline |
| b7_compgcn | CompGCN | — | — | Composition-based baseline |
| p1_eahgt_score | HGT | score | — | EAHGT, score only |
| p2_eahgt_novelty | HGT | novelty | — | EAHGT, novelty only |
| **p3_eahgt_both** | HGT | score + novelty | — | **Proposed full EAHGT** |

(b2/b4/b5 from the previous matrix were dropped: b2 = RTE-on, kept
out because RTE was shown to hurt NDCG previously; b4/b5 were
GATv2 + edge-feature variants and were redundant once b3 was
restandardised under the canonical recipe and once HGT-based EAHGT
ablations covered the edge-feature axis.)

## Headline: RR@N curves (all models)

Mean-of-ratios RR over primary therapeutic areas, swept over the
top-K limit. p3_eahgt_both leads at every K from 10 to 100.

![RR by limit, all models](../../advancement_data/results/ablation_v2/plots/relative_risk_by_limit_katz95.png)

## Lift over the RDG IR baseline

Δ(RR) vs RDG, by limit. Positive bars indicate the model beats the
strongest IR reference.

![Δ RR vs RDG by limit](../../advancement_data/results/ablation_v2/plots/relative_risk_delta_vs_rdg.png)

## Per-therapeutic-area RR

Heatmap of RR@K broken out per therapeutic area, all models.

![RR by therapeutic area heatmap](../../advancement_data/results/ablation_v2/plots/relative_risk_by_ta_heatmap.png)

## Per-stratum RR (pioneer × evidence)

Stratified RR@N by (pioneer × evidence type) — shows whether the
SOTA holds for hard cases (pioneer targets, evidence-free pairs).

![RR by limit by stratum](../../advancement_data/results/ablation_v2/plots/relative_risk_by_limit_by_stratum.png)

## Classification metrics (AUC / AP) per TA

![Classification metrics by TA](../../advancement_data/results/ablation_v2/plots/classification_metrics_by_ta.png)

## Classification metrics per stratum × TA

![Classification metrics by stratum and TA](../../advancement_data/results/ablation_v2/plots/classification_metrics_by_stratum_by_ta.png)

## Per-TA RR distributions

Boxplots of per-TA RR for each model — beyond the mean-of-ratios
headline, this shows dispersion and worst-case TAs.

![RR distributions per TA](../../advancement_data/results/ablation_v2/plots/rr_distributions_ta.png)

## Findings

### 1. The full EAHGT is the SOTA across every operating point
- p3 leads RR@N at every K from 10 to 100 (top-of-list and global).
- p3 dominates AP and is within ~0.05 of the best AUC.
- The lift over both IR references is roughly 3× at RR@100 (see
  the Δ-vs-RDG figure).
- The lift over the strongest non-EAHGT baseline (b6_rgcn) is
  roughly 1.5× at RR@100.

### 2. Edge attributes contribute, jointly more than individually
- p1 (score only) and p2 (novelty only) each beat plain HGT (b1) on
  AUC and AP — both edge features are individually informative.
- Joint use (p3) is the only configuration achieving the top-row AP
  and the top-K RR. Score and novelty are complementary, not
  redundant.
- p2 alone is weak on top-K ranking, suggesting novelty is a
  calibration / distribution signal that needs the score channel
  to anchor it at the top of the list.

### 3. The encoder choice matters — HGT > R-GCN > GATv2 > CompGCN
- Among edge-feature-free baselines (b1, b3, b6, b7), R-GCN is the
  strongest on RR@N — its per-relation weight matrices are doing
  real work on this heterogeneous graph.
- CompGCN ranks well on global AUC but its scores don't concentrate
  positives at the head of the list.
- Plain HGT (b1) is surprisingly weak without edge features — the
  heterogeneous attention with nothing to attend over
  underperforms simpler relation-conditioned schemes.
- Adding edge attributes turns HGT into the strongest model by a
  large margin (b1 → p3). HGT *requires* edge attributes to
  realise its capacity here.

### 4. Heterogeneity matters but isn't sufficient
- All HGT-based variants with edge features (p1, p3) beat every
  non-HGT baseline on RR@100. p2 is the exception, and it's the
  novelty-only ablation — i.e., this is an edge-feature failure
  mode, not an HGT failure mode.
- The combination of (a) per-relation attention (HGT) and (b) edge
  attributes driving the attention is what produces the SOTA.

### 5. Aggregation scheme — undirected variants v1–v5

Side-by-side of alternative undirected aggregation schemes (v1 is
the proposed setup) plus the previous directed SOTA and an additive
variant. Files in
[advancement_data/results/external/plots/](../../advancement_data/results/external/plots/).

![External: RR by limit](../../advancement_data/results/external/plots/relative_risk_by_limit_katz95.png)

![External: Δ RR vs RDG](../../advancement_data/results/external/plots/relative_risk_delta_vs_rdg.png)

![External: classification metrics by TA](../../advancement_data/results/external/plots/classification_metrics_by_ta.png)

- v1 wins on top-K (RR@10).
- v2 is competitive in the mid-range.
- The previous directed SOTA from
  [../2026-04-15/RESULTS.md](../2026-04-15/RESULTS.md) is decisively
  beaten by every undirected variant except v3.

## Caveats

1. **Single seed.** All RR / NDCG numbers are based on one training
   run per row. Multi-seed reruns are needed before publication.
2. **Val metric is brittle.** Current early-stop uses flat
   `ndcg@10`, which sits at 0 for many epochs across HGT variants
   and makes "best epoch" essentially the first non-degenerate
   epoch. The TA-grouped `ndcg_ta_mean@K` needs to be restored.
3. **No statistical significance test yet.** RR differences look
   large but should be confirmed with paired bootstrap once
   multi-seed runs land.

## Reproducing

```
# baselines
bash scripts/advancement_prediction/run_b1_hgt.sh
bash scripts/advancement_prediction/run_b3_gatv2.sh
bash scripts/advancement_prediction/run_b6_rgcn.sh
bash scripts/advancement_prediction/run_b7_compgcn.sh

# proposed-model ablations
bash scripts/advancement_prediction/run_p1_eahgt_score.sh
bash scripts/advancement_prediction/run_p2_eahgt_novelty.sh
bash scripts/advancement_prediction/run_p3_eahgt_both.sh

# evaluation
python evaluate_advancement.py \
  --only b1_hgt,b3_gatv2,b6_rgcn,b7_compgcn,\
p1_eahgt_score,p2_eahgt_novelty,p3_eahgt_both
```

## Next steps

- [ ] Multi-seed reruns (target n=3) for all rows above.
- [ ] Restore `ndcg_ta_mean@K` early-stop signal.
- [ ] Paired-bootstrap significance vs RDG / vs p1.
- [ ] Fix prospective-eval zeros (separate doc — see PROGRESS.md).

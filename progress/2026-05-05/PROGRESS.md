# Progress — 2026-04-30 → 2026-05-05

## What got done

- **Refreshed model-comparison + ablation matrix.** Re-ran every
  baseline / ablation under the canonical undirected LambdaRank recipe
  on the 23.06 graph and rebuilt the comparison table. The proposed
  full **EAHGT (p3, score + novelty edge attributes)** holds the SOTA
  by a wide margin: **RR@100 = 4.82** vs RDG 1.68 / OTS 1.64; AP =
  0.187 vs 0.117 / 0.090. Full numbers and per-stratum breakdown in
  [MODEL_COMPARISON.md](MODEL_COMPARISON.md).
- **New baselines added.** Replaced the previous GATv2-only baselines
  (b3/b4/b5) with a single canonical **GATv2 (b3)** entry and added
  two relation-aware baselines: **R-GCN (b6)** and **CompGCN (b7)**.
  This makes the architecture comparison more defensible — we now
  span attention (GATv2/HGT), relation-conditioned message passing
  (R-GCN), and composition-based (CompGCN).
- **Tuning runs for the new baselines.** `b3_gatv2_tune`,
  `b6_rgcn_tune`, `b7_compgcn_tune` configs added; selected configs
  fed back into the canonical comparison.
- **Undirected variants v2–v5.** Side-by-side runs of alternative
  undirected aggregation schemes (additive variant included for
  reference). v1 remains the proposed setup; v2 is competitive on
  RR@50–60 but weaker on RR@100. See the "External" section of
  [MODEL_COMPARISON.md](MODEL_COMPARISON.md).
- **Prospective evaluation scaffolding.** Added
  `evaluate_prospective_standalone.py` and
  `run_prospective_p3_eahgt.sh` to score every (target, disease) pair
  for a held-out cohort of 156 prospective diseases using a 2015
  cutoff. End-to-end pipeline runs (loads checkpoint, builds context
  graph, scores 1.4M candidate pairs across 73 evaluable diseases),
  but **all P@K / R@K currently return 0.0** — bug in the future-
  positive lookup or in score / candidate alignment. Investigating.

## Headline figure (test set, 23.06 graph)

RR@N curves, all 9 models. p3_eahgt_both leads at every K from 10
to 100. Full breakdown in [MODEL_COMPARISON.md](MODEL_COMPARISON.md).

![RR by limit, all models](../../advancement_data/results/ablation_v2/plots/relative_risk_by_limit_katz95.png)

![Δ RR vs RDG by limit](../../advancement_data/results/ablation_v2/plots/relative_risk_delta_vs_rdg.png)

## In progress

- **Prospective evaluation debug.** The 2015-cutoff prospective run
  on 73 diseases / 1.4M candidate pairs returns all-zero P@K and R@K
  at K ∈ {100, 200, 500}. Need to verify (a) the future-positive
  parquet aligns with the candidate index, (b) scores aren't being
  zeroed by the prior-precedence mask, (c) the model checkpoint
  matches the graph build it's being scored against.
- **Explainability.** Subgraph-around-prediction extraction not yet
  implemented; literature sidecar (`_literature.parquet`,
  `(source, target, datasource, year) → PMIDs`) is in and queryable
  from previous week.

## To do

- Fix the prospective-eval zeros and produce a real P@K / R@K table
  for the 156-disease cohort.
- DERI Day poster (2026-05-07) — bring the new SOTA + full ablation +
  baselines table.
- Midway presentation deck.
- Multi-seed re-runs of the comparison table (current numbers are
  single-seed).
- Restore `ndcg_ta_mean@K` early-stop signal in
  `train_advancement_lambdarank.py` (current `ndcg@10` early-stop is
  degenerate for HGT variants).
- Subgraph extraction wired to literature sidecar for the prediction
  walk-through.

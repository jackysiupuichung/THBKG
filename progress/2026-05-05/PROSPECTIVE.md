# Prospective Target Discovery — Initial Findings

Evaluating the trained EAHGT (`p3_eahgt_both_lambdarank_v2`, replicates `undirected_v1`) as a recommender for **novel** target–disease pairs.

## Setup

- **Cutoff year**: 2015. Candidate pool: all targets that had **no clinical-trial precedence** with the disease by 2015.
- **Positive set**: target–disease pairs where the first clinical-trial or advancement edge appears **strictly after 2015**, excluding pairs with any precedence at or before 2015.
- **Cohort**: 156 diseases curated in [advancement_data/prospective_diseases.csv](../../advancement_data/prospective_diseases.csv); after filtering to in-graph + ≥1 future positive, **73 diseases scored**.
- **Cutoffs**: K = 100, 200, 500.
- **Outputs**: [runs/p3_eahgt_both_lambdarank_v2/prospective/](../../../../../../gpfs/scratch/bty414/opentarget_evidences/23.06/runs/p3_eahgt_both_lambdarank_v2/prospective/) — `prospective_per_disease.csv`, `prospective_macro.csv`, `prospective_predictions.parquet`.

## Headline result

| K | Macro P@K | Macro R@K | n_diseases |
| --- | --- | --- | --- |
| 100 | **0.000** | **0.000** | 73 |
| 200 | **0.000** | **0.000** | 73 |
| 500 | **0.000** | **0.000** | 73 |

**Zero positives surfaced in the top 500 for any of the 73 diseases.** This is significantly worse than random — the candidate pool is ~19,289 targets per disease, with ~10–90 positives each. Random ranking would expect ~1 positive in the top 500 for an average disease.

## Diagnosis — the model is anti-ranking future positives

Inspecting the predictions parquet:

- **Score variance is real**: 1.23M unique scores across 1.41M predictions; std = 0.07; range [0.11, 0.95]. The model is NOT producing constant outputs.
- **Score distributions diverge by label**: positives have scores in [0.14, 0.39] (mean 0.20); negatives span [0.14, 0.95] (mean 0.22). Negatives can score *much higher* than positives ever do.
- **Positives systematically rank low**: mean position fraction across diseases = **0.69** (random = 0.5; lower = better). Positives sit in the bottom third on average.
  - **60 of 73 diseases**: positives rank in the bottom half.
  - **30 of 73 diseases**: positives rank in the bottom 25%.
  - Only 13 diseases have positives in the top half.
- **Alzheimer's case study (MONDO_0004975, 38 future positives, 19,167 candidates)**: median rank of positives = 13,993. Best positive ranks 988th. Top-20 by score contains zero positives.

This is not a calibration bug or a bad checkpoint — the model produces a coherent, high-variance ranking that is **negatively correlated with future positivity**.

## Likely cause: task mismatch, not implementation bug

The advancement model was trained to predict **Phase 2 → Phase 3 progression**, where positives are pairs that already have clinical-trial evidence at training time. What it learned to score high: target–disease pairs that **already have rich evidence networks** — lots of multi-hop graph paths, established by prior trials.

The prospective eval asks the opposite question. The candidate-pool filter explicitly *excludes* pairs with any clinical-trial precedence by 2015, so by construction every prospective candidate is **novel** (no prior trial). The future positives are therefore exactly the targets the training-time model learned to *down-weight* — they look like "no clinical evidence yet" pairs, which during training were the negatives.

The negative result is **honest** (no leakage detected, no scoring bug) but reflects a fundamental mismatch between training objective and prospective-discovery framing.

## What to try next

Three options, in increasing departure from the current setup:

1. **Relax the candidate-pool filter** (no precedence exclusion). Score the full target × disease grid. "Positive" = any new trial-related edge after cutoff. Aligns the eval with the training distribution; tests "where will the next phase advancement happen" rather than "which novel target will enter trials."

2. **Change the positive set**. Keep the candidate pool but redefine positives to include any new evidence (literature, somatic, etc.) post-cutoff, not just clinical-trial edges. Brings positives closer to the training-time advancement signal.

3. **Train a model for prospective discovery**. Reformulate the link-prediction objective: predict edges that *will appear* given the graph state at year T, with positives sampled from year T+1 onward. Different model class, not just different eval. This is the cleanest path but requires a separate training run.

Option (1) is the smallest change and tells us whether the issue is the candidate-pool filter or something deeper. Recommend trying it first.

## Performance notes

- Single-loader concatenated pass (~1.4M candidate pairs in one `LinkNeighborLoader` call) completed in ~5–10 min on `gpushort`. Earlier per-disease loops produced ~30 cosmetic `MultiProcessingDataLoaderIter.__del__` warnings (harmless, but eliminated by the single-loader refactor).
- Zero-positive diseases (83 of 156) are correctly skipped before the GPU pass.
- Candidate pool is essentially the full target list: mean = 19,260, max = 19,316, total targets = 19,316. Prior-precedence filter only removes 0–389 targets per disease.

## Proposal — first-appearance temporal link prediction

The negative result above is a training/eval distribution mismatch, not an architecture problem. Reformulate the task so the train and prospective-deploy distributions coincide: every (t, d) pair the model sees — train, val, eval — should have **zero clinical-trial edges in its input**. The trial-halo problem then disappears not because we mask it, but because by construction it does not exist yet at the moment we encode the pair.

### The invariant

Anchor the feature snapshot at **τ(t, d) = the year of the pair's first non-trial edge**. Under [config/edge_schema.yaml](../../config/edge_schema.yaml), trial edges are `clinical_trial`, `modulated_by`, and `clinically_associated` (datasource `clinical_precedence`); the remaining ~17 edge types (genetic, somatic, pathway, expression, literature, etc.) are non-trial. The snapshot at τ(t, d) is, by definition, the moment the pair becomes graph-visible *via upstream evidence only*. Train and deploy both query the model on pre-trial pairs.

### Concrete recipe

1. For each (t, d) pair, compute τ(t, d) = min `year` over non-trial edges between t and d.
2. Snapshot the graph at τ(t, d). Encode the pair from that snapshot — either per-pair subgraphs via `LinkNeighborLoader` with a pair-specific time mask, or bucketed yearly snapshots with a year embedding (cheaper, slight intra-bucket leakage but probably fine).
3. **Label** = 1 if any clinical-trial edge between t and d appears in (τ, τ + H] for a fixed horizon H (suggest H = 5 years); 0 otherwise.
4. **Negatives** = pairs with τ defined but no trial edge in horizon. Same feature support as positives; only the label differs. **Hard negatives** = highly-connected-but-never-trialed targets at τ — these force the encoder to learn beyond a degree prior.
5. **Held-out evaluation** = pairs whose τ falls in a held-out year band (e.g., τ ∈ [2018, 2020] held out, horizon to 2025). The 156-disease cohort in [advancement_data/prospective_diseases.csv](../../advancement_data/prospective_diseases.csv) can be reused as the disease filter.

### Contrast with GATher (arXiv:2409.16327)

GATher fixes a single global cutoff (2018) and varies *label* granularity (Phase 1/2/3 efficacy/safety outcomes) to handle pairs at different evidence stages. This proposal fixes the *causal moment* (first appearance) and uses a single binary label, removing trial-halo features by construction rather than by label decomposition. Both approaches unify advancement and prospective discovery — they differ in which axis (label richness vs feature regime) absorbs the heterogeneity. They are not mutually exclusive: a two-head model could combine first-appearance binary prediction with stage-specific outcome heads.

### Why this addresses the negative result above

The current advancement model is trained on pairs that already have trial edges plus the post-trial evidence halo. The 2015-cutoff prospective candidate pool explicitly excludes pairs with prior trials, so by construction those candidates lack the features the model learned to weight — which is what produces the position-fraction = 0.69 inversion documented above. Under the reformulation, train and deploy distributions are both "pre-trial pairs," and the covariate shift collapses.

### Costs and caveats

- **τ itself is informative** (era effect). A pair with τ = 2008 had longer to accumulate non-trial evidence pre-cutoff, and longer follow-up post-cutoff, than one with τ = 2020. Either condition on τ explicitly via year embedding, restrict training to a narrow τ band, or accept partial era leakage.
- **Right-censoring**: pairs with recent τ have shorter follow-up. Fixed horizon H handles this cleanly; survival framing is an alternative if H feels too restrictive.
- **Selection bias does not vanish.** The model learns "what gets trialed next," not "what would work if trialed." Same caveat applies to GATher and to any retrospective evaluation against trial-derived labels.
- **Phase 2 → Phase 3 progression is not expressible** in this framing — positives by construction have no trials. This proposal *replaces* rather than augments the progression task; if both are wanted, a two-head model is required.

### Pre-build verifications

Before committing to a build, two quick checks:

**(a) Coverage of the framing.** How many (t, d) pairs have a non-trial edge year strictly earlier than their first trial edge? If most pairs' first graph appearance *is* a trial edge, τ collapses to the trial year and the proposal degenerates. Check against [advancement_data/datasets/evaluation_dataset.zarr](../../advancement_data/datasets/evaluation_dataset.zarr) and the literature sidecar parquet.

**(b) Year-granularity reliability.** Some OT datasources have noisy or imputed timestamps. Decide whether yearly buckets are safe, or whether finer per-edge time masks are needed (especially for `literature` and `affected_pathway`).

### Expected qualitative effect

- Lose some advancement-task RR@K (fewer trial-halo features available).
- Gain non-trivial prospective P@K (positives now in-distribution).
- Feature attribution should shift: genetic / mechanism / pathway edges rise; literature edges that historically lag trials fall. The attribution shift, not just the headline metric, is the most publishable evidence the reformulation is doing what it claims.

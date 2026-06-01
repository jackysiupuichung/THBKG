# Training Details: Hyperparameters & Model Tuning

---

## Overview

All models predict drug advancement (target–disease pairs advancing to a higher clinical phase) on a heterogeneous temporal graph built from Open Targets 23.06. The task is link prediction on the `advancement` edge type, evaluated temporally: training on edges ≤ 2015, testing on edges ≥ 2016.

---

## Experiment Variants

| ID | Name | Model | Edge Features | Loss | Notes |
|----|------|-------|---------------|------|-------|
| B1 | HGT | HGT | None | Focal BCE | Baseline |
| B2 | HGT + RTE | HGT | None | Focal BCE | Adds relative temporal encoding |
| B3 | GATv2-score | GATv2 | Score only | Focal BCE | |
| B4 | GATv2-novelty | GATv2 | Novelty only | Focal BCE | |
| B5 | GATv2-both | GATv2 | Score + novelty | Focal BCE | |
| P1 | EA-HGT-score | HGT | Score only | Focal BCE | Edge-aware HGT |
| P2 | EA-HGT-novelty | HGT | Novelty only | Focal BCE | |
| P3 | EA-HGT-both | HGT | Score + novelty | Focal BCE | Tuned via Optuna |
| LR | LambdaRank | HGT | Score + novelty | LambdaRank | Ranking objective; our best model |

---

## Hyperparameters by Experiment

### B1 — HGT (Baseline)

| Parameter | Value |
|-----------|-------|
| hidden_dim | 32 |
| num_heads | 4 |
| num_layers | 1 |
| dropout | 0.25 |
| use_rte | false |
| use_edge_features | false |
| learning_rate | 7.0174e-4 |
| weight_decay | 1.3383e-6 |
| batch_size | 256 |
| num_neighbors | [10, 20] |
| epochs | 50 |
| focal_gamma | 2.0 |
| early_stopping_patience | 10 |
| early_stopping_metric | val rs@100 |

### B2 — HGT + RTE

Same as B1 except `use_rte: true`.

### B3 / B4 / B5 — GATv2 Variants

Same hyperparameters as B1 except:

| | B3 | B4 | B5 |
|--|----|----|-----|
| use_edge_features | true | true | true |
| edge_feature_columns | [0] | [1] | [0, 1] |
| edge_feat_dim | 1 | 1 | 2 |

### P1 / P2 — Edge-Aware HGT (score / novelty)

Same hyperparameters as B1 except:

| | P1 | P2 |
|--|----|-----|
| use_edge_features | true | true |
| edge_feature_columns | [0] | [1] |
| edge_feat_dim | 1 | 1 |

### P3 — Edge-Aware HGT (score + novelty, tuned)

Tuned with Optuna (50 trials). Best configuration:

| Parameter | Value |
|-----------|-------|
| hidden_dim | 32 |
| num_heads | 1 |
| num_layers | 2 |
| dropout | 0.2 |
| use_rte | false |
| use_edge_features | true |
| edge_feature_columns | [0, 1] |
| edge_feat_dim | 2 |
| learning_rate | 2.8805e-4 |
| weight_decay | 2.1414e-5 |
| batch_size | 512 |
| num_neighbors | [30, 20] |
| epochs | 50 |
| focal_gamma | 2.0 |
| eta_min | 1e-6 |
| early_stopping_patience | 10 |
| early_stopping_metric | val rs@100 |

### LambdaRank — HGT with LambdaRank Loss (best model)

Tuned with Optuna (50 trials). Best configuration:

| Parameter | Value |
|-----------|-------|
| hidden_dim | 128 |
| num_heads | 4 |
| num_layers | 2 |
| dropout | 0.5 |
| use_rte | false |
| use_edge_features | true |
| edge_feature_columns | [0, 1] |
| edge_feat_dim | 2 |
| learning_rate | 1e-4 |
| weight_decay | 5e-3 |
| batch_size | 512 |
| num_neighbors | [20, 10] |
| epochs | 50 |
| cosine_t_max | 10 |
| eta_min | 1e-6 |
| lambdarank_sigma | 1.0 |
| lambdarank_ndcg_k | 100 |
| early_stopping_patience | 5 |
| early_stopping_metric | ndcg@10 |

---

## Model Architectures

### HGT Encoder

Standard Heterogeneous Graph Transformer (Hu et al., 2020) extended with optional edge-feature injection and relative temporal encoding (RTE).

1. **Input projection**: per-node-type `Linear(input_dim → hidden_dim)`
2. **Message passing**: `num_layers` stacks of `HGTConv`
3. **Output**: per-node embeddings of dimension `hidden_dim`

**HGTConv attention (per edge type)**:

```
α_ij = softmax( (q_i · k_j) / sqrt(d_k) * p_rel * ef_scalar )
```

- `p_rel` — learnable per-edge-type relation scalar `[1, heads]`
- `ef_scalar` — edge-feature projection `[edge_feat_dim → heads]` (only when `use_edge_features=true`)
- RTE adds a Fourier-based positional encoding to keys when `use_rte=true`
- Skip connections via learnable per-node-type sigmoid gate

### GATv2 Encoder

Wraps `GATv2Conv` in `HeteroConv` with per-edge-type message passing, summed across edge types. Optionally injects edge features via `edge_dim`. Same input projection and layer norm structure as HGT.

### Decoder (shared)

`DualHeadDecoder` takes the concatenated source and destination embeddings `[2 × hidden_dim]` (optionally + a time embedding when `use_recency=true`) and produces two scalar outputs:

```
z = ReLU(Linear(2·h + t → h)) → Dropout
z = ReLU(Linear(h → h/2))    → Dropout
logits_exist = Linear(h/2 → 1)   # binary link existence
logits_prob  = Linear(h/2 → 1)   # advancement score / rank
```

Default decoder dropout: 0.1.

### Recency Conditioning (LambdaRank model)

A `TimeEncoder` maps the entry year of each candidate pair to a `time_dim`-dimensional embedding before the decoder:

- Input: normalized year `t ∈ [0, 1]`
- Architecture: `Linear(1 → time_dim/2)` → `sin` and `cos` → concatenate → projection
- `time_dim=8` in the recency-conditioned variant

---

## Optimizer & Scheduler

All experiments use **AdamW** with **Cosine Annealing**:

| Setting | Baseline/P1–P3 | LambdaRank |
|---------|---------------|------------|
| Optimizer | AdamW | AdamW |
| lr | 7.0174e-4 (B1–P2); 2.8805e-4 (P3) | 1e-4 |
| weight_decay | 1.3383e-6 (B1–P2); 2.1414e-5 (P3) | 5e-3 |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| T_max | num_epochs | 10 |
| eta_min | 1e-6 | 1e-6 |
| Gradient clipping | max_norm=1.0 | max_norm=1.0 |

---

## Loss Functions

### Focal BCE (B1–P3)

Focal loss down-weights easy negatives, focusing training on hard examples:

```
BCE   = binary_cross_entropy_with_logits(logits, labels, pos_weight)
p_t   = σ(logits) · labels + (1 − σ(logits)) · (1 − labels)
loss  = BCE · (1 − p_t)^γ        γ = 2.0
```

`pos_weight = n_neg / n_pos` computed per training split to handle class imbalance.

### LambdaRank (LR)

Pair-wise ranking loss weighted by the NDCG gain difference between pairs:

```
For pairs (i, j) where label_i > label_j:
  Δg   = |2^l_i − 2^l_j|           (gain difference)
  Δd   = |1/log2(r_i+1) − 1/log2(r_j+1)|   (discount difference)
  ΔNDCG_ij = Δg · Δd / IDCG
  loss_ij  = ΔNDCG_ij · log(1 + exp(−σ (s_i − s_j)))
```

`σ = 1.0`. Normalized by the count of non-trivial pairs. Numerically stabilized via softplus.

---

## Data Splits & Temporal Filtering

| Split | Criterion |
|-------|-----------|
| Train | `edge_time ≤ cutoff_year` (default 2010) |
| Val | `cutoff_year < edge_time ≤ 2015` |
| Test | `edge_time ≥ 2016` |

Neighborhood sampling uses `LinkNeighborLoader` with `temporal_strategy="last"` so that only edges observed before the query timestamp are included in the sampled subgraph.

---

## Evaluation Metrics

| Category | Metrics |
|----------|---------|
| Classification | Precision, Recall, F1, MCC, ROC-AUC, Avg. Precision, Brier score, Log loss |
| Ranking (P@K) | Precision@K, Recall@K, AP@K for K ∈ {10, 30, 50, 100} |
| Relative Success | RS@K for K ∈ {10, 20, 30, 50, 90, 100} |
| NDCG | NDCG@K for K ∈ {10, 30, 50, 100} |

Primary early-stopping metric: **val RS@100** (B1–P3) / **val NDCG@10** (LambdaRank).

---

## Hyperparameter Search

### Optuna — P3 search space (50 trials, TPE sampler)

| Parameter | Search space |
|-----------|-------------|
| hidden_dim | {32, 64, 128} |
| num_heads | {1, 2, 4} |
| num_layers | {1, 2} |
| dropout | [0.0, 0.3] step 0.05 |
| lr | log-uniform [5e-5, 1e-3] |
| weight_decay | log-uniform [1e-6, 1e-3] |
| batch_size | {256, 512} |
| num_neighbors[0] | {10, 20, 30} |
| num_neighbors[1] | {5, 10, 20} |
| focal_gamma | {null, 2.0} |

### Optuna — LambdaRank search space (50 trials)

| Parameter | Search space |
|-----------|-------------|
| hidden_dim | {32, 64, 128} |
| num_heads | {1, 2} |
| num_layers | {1, 2} |
| dropout | [0.1, 0.6] step 0.1 |
| lr | log-uniform [3e-5, 5e-4] |
| weight_decay | log-uniform [1e-4, 1e-2] |
| cosine_t_max | {5, 10, 20} |
| batch_size | {256, 512, 1024} |
| num_neighbors[0] | {10, 20, 30} |
| num_neighbors[1] | {5, 10, 20} |
| lambdarank_sigma | {0.5, 2.0} |

### Search observations

- OOM occurs at `num_layers ≥ 2` combined with `hidden_dim ≥ 128` on the full graph; deeper models require smaller hidden dims.
- `num_heads=1` frequently wins in P3 — a single head per edge-type relation is sufficient when edge features already provide evidence-level signal.
- Setting `focal_gamma=null` (plain BCE with pos_weight) is competitive; focal loss helps mainly when the validation set contains many hard negatives early in training.
- LambdaRank benefits from higher weight decay (5e-3) compared to focal BCE runs (≤1e-5), consistent with a ranking objective that relies on score ordering rather than calibrated probabilities.

---

## Regularization Summary

| Technique | Value / Setting |
|-----------|----------------|
| Dropout (encoder + decoder) | 0.1–0.5 (config-dependent) |
| L2 weight decay | 1.3383e-6 – 5e-3 (config-dependent) |
| Gradient clipping | max_norm = 1.0 |
| Early stopping | patience 5–10 epochs |
| Class imbalance | pos_weight = n_neg / n_pos (BCE runs) |
| Focal loss | γ = 2.0 (down-weights easy negatives) |

# EAHGT — Model Architecture and Hyperparameters

Edge-Attributed Heterogeneous Graph Transformer (EAHGT) trained with LambdaRank loss for clinical advancement prediction. Configuration corresponds to `p3_lambdarank` (config: `config/experiments/p3_eahgt_both.yaml`).

---

## Architecture Overview

EAHGT is a heterogeneous graph transformer that operates over a bipartite target–disease graph. It encodes each node via multi-layer message passing, then scores target–disease pairs with a two-head MLP decoder. The key distinguishing features relative to the RDG baseline are: (1) graph-structured representations that capture multi-hop biological relationships; (2) edge attribute injection that modulates attention scores using association score and evidence novelty; and (3) a ranking-optimised loss (LambdaRank) in place of binary cross-entropy.

---

## Encoder: Heterogeneous Graph Transformer (HGT)

| Parameter | Value |
| --- | --- |
| Number of layers | 2 |
| Hidden dimension | 32 |
| Output dimension | 32 |
| Attention heads | 1 |
| Dropout (per layer) | 0.2 |
| Node types | `target`, `disease` |
| Neighbourhood sampling (layer 1, layer 2) | 30, 20 neighbours |
| Temporal sampling strategy | `last` (most recent edges) |

**Input projection.** Each node type has a separate linear projection from its raw feature dimension into the shared hidden dimension (32).

**Attention mechanism.** Each HGT layer computes query, key and value projections per node type via heterogeneous linear layers. The attention weight for an edge is:

$$\alpha_{ij} = \text{softmax}\!\left(\frac{Q_i \cdot K_j}{\sqrt{d}} \cdot p_{\text{rel}} \cdot f_{\text{edge}}\right)$$

where $p_{\text{rel}}$ is a learnable per-relation-type scalar and $f_{\text{edge}}$ is the projected edge attribute scalar (see below).

**Skip connection.** A learnable scalar $\alpha_{\text{skip}} = \sigma(\theta)$ blends the transformed output with the input residual:

$$h' = \alpha_{\text{skip}} \cdot \text{out} + (1 - \alpha_{\text{skip}}) \cdot h$$

**Layer normalisation** is applied per node type after the skip connection, followed by dropout.

---

## Edge Feature Injection

| Parameter | Value |
| --- | --- |
| Enabled | Yes |
| Edge feature dimension | 2 |
| Features used | Association score (col 0), evidence novelty (col 1) |
| Projection | Per-edge-type linear: `[edge_feat_dim, heads]` → scalar per head |

Edge attributes are projected to a per-head scalar and multiplied into the attention logits before softmax. This allows the model to up- or down-weight neighbours based on the quality and novelty of supporting evidence, rather than treating all edges uniformly.

---

## Relational Time Encoding (RTE)

RTE is **disabled** in `p3_lambdarank`. When enabled (ablation variant `p2`), a sinusoidal positional encoding is applied to edge timestamps and added to the key/value projections, giving the model explicit awareness of edge age. The contribution of RTE is assessed in the ablation study (b2 vs. b1).

---

## Decoder

A single-head MLP operates on the concatenated source and target node embeddings and outputs one unbounded ranking logit per pair:

| Layer | Input → Output | Activation |
| --- | --- | --- |
| Linear 1 | 64 → 32 | ReLU + Dropout(0.1) |
| Linear 2 | 32 → 16 | ReLU + Dropout(0.1) |
| Linear 3 | 16 → 1 | — |

The output is a scalar ranking score used directly by the LambdaRank loss; no sigmoid is applied.

---

## Loss Function: LambdaRank

LambdaRank (Burges, 2010) directly optimises a surrogate for NDCG by weighting pairwise logistic losses by the absolute NDCG change from swapping each pair:

$$\mathcal{L} = \sum_{(i,j):\, y_i > y_j} |\Delta\text{NDCG}_{ij}| \cdot \log\!\left(1 + e^{-\sigma(s_i - s_j)}\right)$$

| Parameter | Value |
| --- | --- |
| Sigma (logistic slope) | 1.0 |
| NDCG truncation k | 100 |
| ΔNDCG gain formula | $2^{y_i} - 1$ |
| ΔNDCG discount formula | $1 / \log_2(\text{rank} + 1)$ |
| Numerical floor (IDCG) | 1e-10 |
| Normalisation | Divided by number of pairs with ΔNDCG > 0 |

---

## Optimiser and Training Schedule

| Parameter | Value |
| --- | --- |
| Optimiser | AdamW |
| Learning rate | 2.880 × 10⁻⁴ |
| Weight decay | 2.141 × 10⁻⁵ |
| LR scheduler | CosineAnnealingLR |
| T_max | 50 (= num_epochs) |
| η_min | 1 × 10⁻⁶ |
| Epochs | 50 |
| Batch size | 512 |
| Gradient clipping | max_norm = 1.0 |
| Early stopping patience | 10 epochs |
| Early stopping metric | NDCG@100 |

---

## Regularisation Summary

| Mechanism | Value |
| --- | --- |
| Dropout (HGT layers) | 0.2 |
| Dropout (decoder MLP) | 0.1 |
| L2 weight decay | 2.141 × 10⁻⁵ |
| Gradient clipping | max_norm = 1.0 |

---

## Hyperparameter Search Space

Learning rate and weight decay were found via Optuna; other hyperparameters were set by grid search over the following ranges:

| Parameter | Search space |
| --- | --- |
| Hidden dimension | {32, 64, 128} |
| Attention heads | {1, 2, 4} |
| Number of layers | {1, 2} |
| Dropout | 0.0 – 0.3 (step 0.05) |
| Learning rate | 5 × 10⁻⁵ – 1 × 10⁻³ (log scale) |
| Weight decay | 1 × 10⁻⁶ – 1 × 10⁻³ (log scale) |
| Batch size | {256, 512} |

The selected configuration (hidden=32, heads=1, layers=2) is the best on NDCG@100 on the validation split.

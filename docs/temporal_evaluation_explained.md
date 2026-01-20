# Temporal Graph Evaluation for Recommendation

## Current Setup

You have **temporal snapshots** (2018-2022) where each snapshot contains:
- **Cumulative edges**: All edges from years ≤ snapshot year
- **Supervision edges**: `disease → clinical_trial::chembl → target`

---

## Evaluation Strategy

### 1. Temporal Split

```
Timeline: ────────────────────────────────────────>
          2018   2019   2020   2021   2022
          [----Train----][Val][--Test--]
```

**Example with 2020 cutoff**:
- **Train**: All edges from years ≤ 2020 (snapshot 2020)
- **Val**: Edges from 2021 (new edges in 2021 snapshot)
- **Test**: Edges from 2022 (new edges in 2022 snapshot)

---

## 2. Training Phase

**Input**: 2020 snapshot
```python
hetero_data = load_snapshot("temporal_graph.pt", year=2020)

# Train on all edges in 2020 snapshot
train_edges = hetero_data['disease', 'clinical_trial::chembl', 'target'].edge_index
```

**Model learns**: 
- Disease-target associations known up to 2020
- Graph structure from all edge types (context + supervision)

---

## 3. Validation Phase

**Goal**: Predict **new** disease-target pairs in 2021

**Input**:
```python
# Load 2021 snapshot
data_2021 = load_snapshot("temporal_graph.pt", year=2021)

# New edges = edges in 2021 NOT in 2020
val_edges = [edges in 2021] - [edges in 2020]
```

**Evaluation**:
- For each disease in val_edges
- Rank ALL targets
- Exclude edges already known in train (2020)
- Compute metrics on actual val edges

---

## 4. Test Phase (Recommendation)

**Goal**: For each disease, recommend top-k targets (exhaustive ranking)

### Step-by-Step Process

#### 4.1 Load Test Snapshot
```python
# Load 2022 snapshot for graph structure
data_2022 = load_snapshot("temporal_graph.pt", year=2022)

# Ground truth: new edges in 2022
test_edges_2022 = [edges in 2022] - [edges in 2021]
```

#### 4.2 Collect Embeddings

**Disease embeddings**:
```python
disease_loader = NeighborLoader(
    data=data_2021,  # Use 2021 graph (no leakage!)
    input_nodes='disease',
    num_neighbors=[20, 10],
    batch_size=256,
)

disease_embs = []
for batch in disease_loader:
    emb = model.encoder(batch.x_dict, batch.edge_index_dict)['disease']
    disease_embs.append(emb[:batch['disease'].batch_size])
disease_emb = torch.cat(disease_embs, dim=0)  # [num_diseases, hidden_dim]
```

**Target embeddings**:
```python
target_loader = NeighborLoader(
    data=data_2021,  # Use 2021 graph (no leakage!)
    input_nodes='target',
    num_neighbors=[20, 10],
    batch_size=256,
)

target_embs = []
for batch in target_loader:
    emb = model.encoder(batch.x_dict, batch.edge_index_dict)['target']
    target_embs.append(emb[:batch['target'].batch_size])
target_emb = torch.cat(target_embs, dim=0)  # [num_targets, hidden_dim]
```

#### 4.3 Exhaustive Ranking with Filtering

```python
from torch_geometric import EdgeIndex
from torch_geometric.nn import MIPSKNNIndex

# Create k-NN index for efficient search
mips = MIPSKNNIndex(target_emb)

# Edges to EXCLUDE (all known edges up to 2021)
exclude_edges = data_2021['disease', 'clinical_trial::chembl', 'target'].edge_index
exclude_links = EdgeIndex(
    exclude_edges.to(device),
    sparse_size=(num_diseases, num_targets),
).sort_by('row')[0]

# Ground truth (new edges in 2022)
test_edge_index = test_edges_2022
test_edge_label_index = EdgeIndex(
    test_edge_index.to(device),
    sparse_size=(num_diseases, num_targets),
).sort_by('row')[0]

# Initialize metrics
k_values = [10, 20, 50, 100]
metrics = {}

for k in k_values:
    map_metric = LinkPredMAP(k=k).to(device)
    precision_metric = LinkPredPrecision(k=k).to(device)
    recall_metric = LinkPredRecall(k=k).to(device)
    
    # For each disease
    num_processed = 0
    for i in range(num_diseases):
        disease_emb_i = disease_emb[i:i+1]  # [1, hidden_dim]
        
        # Get exclude links for this disease
        exclude_i = exclude_links.sparse_narrow(dim=0, start=i, length=1)
        
        # Get ground truth for this disease
        truth_i = test_edge_label_index.sparse_narrow(dim=0, start=i, length=1)
        
        # k-NN search: rank ALL targets, excluding known
        _, pred_index_mat = mips.search(disease_emb_i, k, exclude_i)
        # pred_index_mat: [1, k] - top-k predicted target indices
        
        # Update metrics
        map_metric.update(pred_index_mat, truth_i)
        precision_metric.update(pred_index_mat, truth_i)
        recall_metric.update(pred_index_mat, truth_i)
    
    metrics[f'MAP@{k}'] = map_metric.compute().item()
    metrics[f'Precision@{k}'] = precision_metric.compute().item()
    metrics[f'Recall@{k}'] = recall_metric.compute().item()
```

---

## 5. What Each Metric Means

**For a disease with 5 new targets in 2022**:

**Precision@10**: Of the top-10 recommendations, how many are correct?
- Model recommends 10 targets
- 3 of them are in ground truth
- Precision@10 = 3/10 = 0.30

**Recall@10**: Of all correct targets, how many were in top-10?
- Ground truth has 5 targets
- 3 of them in top-10
- Recall@10 = 3/5 = 0.60

**MAP@10**: Average precision across all ranks
- Measures quality of ranking
- Higher if correct items ranked higher

---

## 6. Complete Evaluation Flow

```python
def evaluate_recommendation(model, train_year, test_year):
    # 1. Load graphs
    train_data = load_snapshot("temporal_graph.pt", year=train_year)
    test_data = load_snapshot("temporal_graph.pt", year=test_year)
    
    # 2. Identify test edges (new in test_year)
    test_edges = get_new_edges(test_data, train_data)
    
    # 3. Collect embeddings (using train_data graph)
    disease_emb = collect_embeddings(model, train_data, 'disease')
    target_emb = collect_embeddings(model, train_data, 'target')
    
    # 4. Setup k-NN and exclusions
    mips = MIPSKNNIndex(target_emb)
    exclude = get_known_edges(train_data)
    
    # 5. For each disease, rank all targets
    for disease_id in range(num_diseases):
        top_k_targets = mips.search(
            disease_emb[disease_id],
            k=100,
            exclude=exclude[disease_id]
        )
        
        # 6. Compare with ground truth
        ground_truth = test_edges[disease_id]
        
        # 7. Compute metrics
        update_metrics(top_k_targets, ground_truth)
    
    return compute_final_metrics()
```

---

## Summary

✅ **Train**: Learn from 2020 snapshot
✅ **Val**: Rank targets for diseases with new edges in 2021
✅ **Test**: Exhaustive ranking (each disease → all targets, excluding known)
✅ **Metrics**: MAP@k, Precision@k, Recall@k at multiple k values
✅ **No leakage**: Use 2021 graph structure for 2022 predictions

This is exactly how the PyG `recommender_system.py` example works!

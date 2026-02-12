#!/usr/bin/env python3
"""
Novel Target Prioritization Evaluator.

Ranks potential targets for diseases based on 'op' (approval) probability.
Filters out known training edges (history) to focus on novel target discovery.
"""

import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

from torch_geometric.data import HeteroData
# Handle potential module import issues if run from different cwd
try:
    from src.models.multitask_mlp import MultiTaskClinicalMLP
except ImportError:
    # Try local import if src is not in path
    sys.path.append(".")
    from src.models.multitask_mlp import MultiTaskClinicalMLP

from src.models.utils import build_model


def extract_labels_from_graph(graph, split_year, node_mappings):
    """
    Dynamically extract labels from graph edges up to a specific year.
    (Duplicated from train_clinical_multitask.py for self-containment)
    """
    print(f"   Extracting labels up to year {split_year}...")
    
    # Task to edge type mapping
    task_edge_map = {
        'pos': 'clinical_trial_positive::chembl',
        'unmet': 'clinical_trial_unmet_efficacy::chembl',
        'adv': 'clinical_trial_adverse_effects::chembl',
        'op': 'clinical_trial_Unknown/Operational::chembl'
    }
    
    label_data = {}
    
    for task, edge_type_name in task_edge_map.items():
        etype = ('disease', edge_type_name, 'target')
        
        if etype not in graph.edge_types:
            continue
            
        edge_store = graph[etype]
        edge_index = edge_store.edge_index
        edge_attr = edge_store.edge_attr
        edge_time = edge_store.edge_time if hasattr(edge_store, 'edge_time') else None
        
        if edge_time is None:
             mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        else:
             mask = edge_time <= int(split_year)
        
        filtered_indices = edge_index[:, mask]
        filtered_attr = edge_attr[mask] if edge_attr is not None else torch.ones(mask.sum(), 1)
        
        num_edges = filtered_indices.size(1)
        
        src_indices = filtered_indices[0].cpu().numpy()
        dst_indices = filtered_indices[1].cpu().numpy()
        scores = filtered_attr.squeeze().cpu().numpy()
        if scores.ndim == 0: scores = np.array([scores])
        
        for i in range(num_edges):
            d_idx = int(src_indices[i])
            t_idx = int(dst_indices[i])
            score = float(scores[i])
            
            key = (d_idx, t_idx)
            if key not in label_data:
                label_data[key] = {t: 0.0 for t in task_edge_map.keys()}
            label_data[key][task] = max(label_data[key][task], score)

    # Return as set of (d_idx, t_idx) tuples for easy filtering/lookup
    # We only care about existence for history filtering
    pairs = set(label_data.keys())
    
    # For ground truth (test), we specifically want pairs where 'op' > 0 (or some threshold)
    # The user request implies ranking based on predicted 'op'.
    # Ground truth is typically "did it actually succeed?" (op > 0)
    # Or "did it exist in test set?" 
    # Let's return the full data dict so we can filter later.
    return label_data


def evaluate_ranking(
    model, 
    embeddings, 
    train_pairs, 
    test_pairs, 
    num_disease_nodes, 
    num_target_nodes, 
    device, 
    k_values=[100, 200, 500]
):
    """
    Evaluate ranking metrics per disease.
    """
    model.eval()
    
    # Pre-compute metrics storage
    # Adding Precision and Recall
    metrics = {k: {'precision': [], 'recall': [], 'hits': [], 'mrr': [], 'ndcg': []} for k in k_values}
    
    # Identify test diseases (diseases that have at least one test pair)
    test_diseases = set(d for d, t in test_pairs.keys())
    
    print(f"\\n🔍 Evaluating Ranking on {len(test_diseases)} diseases...")
    print(f"   K values: {k_values}")
    
    # Pre-organize ground truth: disease -> set(target_indices)
    test_ground_truth = {}
    for (d, t), scores in test_pairs.items():
        if d not in test_ground_truth: test_ground_truth[d] = set()
        test_ground_truth[d].add(t)
        
    # Pre-organize history: disease -> set(target_indices)
    history_map = {}
    for (d, t) in train_pairs.keys():
        if d not in history_map: history_map[d] = set()
        history_map[d].add(t)
        
    # All target indices (candidates)
    all_targets = torch.arange(num_target_nodes, device=device)
    target_emb_all = embeddings['target'].to(device) # [N_t, dim]
    
    # Loop over diseases
    for d_idx in tqdm(test_diseases):
        true_targets = test_ground_truth[d_idx]
        history = history_map.get(d_idx, set())
        
        # Prepare inputs
        d_emb = embeddings['disease'][d_idx].unsqueeze(0).to(device) # [1, dim]
        
        # Expand disease emb to match all targets
        d_emb_expanded = d_emb.expand(num_target_nodes, -1)
        
        # Forward pass
        with torch.no_grad():
            logits = model(d_emb_expanded, target_emb_all) # returns dict
            # Use 'op' score for ranking
            scores = torch.sigmoid(logits['op']) # [N_t]
            
        # Mask history
        if history:
            hist_indices = torch.tensor(list(history), device=device, dtype=torch.long)
            scores[hist_indices] = -1.0 # Set to lowest possible
            
        # Ranking
        max_k = max(k_values)
        top_k_scores, top_k_indices = torch.topk(scores, max_k)
        top_k_indices = top_k_indices.cpu().tolist()
        
        # Metrics
        for k in k_values:
            curr_top = top_k_indices[:k]
            intersects = len(set(curr_top) & true_targets)
            
            # Recall@K (Sensitivity / Hit Rate)
            if len(true_targets) > 0:
                recall = intersects / len(true_targets)
            else:
                recall = 0.0
            metrics[k]['recall'].append(recall)
            
            # Precision@K
            precision = intersects / k
            metrics[k]['precision'].append(precision)
            
            # Hits@K (Binary check - did we find at least one?)
            metrics[k]['hits'].append(1.0 if intersects > 0 else 0.0)
            
            # MRR (Mean Reciprocal Rank)
            rr = 0.0
            for rank, t_idx in enumerate(curr_top):
                if t_idx in true_targets:
                    rr = 1.0 / (rank + 1)
                    break
            metrics[k]['mrr'].append(rr)
            
            # NDCG
            dcg = 0.0
            idcg = 0.0
            
            # DCG
            for i, t_idx in enumerate(curr_top):
                if t_idx in true_targets:
                    dcg += 1.0 / np.log2(i + 2)
            
            # IDCG (Perfect ranking)
            num_relevant = min(k, len(true_targets))
            for i in range(num_relevant):
                idcg += 1.0 / np.log2(i + 2)
                
            metrics[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0)

    # Average metrics
    final_results = {}
    print(f"\\n📊 Ranking Results (Prioritization by 'op' score):")
    for k in k_values:
        avg_rec = np.mean(metrics[k]['recall'])
        avg_prec = np.mean(metrics[k]['precision'])
        avg_mrr = np.mean(metrics[k]['mrr'])
        avg_ndcg = np.mean(metrics[k]['ndcg'])
        
        final_results[f'Recall@{k}'] = float(avg_rec)
        final_results[f'Precision@{k}'] = float(avg_prec)
        final_results[f'MRR@{k}'] = float(avg_mrr)
        final_results[f'NDCG@{k}'] = float(avg_ndcg)
        
        print(f"   K={k:<3}: Recall={avg_rec:.4f} | Precision={avg_prec:.4f} | MRR={avg_mrr:.4f} | NDCG={avg_ndcg:.4f}")
        
    return final_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment config (yaml)")
    parser.add_argument("--checkpoint", help="Path to checkpoint (default: output_dir/best_decoder.pt)")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    output_dir = Path(cfg.train.output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 Novel Target Prioritization Evaluator")
    print(f"   Config: {args.config}")
    print(f"   Device: {device}")
    
    # 1. Load Data
    print(f"\\n📂 Loading graph data...")
    graph = torch.load(cfg.data.graph_file, weights_only=False)
    mappings = torch.load(cfg.data.mappings_file, weights_only=False)
    node_mappings = mappings['node_mapping']
    
    # Embeddings
    print(f"🔧 Extracting features...")
    embeddings = {}
    if cfg.model.get('use_encoder', False):
        print("⚠️  Encoder support not fully implemented in this standalone script yet. Using raw features.")
    
    for nt in ['disease', 'target']:
        if graph[nt].x is not None:
             embeddings[nt] = graph[nt].x.float()
        else:
             raise ValueError(f"Node type {nt} has no features!")

    # 2. Extract Edges
    # Define "History" as everything up to end of VALIDATION or TRAIN?
    # User request: "exclusively test edge as novel disease targets"
    # To be "novel" in Test, it must NOT be in Train OR Val.
    # Check config for Val year.
    
    ts = cfg.data.temporal_split
    
    # Determine Cutoff Years
    # Train is strictly training.
    # Val is usually known history during model selection.
    # So History should include Val.
    
    if hasattr(ts, 'val') and ts.val is not None:
        history_year = ts.val[1] # End of validation (e.g., 2020)
        print(f"   Using Validation End Year ({history_year}) as History Cutoff.")
    else:
        history_year = ts.train[1]
        print(f"   Using Training End Year ({history_year}) as History Cutoff (No val split found).")
        
    test_year = ts.test[1]
    
    print(f"\\n📊 Extracting History (All edges <= {history_year})...")
    history_data = extract_labels_from_graph(graph, history_year, node_mappings)
    
    print(f"📊 Extracting Test Candidates (Edges <= {test_year})...")
    full_test_data = extract_labels_from_graph(graph, test_year, node_mappings)
    
    # Novel Test = (Edges <= Test) - (Edges <= History)
    test_novel_data = {
        k: v for k, v in full_test_data.items() 
        if k not in history_data
    }
    
    print(f"   Total Edges in Test Period Window: {len(full_test_data):,}")
    print(f"   Known History Edges: {len(history_data):,}")
    print(f"   Strictly Novel Test Edges: {len(test_novel_data):,} (History Removed)")
    
    if len(test_novel_data) == 0:
        print("❌ No novel edges found in test split! Check temporal splits.")
        sys.exit(1)

    # 3. Load Model
    print(f"\\n🧠 Loading model...")
    checkpoint_path = args.checkpoint if args.checkpoint else output_dir / "best_decoder.pt"
    
    # Need input dimensions
    disease_dim = embeddings['disease'].size(1)
    target_dim = embeddings['target'].size(1)
    input_dim = disease_dim + target_dim
    
    model = MultiTaskClinicalMLP(
        input_dim=input_dim, 
        hidden_dim=cfg.model.decoder.hidden_dim, 
        dropout=cfg.model.decoder.dropout
    ).to(device)
    
    print(f"   Loading weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    
    # 4. Evaluate
    results = evaluate_ranking(
        model, 
        embeddings, 
        history_data, # Use full history (Train+Val) to mask
        test_novel_data, 
        graph['disease'].num_nodes,
        graph['target'].num_nodes,
        device,
        k_values=[100, 200, 500] # Updated as per user request
    )
    
    # 5. Save
    out_file = output_dir / "results_ranking.yaml"
    with open(out_file, 'w') as f:
        yaml.dump(results, f)
    print(f"\\n✅ Saved ranking results to {out_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Novel Target Prioritization Evaluator using OpenTargets Association Scores.

Uses pre-computed association scores from OpenTargets parquet files instead of 
trained model predictions for baseline evaluation.
"""

import sys
import torch
import pandas as pd
import numpy as np
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from glob import glob

from torch_geometric.data import HeteroData


def load_opentargets_associations_timestamped(parquet_dir, node_mappings, max_year):
    """
    Load all parquet files iteratively and create disease-target score mapping.
    Optimized for memory usage by processing one file at a time.
    
    Args:
        parquet_dir: Directory containing OpenTargets association parquet files
        node_mappings: Dict with 'disease' and 'target' ID to index mappings
        max_year: load associations up to this year and take max score
        
    Returns:
        dict: {(disease_idx, target_idx): score}
    """
    print(f"\n📂 Loading OpenTargets associations from {parquet_dir}...")
    if max_year is not None:
        print(f"   Filtering to associations <= year {max_year} (cumulative max score)")
    
    parquet_files = glob(str(Path(parquet_dir) / "*.parquet"))
    print(f"   Found {len(parquet_files)} parquet files")
    
    if len(parquet_files) == 0:
        raise ValueError(f"No parquet files found in {parquet_dir}")
    
    # Map IDs to graph indices using sets for faster lookup
    disease_mapping = node_mappings['disease']
    target_mapping = node_mappings['target']
    
    valid_diseases = set(disease_mapping.keys())
    valid_targets = set(target_mapping.keys())
    
    association_scores = {}
    total_processed = 0
    total_mapped = 0
    
    # Process files iteratively
    for pf in tqdm(parquet_files, desc="Processing parquet files"):
        try:
            # Read only essential columns (check if 'year' exists first)
            import pyarrow.parquet as pq
            schema_cols = pq.read_schema(pf).names
            columns = ['diseaseId', 'targetId', 'score']
            has_year = 'year' in schema_cols
            if max_year is not None and has_year:
                columns.append('year')
                
            df = pd.read_parquet(pf, columns=columns)
            
            if df.empty: continue
            
            # Filter to cumulative (<= max_year) to get best historical score
            if max_year is not None and has_year and 'year' in df.columns:
                df = df[df['year'] <= max_year]
                    
            if df.empty: continue
            
            # Pre-filter rows where both IDs are in our mapping
            # This drastically reduces rows before the slow iterrows loop
            mask = df['diseaseId'].isin(valid_diseases) & df['targetId'].isin(valid_targets)
            df_filtered = df[mask].copy()
            
            if df_filtered.empty: continue
            
            # Handle NaN scores
            df_filtered['score'] = df_filtered['score'].fillna(0.0)
            
            total_processed += len(df)
            total_mapped += len(df_filtered)
            
            # Update dictionary
            # Using vectorization via mapping then manual update is faster but dict update is easier
            # Let's do a semi-vectorized approach for speed
            
            # Map IDs to indices
            # We can use map() but let's be careful with missing keys (already filtered)
            d_indices = df_filtered['diseaseId'].map(disease_mapping)
            t_indices = df_filtered['targetId'].map(target_mapping)
            scores = df_filtered['score'].values
            
            # Update dict
            for d_idx, t_idx, score in zip(d_indices, t_indices, scores):
                key = (d_idx, t_idx)
                if key in association_scores:
                    association_scores[key] = max(association_scores[key], float(score))
                else:
                    association_scores[key] = float(score)
            
            # Explicit garbage collection hint
            del df, df_filtered
            
        except Exception as e:
            print(f"⚠️ Error processing {pf}: {e}")
            continue

    print(f"   Total associations processed: {total_processed:,}")
    print(f"   Mapped associations: {len(association_scores):,}")
    
    return association_scores


def evaluate_ranking_with_scores(
    association_scores,
    train_pairs, 
    test_pairs, 
    num_disease_nodes, 
    num_target_nodes, 
    k_values=[100, 200, 500],
    node_mappings=None,
    random_seed=42
):
    """
    Evaluate ranking metrics per disease using pre-computed association scores.
    
    For each disease, creates a candidate set of novel targets (excluding train+val history),
    scores only those candidates using association scores, and computes ranking metrics.
    """
    # Pre-compute metrics storage
    metrics = {k: {'precision': [], 'recall': [], 'hits': [], 'mrr': [], 'ndcg': []} for k in k_values}
    ranking_dfs = {}
    
    # Create reverse mappings for debug info if mappings are provided
    idx_to_disease = {}
    idx_to_target = {}
    if node_mappings:
        idx_to_disease = {v: k for k, v in node_mappings['disease'].items()}
        idx_to_target = {v: k for k, v in node_mappings['target'].items()}
    
    # Identify test diseases (diseases that have at least one test pair)
    # Convert to sorted list to ensure reproducible metric indexing
    test_diseases = sorted(list(set(d for d, t in test_pairs.keys())))
    
    print(f"\n🔍 Evaluating Ranking on {len(test_diseases)} diseases...")
    print(f"   K values: {k_values}")
    
    # Pre-organize ground truth: disease -> {target_idx: max_phase_score}
    # test_pairs now contains max_scores (float values representing clinical phase)
    # Store scores for graded relevance NDCG (User Request: use ground truth scoring in ranking metrics)
    test_ground_truth = {}
    test_ground_truth_scores = {}
    for (d, t), max_score in test_pairs.items():
        if max_score > 0:  # Only consider pairs with actual clinical trial activity
            if d not in test_ground_truth: 
                test_ground_truth[d] = set()
                test_ground_truth_scores[d] = {}
            test_ground_truth[d].add(t)
            test_ground_truth_scores[d][t] = max_score
        
    # Pre-organize history: disease -> set(target_indices)
    history_map = {}
    for (d, t) in train_pairs.keys():
        if d not in history_map: history_map[d] = set()
        history_map[d].add(t)
    
    all_target_indices = set(range(num_target_nodes))
    
    # Loop over diseases
    for d_idx in tqdm(test_diseases):
        # Ensure true_targets are unique per disease (User Request)
        # Using a set handles duplicates automatically
        true_targets = set(test_ground_truth.get(d_idx, []))
        
        if len(true_targets) == 0:
            continue  # Skip diseases with no positive test pairs
            
        history = history_map.get(d_idx, set())
        
        # Build candidate set: all targets EXCEPT those in train+val history
        # Using sets ensures usage of unique targets
        candidate_targets = all_target_indices - history
        candidate_list = sorted(list(candidate_targets))
        
        # Count non-zero stats for logging
        non_zero_truth = sum(1 for t in true_targets if association_scores.get((d_idx, t), 0.0) > 0)
        non_zero_candidates_approx = sum(1 for t in candidate_list if association_scores.get((d_idx, t), 0.0) > 0)        

        # Skip if no candidates or too few true targets
        if len(candidate_list) == 0 or len(true_targets) <= 3 or non_zero_truth == 0:
            continue

        print(f"   Disease {d_idx}: Truth={len(true_targets)} (NonZero={non_zero_truth}), History={len(history)}, Candidates={len(candidate_list)} (NonZero={non_zero_candidates_approx})")

        
        # Get scores for candidates using association scores (default 0.0 if not found)
        # GATher paper approach: unassociated targets (score=0) receive a random score
        # drawn from Uniform(0, min_disease_score) to generate genome-wide ranking.
        # This breaks the all-zero tie while keeping unassociated targets below associated ones.
        rng = np.random.default_rng(random_seed + d_idx)  # per-disease seed for reproducibility
        
        candidate_scores = []
        for t_idx in candidate_list:
            score = association_scores.get((d_idx, t_idx), 0.0)
            candidate_scores.append(score)
        
        candidate_scores = np.array(candidate_scores, dtype=np.float64)
        
        # Compute per-disease min non-zero association score
        nonzero_mask = candidate_scores > 0
        if nonzero_mask.any():
            min_disease_score = candidate_scores[nonzero_mask].min()
            # Assign random uniform scores in [0, min_disease_score) for unassociated targets
            n_zero = (~nonzero_mask).sum()
            candidate_scores[~nonzero_mask] = rng.uniform(0.0, min_disease_score, size=n_zero)
        # If all candidates are zero (no association data), scores stay 0

        sorted_indices = np.argsort(-candidate_scores)  # Descending order
        
        # Get top-k
        max_k = min(max(k_values), len(candidate_list))
        top_k_local_indices = sorted_indices[:max_k]
        top_k_indices = [candidate_list[i] for i in top_k_local_indices]
        
        # Store debugging info (User Request: dict{disease: df})
        # Format similar to MetronAtK: user, item, score, golden, rank
        
        # Merge candidates and true targets into one list for ranking
        # Note: 'candidate_list' already contains non-zero candidates
        # 'true_targets' contains ground truth
        
        # We need a unified list of all items to rank
        # Candidates are already filtered by non-zero (if enabled)
        # We should ensure true targets are included even if they have zero score (to reflect poor ranking)
        # BUT, the current logic calculates metrics based on 'candidate_scores' which might filter out zero-score items
        
        # Let's rebuild the full ranking list for this disease
        # 1. All candidates (filtered or not)
        # 2. All true targets
        
        # Re-fetch all potential candidates for the dataframe
        # (We need to be consistent with what was used for metrics, but the user wants a full view)
        # The user's snippet implies a full list of test items + negative items
        
        # Use the 'sorted_indices' and 'candidate_list' from the metric calculation
        # enabling direct inspection of the ranking used for metrics.
        
        debug_data = []
        for idx in sorted_indices:
             t_idx = candidate_list[idx]
             score = candidate_scores[idx]
             is_golden = t_idx in true_targets
             
             debug_data.append({
                 'user': d_idx,
                 'item': t_idx,
                 'score': score,
                 'golden': is_golden,
                 'disease_id': idx_to_disease.get(d_idx, f"D_{d_idx}"),
                 'target_id': idx_to_target.get(t_idx, f"T_{t_idx}")
             })
             
        # Create DF
        df_disease = pd.DataFrame(debug_data)
        
        # Add Rank (Dense rank descending)
        if not df_disease.empty:
            df_disease['rank'] = df_disease['score'].rank(method='min', ascending=False)
            df_disease.sort_values('rank', inplace=True)
            
        ranking_dfs[d_idx] = df_disease

        # Metrics
        for k in k_values:
            k_actual = min(k, len(top_k_indices))
            curr_top = top_k_indices[:k_actual]
            intersects = len(set(curr_top) & true_targets)
            
            # Recall@K
            if len(true_targets) > 0:
                recall = intersects / len(true_targets)
            else:
                recall = 0.0
            metrics[k]['recall'].append(recall)
            
            # Precision@K
            precision = intersects / k_actual if k_actual > 0 else 0.0
            metrics[k]['precision'].append(precision)
            
            # Hits@K
            metrics[k]['hits'].append(1.0 if intersects > 0 else 0.0)
            
            # MRR
            rr = 0.0
            for rank, t_idx in enumerate(curr_top):
                if t_idx in true_targets:
                    rr = 1.0 / (rank + 1)
                    break
            metrics[k]['mrr'].append(rr)
            
            # NDCG
            dcg = 0.0
            idcg = 0.0
            
            # DCG - Use graded relevance (max_phase score) instead of binary 1.0
            for i, t_idx in enumerate(curr_top):
                if t_idx in true_targets:
                    relevance_score = test_ground_truth_scores[d_idx].get(t_idx, 1.0)
                    dcg += relevance_score / np.log2(i + 2)
            
            # IDCG (Perfect ranking) - Sort true targets by their max_phase scores for ideal order
            true_target_scores = [test_ground_truth_scores[d_idx].get(t, 1.0) for t in true_targets]
            sorted_relevances = sorted(true_target_scores, reverse=True)
            num_relevant = min(k_actual, len(true_targets))
            for i in range(num_relevant):
                idcg += sorted_relevances[i] / np.log2(i + 2)
                
            metrics[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0)

    # Average metrics (Macro-average: average of per-disease metrics)
    final_results = {}
    print(f"\n📊 Ranking Results (OpenTargets Association Scores):")
    
    # Track per-disease performance for grouping
    # User Request: Rank by Precision@200
    k_main = 200
    if k_main not in metrics and k_values:
        k_main = k_values[0]
        
    per_disease_perf = []
    if k_main in metrics:
        for i, d_idx in enumerate(test_diseases):
            if i < len(metrics[k_main]['precision']):
                per_disease_perf.append({
                    'index': i,
                    'd_idx': d_idx,
                    'd_id': idx_to_disease.get(d_idx, f"D_{d_idx}"),
                    'precision': metrics[k_main]['precision'][i],
                    'ndcg': metrics[k_main]['ndcg'][i],
                    'recall': metrics[k_main]['recall'][i]
                })
    
    # Sort by performance (Precision descending, then NDCG as tie-breaker)
    per_disease_perf.sort(key=lambda x: (x['precision'], x['ndcg']), reverse=True)
    num_eval = len(per_disease_perf)
    
    if num_eval > 0:
        n_group = min(20, num_eval)
        
        # Define groups: Best, Median (Middle), Worst
        mid_start = max(0, num_eval // 2 - n_group // 2)
        mid_end = min(num_eval, mid_start + n_group)
        
        groups = [
            ("🏆 Top 20 Diseases", per_disease_perf[:n_group]),
            ("⚖️ Median 20 Diseases", per_disease_perf[mid_start:mid_end]),
            ("📉 Worst 20 Diseases", per_disease_perf[-n_group:])
        ]
        
        for name, group in groups:
            if not group: continue
            print(f"\n{name} Performance (N={len(group)}):")
            print(f"   {'K':<5} | {'Mean Prec':<10} | {'Med Prec':<10} | {'Mean NDCG':<10} | {'Med NDCG':<10}")
            print(f"   {'-'*65}")
            
            group_indices = [item['index'] for item in group]
            
            for k in k_values:
                g_prec_mean = np.mean([metrics[k]['precision'][idx] for idx in group_indices])
                g_prec_med  = np.median([metrics[k]['precision'][idx] for idx in group_indices])
                g_ndcg_mean = np.mean([metrics[k]['ndcg'][idx] for idx in group_indices])
                g_ndcg_med  = np.median([metrics[k]['ndcg'][idx] for idx in group_indices])
                print(f"   {k:<5} | {g_prec_mean:.4f}    | {g_prec_med:.4f}    | {g_ndcg_mean:.4f}    | {g_ndcg_med:.4f}")

    print(f"\n📈 Overall Dataset Mean Metrics (N={num_eval}):")
    for k in k_values:
        avg_rec = np.mean(metrics[k]['recall'])
        avg_prec = np.mean(metrics[k]['precision'])
        avg_mrr = np.mean(metrics[k]['mrr'])
        avg_ndcg = np.mean(metrics[k]['ndcg'])
        
        med_rec = np.median(metrics[k]['recall'])
        med_prec = np.median(metrics[k]['precision'])
        med_mrr = np.median(metrics[k]['mrr'])
        med_ndcg = np.median(metrics[k]['ndcg'])
        
        final_results[f'Recall@{k}'] = float(avg_rec)
        final_results[f'Precision@{k}'] = float(avg_prec)
        final_results[f'MRR@{k}'] = float(avg_mrr)
        final_results[f'NDCG@{k}'] = float(avg_ndcg)
        
        final_results[f'Median_Recall@{k}'] = float(med_rec)
        final_results[f'Median_Precision@{k}'] = float(med_prec)
        final_results[f'Median_MRR@{k}'] = float(med_mrr)
        final_results[f'Median_NDCG@{k}'] = float(med_ndcg)
        
        print(f"   K={k:<3}: Recall={avg_rec:.4f} (Med={med_rec:.4f}) | Precision={avg_prec:.4f} (Med={med_prec:.4f}) | MRR={avg_mrr:.4f} (Med={med_mrr:.4f}) | NDCG={avg_ndcg:.4f} (Med={med_ndcg:.4f})")
    
    return final_results, ranking_dfs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment config (yaml)")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    output_dir = Path(cfg.eval.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Novel Target Prioritization Evaluator (OpenTargets Baseline)")
    print(f"   Config: {args.config}")
    
    # 1. Load Mappings
    print(f"\n📂 Loading mappings from {cfg.data.mappings_file}...")
    mappings = torch.load(cfg.data.mappings_file, weights_only=False)
    node_mappings = mappings['node_mapping']

    # 2. Load Ground Truth from ChEMBL Splits (same source as _json.py)
    chembl_train_path = "data/chembl_splits/train_pairs.parquet"
    chembl_val_path   = "data/chembl_splits/val_novel_pairs.parquet"
    chembl_test_path  = "data/chembl_splits/test_novel_pairs.parquet"

    for p in [chembl_train_path, chembl_val_path, chembl_test_path]:
        if not Path(p).exists():
            print(f"❌ Required file not found: {p}")
            print(f"   Run: python create_chembl_temporal_splits.py")
            sys.exit(1)

    print(f"\n📂 Loading test pairs from ChEMBL ground truth...")
    chembl_test_df = pd.read_parquet(chembl_test_path)
    print(f"   Loaded {len(chembl_test_df)} novel test pairs")

    # Load validation diseases for filtering if provided in config
    validation_disease_indices = None
    if 'validation_diseases_file' in cfg.data and cfg.data.validation_diseases_file:
        print(f"📋 Loading validation diseases from {cfg.data.validation_diseases_file}...")
        val_diseases_df = pd.read_csv(cfg.data.validation_diseases_file)
        # Filter out diseases not in graph (graph_node_idx == -1)
        val_diseases_df = val_diseases_df[val_diseases_df['graph_node_idx'] != -1]
        validation_disease_indices = set(val_diseases_df['graph_node_idx'].tolist())
        print(f"   Loaded {len(validation_disease_indices)} validation diseases for benchmark")

    novel_test_pairs = {}
    for _, row in chembl_test_df.iterrows():
        d_idx = int(row['disease_idx'])
        t_idx = int(row['target_idx'])
        
        # Filter by validation diseases if provided
        if validation_disease_indices is not None and d_idx not in validation_disease_indices:
            continue
            
        key = (d_idx, t_idx)
        novel_test_pairs[key] = float(row['score'])

    print(f"\n📊 Filtered Test Set Statistics:")
    if validation_disease_indices is not None:
        print(f"   Filtering active: Only {len(validation_disease_indices)} diseases considered")
    
    # Identify test diseases (diseases that have at least one test pair)
    # Convert to sorted list to ensure reproducible metric indexing
    unique_test_diseases = sorted(list(set(d for d, t in novel_test_pairs.keys())))
    unique_test_targets = set(t for d, t in novel_test_pairs.keys())
    
    print(f"   Unique diseases: {len(unique_test_diseases)}")
    print(f"   Unique targets:  {len(unique_test_targets)}")
    print(f"   Total pairs:     {len(novel_test_pairs)}")

    if len(novel_test_pairs) == 0:
        print("❌ No novel edges found in test split! Check temporal splits.")
        sys.exit(1)

    # 2b. Load History (Train + Val) to exclude from candidates
    print(f"\n📂 Loading history pairs (Train + Val) to exclude from evaluation...")
    history_pairs = {}
    for path, label in [(chembl_train_path, "train"), (chembl_val_path, "val")]:
        df = pd.read_parquet(path)
        print(f"   Loaded {len(df)} {label} pairs")
        for _, row in df.iterrows():
            key = (int(row['disease_idx']), int(row['target_idx']))
            history_pairs[key] = 1.0
    print(f"   Total history pairs to exclude: {len(history_pairs)}")

    # 3. Load Association Scores
    association_dir = cfg.data.association_dir
    print(f"\nUsing association data from: {association_dir}")
    association_scores = load_opentargets_associations_timestamped(association_dir, node_mappings, max_year=2018)

    # 3b. Check test set coverage
    print("\n🔍 Checking Test Set Coverage in Association Data:")
    test_pair_count = len(novel_test_pairs)
    test_with_score_count = sum(1 for k in novel_test_pairs if k in association_scores)
    print(f"   Test Pairs with Non-Zero Association Score: {test_with_score_count}/{test_pair_count} ({test_with_score_count/test_pair_count*100:.1f}%)")

    # 4. Evaluate
    num_disease_nodes = len(node_mappings['disease'])
    num_target_nodes  = len(node_mappings['target'])

    results, ranking_dfs = evaluate_ranking_with_scores(
        association_scores,
        history_pairs,
        novel_test_pairs,
        num_disease_nodes,
        num_target_nodes,
        k_values=[100, 200, 500],
        node_mappings=node_mappings
    )

    # 5. Save debug rankings
    debug_file = Path("output/graph/debug_rankings.pkl")
    debug_file.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(debug_file, 'wb') as f:
        pickle.dump(ranking_dfs, f)
    print(f"✅ Saved debug ranking DataFrames to {debug_file}")

    # 6. Save metric results
    cfg_name = Path(args.config).stem
    out_file = output_dir / f"results_ranking_opentargets_baseline_{cfg_name}.yaml"
    with open(out_file, 'w') as f:
        yaml.dump(results, f)
    print(f"\n✅ Saved ranking results to {out_file}")


if __name__ == "__main__":
    main()

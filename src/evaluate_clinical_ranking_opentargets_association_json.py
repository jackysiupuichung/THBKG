#!/usr/bin/env python3
"""
Evaluate clinical trial ranking using OpenTargets JSON association snapshots.
This baseline uses pre-computed association scores from a specific OpenTargets database release (e.g., 2017).
Unlike the time-series parquet baseline, this does NOT filter by year because the input file itself 
is assumed to be a temporal snapshot (grounded by the database release version).

Usage:
    python src/evaluate_clinical_ranking_opentargets_association_json.py --config config/experiments/clinical_ranking_eval.yaml
"""

import sys
import glob
import json
import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.evaluate_clinical_ranking_opentargets_association import (
    extract_labels_from_graph, 
    evaluate_ranking_with_scores
)

def load_opentargets_associations_json(json_path, node_mappings):
    """
    Load association scores from OpenTargets JSON/JSONL files (database snapshot).
    
    Args:
        json_path: Path to a JSON file or directory containing JSON files
        node_mappings: Dict with 'disease' and 'target' ID to index mappings
        
    Returns:
        dict: {(disease_idx, target_idx): score}
    """
    print(f"\n📂 Loading OpenTargets associations from {json_path}...")
    
    association_scores = {}
    
    # Handle direct file or directory
    path_obj = Path(json_path)
    if path_obj.is_dir():
        files = glob.glob(str(path_obj / "*.json")) + glob.glob(str(path_obj / "*.jsonl"))
    else:
        files = [str(path_obj)]
    
    print(f"   Found {len(files)} JSON files")
    
    # Graph mappings
    disease_to_idx = node_mappings['disease']
    target_to_idx = node_mappings['target']
    
    # Optimized set lookup
    valid_diseases = set(disease_to_idx.keys())
    valid_targets = set(target_to_idx.keys())
    
    total_processed = 0
    mapped_count = 0
    
    for f in tqdm(files, desc="Processing JSON files"):
        with open(f, 'r') as f_in:
            for line in f_in:
                if not line.strip(): continue
                total_processed += 1
                
                try:
                    data = json.loads(line)
                    
                    # Extract IDs
                    # Note: Structure is data['target']['id'] and data['disease']['id']
                    # Some versions might have different structure, adhering to user provided sample
                    
                    target_id = data.get('target', {}).get('id')
                    disease_id = data.get('disease', {}).get('id')
                    
                    if not target_id or not disease_id:
                        continue
                        
                    # Check mapping
                    if disease_id in valid_diseases and target_id in valid_targets:
                        # Extract Overall Score
                        # "association_score": {"overall": 1.0, ...}
                        score = data.get('association_score', {}).get('overall', 0.0)
                        
                        d_idx = disease_to_idx[disease_id]
                        t_idx = target_to_idx[target_id]
                        
                        # Store max score if duplicate pairs exist (unlikely in snapshot but safe)
                        key = (d_idx, t_idx)
                        if key in association_scores:
                            association_scores[key] = max(association_scores[key], float(score))
                        else:
                            association_scores[key] = float(score)
                            
                        mapped_count += 1
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue

    print(f"   Total associations processed: {total_processed:,}")
    print(f"   Mapped associations: {mapped_count:,}")
    
    return association_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--association_json', type=str, required=False, help='Override association JSON path')
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)
    
    # 1. Load Graph & Mappings
    print(f"Loading graph from {cfg.data.graph_file}...")
    graph = torch.load(cfg.data.graph_file, weights_only=False)
    mappings = torch.load(cfg.data.mappings_file, weights_only=False)
    node_mappings = mappings['node_mapping']

    # 2. Extract Validation/Test Edges
    # We use validation set history for filtering, and predict on test set
    temporal_split = cfg.data.temporal_split
    history_year = temporal_split.val[1] # e.g. 2017
    test_year = temporal_split.test[1]   # e.g. 2024
    
    print(f"Extracting history edges (up to {history_year})...")
    # 'train_pairs' here includes train+val for history filtering
    history_pairs, _ = extract_labels_from_graph(graph, history_year, node_mappings)
    
    print(f"Extracting test edges (up to {test_year})...")
    # Identify unique test edges that are NOT in history
    # extract_labels_from_graph returns ALL edges up to test_year
    # We need to filter out the history ones to get the "Novel" test set
    full_test_pairs, full_test_max_scores = extract_labels_from_graph(graph, test_year, node_mappings)
    
    # Identify ONLY the novel edges in the test window
    novel_test_pairs = {}
    for key, max_score in full_test_pairs.items():
        if key not in history_pairs:
            novel_test_pairs[key] = full_test_max_scores[key] # Use max_score from test period

    print(f"Total test pairs (cumulative): {len(full_test_pairs)}")
    print(f"Novel test pairs (target for prediction): {len(novel_test_pairs)}")
    
    # Filter validation diseases if specified
    validation_diseases = set()
    if hasattr(cfg.data, 'validation_diseases_file') and Path(cfg.data.validation_diseases_file).exists():
        val_df = pd.read_csv(cfg.data.validation_diseases_file)
        # Filter valid mappings
        val_df = val_df[val_df['graph_node_idx'] != -1]
        validation_diseases = set(val_df['graph_node_idx'].tolist())
        print(f"Loaded {len(validation_diseases)} validation diseases to filter evaluation.")
        
        # Filter test pairs to only these diseases
        novel_test_pairs = {k: v for k, v in novel_test_pairs.items() if k[0] in validation_diseases}
        print(f"Novel test pairs after disease filtering: {len(novel_test_pairs)}")

    if len(novel_test_pairs) == 0:
        print("❌ No novel edges found in test split! Check temporal splits.")
        sys.exit(1)

    # 3. Load Association Scores from JSON
    # Use command line arg if provided, else use config path (assuming a new config field or reusing association_dir)
    json_path = args.association_json
    if not json_path:
        # Fallback to association_dir but check if it contains json
        json_path = cfg.data.association_dir
    
    print(f"Using association data from: {json_path}")
    association_scores = load_opentargets_associations_json(json_path, node_mappings)

    # 4. Evaluate
    num_disease_nodes = graph['disease'].num_nodes
    num_target_nodes = graph['target'].num_nodes
    
    results = evaluate_ranking_with_scores(
        association_scores,
        history_pairs, 
        novel_test_pairs, 
        num_disease_nodes, 
        num_target_nodes, 
        k_values=[100, 200, 500]
    )
    
    # Save results
    output_dir = Path("runs/clinical_ranking_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results_ranking_opentargets_json_baseline.yaml"
    
    with open(output_file, 'w') as f:
        OmegaConf.save(results, f)
        
    print(f"\n✅ Saved ranking results to {output_file}")

if __name__ == "__main__":
    main()

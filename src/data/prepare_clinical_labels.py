#!/usr/bin/env python3
"""
Prepare multi-task clinical trial labels per as-of year.

Extracts max phase scores by outcome type for temporal evaluation:
- y_pos: max phase for positive trials
- y_unmet: max phase for unmet efficacy trials
- y_adv: max phase for adverse effects trials
- y_op: max phase for operational/unknown trials

Phase scores: 0.05, 0.1, 0.2, 0.35, 1.0 (used directly, no transformation)
"""

import os
import sys
import argparse
import torch
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.temporal_loader import load_event_graph


# Clinical trial edge type mapping
OUTCOME_TYPES = {
    'clinical_trial_positive::chembl': 'pos',
    'clinical_trial_unmet_efficacy::chembl': 'unmet',
    'clinical_trial_adverse_effects::chembl': 'adv',
    'clinical_trial_Unknown/Operational::chembl': 'op'
}


def extract_clinical_labels_as_of_year(graph, year_cutoff, mappings):
    """
    Extract multi-task labels for all disease-target pairs as of a given year.
    
    Args:
        graph: HeteroData temporal graph
        year_cutoff: Only include trials up to this year
        mappings: Node ID mappings (disease, target)
    
    Returns:
        DataFrame with columns: disease_id, target_id, y_pos, y_unmet, y_adv, y_op
    """
    print(f"\n📊 Extracting labels as-of {year_cutoff}")
    
    # Initialize storage for max phase per pair per outcome type
    # Key: (disease_idx, target_idx), Value: {outcome_type: max_phase_score}
    pair_labels = defaultdict(lambda: {'pos': 0.0, 'unmet': 0.0, 'adv': 0.0, 'op': 0.0})
    
    # Process each clinical trial edge type
    for edge_type in graph.edge_types:
        src_type, rel, dst_type = edge_type
        
        # Only process clinical trial edges
        if rel not in OUTCOME_TYPES:
            continue
        
        outcome = OUTCOME_TYPES[rel]
        edge_store = graph[edge_type]
        
        # Get edges, times, and phase scores
        edge_index = edge_store.edge_index
        edge_times = edge_store.edge_time
        edge_scores = edge_store.edge_attr.flatten()
        
        # Filter by year cutoff
        mask = edge_times <= year_cutoff
        
        print(f"   {rel}: {mask.sum():,} edges ≤ {year_cutoff} (out of {len(edge_times):,})")
        
        # Update max phase for each pair
        for i in range(edge_index.size(1)):
            if not mask[i]:
                continue
            
            disease_idx = edge_index[0, i].item()
            target_idx = edge_index[1, i].item()
            phase_score = edge_scores[i].item()
            
            # Update max phase for this outcome type
            pair_labels[(disease_idx, target_idx)][outcome] = max(
                pair_labels[(disease_idx, target_idx)][outcome],
                phase_score
            )
    
    print(f"   Total unique pairs: {len(pair_labels):,}")
    
    # Convert to DataFrame
    rows = []
    disease_id_map = {v: k for k, v in mappings['node_mapping']['disease'].items()}
    target_id_map = {v: k for k, v in mappings['node_mapping']['target'].items()}
    
    for (disease_idx, target_idx), labels in pair_labels.items():
        rows.append({
            'disease_id': disease_id_map[disease_idx],
            'target_id': target_id_map[target_idx],
            'y_pos': labels['pos'],
            'y_unmet': labels['unmet'],
            'y_adv': labels['adv'],
            'y_op': labels['op']
        })
    
    df = pd.DataFrame(rows)
    
    # Summary statistics
    print(f"\n   Label Statistics:")
    for col in ['y_pos', 'y_unmet', 'y_adv', 'y_op']:
        non_zero = (df[col] > 0).sum()
        print(f"     {col}: {non_zero:,} pairs with trials ({non_zero/len(df)*100:.1f}%)")
        if non_zero > 0:
            print(f"            mean={df[df[col]>0][col].mean():.3f}, max={df[col].max():.3f}")
    
    return df


def create_progression_labels(df_early, df_late, outcome='pos'):
    """
    Create binary progression labels for prospective evaluation.
    
    Args:
        df_early: Labels at early timepoint (e.g., 2017)
        df_late: Labels at late timepoint (e.g., 2024)
        outcome: Outcome type ('pos', 'unmet', 'adv', 'op')
    
    Returns:
        DataFrame with progression labels
    """
    # Merge on pair IDs
    merged = df_early.merge(
        df_late,
        on=['disease_id', 'target_id'],
        suffixes=('_early', '_late')
    )
    
    # Progression: late > early
    y_col_early = f'y_{outcome}_early'
    y_col_late = f'y_{outcome}_late'
    
    merged[f'y_prog_{outcome}'] = (merged[y_col_late] > merged[y_col_early]).astype(int)
    
    return merged


def create_eval_anchor_dataset(df_2017, df_2024):
    """
    Create evaluation dataset anchored at 2017 for prospective metrics.
    
    Args:
        df_2017: Labels as-of 2017
        df_2024: Labels as-of 2024
    
    Returns:
        DataFrame with progression labels for all outcome types
    """
    print("\n📊 Creating evaluation anchor dataset (2017 → 2024)")
    
    # Start with 2017 labels
    eval_df = df_2017.copy()
    
    # Add progression labels for each outcome type
    for outcome in ['pos', 'unmet', 'adv', 'op']:
        prog_df = create_progression_labels(df_2017, df_2024, outcome)
        eval_df = eval_df.merge(
            prog_df[['disease_id', 'target_id', f'y_prog_{outcome}']],
            on=['disease_id', 'target_id'],
            how='left'
        )
        
        # Fill NaN (pairs not in 2024) with 0
        eval_df[f'y_prog_{outcome}'].fillna(0, inplace=True)
        eval_df[f'y_prog_{outcome}'] = eval_df[f'y_prog_{outcome}'].astype(int)
    
    # Summary
    print(f"   Total pairs: {len(eval_df):,}")
    for outcome in ['pos', 'unmet', 'adv', 'op']:
        prog_count = eval_df[f'y_prog_{outcome}'].sum()
        print(f"   y_prog_{outcome}: {prog_count:,} progressed ({prog_count/len(eval_df)*100:.1f}%)")
    
    return eval_df


def main():
    parser = argparse.ArgumentParser(description="Prepare clinical trial labels")
    parser.add_argument("--graph", required=True, help="Path to temporal graph")
    parser.add_argument("--mappings", required=True, help="Path to node mappings")
    parser.add_argument("--output-dir", default="data/clinical_labels", help="Output directory")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CLINICAL TRIAL LABEL PREPARATION")
    print("="*80)
    
    # Load graph
    print(f"\n📂 Loading graph: {args.graph}")
    graph = load_event_graph(args.graph, to_undirected=False)
    
    # Load mappings
    print(f"📂 Loading mappings: {args.mappings}")
    mappings = torch.load(args.mappings)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract labels for each as-of year
    years = [2015, 2017, 2024]
    label_dfs = {}
    
    for year in years:
        df = extract_clinical_labels_as_of_year(graph, year, mappings)
        label_dfs[year] = df
        
        # Save
        output_path = output_dir / f"labels_{year}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"   ✅ Saved to {output_path}")
    
    # Create evaluation anchor dataset
    eval_df = create_eval_anchor_dataset(label_dfs[2017], label_dfs[2024])
    eval_path = output_dir / "eval_anchor2017.parquet"
    eval_df.to_parquet(eval_path, index=False)
    print(f"   ✅ Saved to {eval_path}")
    
    print("\n" + "="*80)
    print("✅ LABEL PREPARATION COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    for year in years:
        print(f"  - labels_{year}.parquet: {len(label_dfs[year]):,} pairs")
    print(f"  - eval_anchor2017.parquet: {len(eval_df):,} pairs")


if __name__ == "__main__":
    main()

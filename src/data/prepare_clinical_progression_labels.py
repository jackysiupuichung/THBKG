#!/usr/bin/env python3
"""
Extract clinical progression labels for prospective evaluation.

Implements GATher-style cohort extraction:
- Cohort A: Positive Progression (First-in-Class detection)
- Cohort B: Clinical Regression (identifying targets with efficacy failures)
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from torch_geometric.data import HeteroData


def extract_progression_labels(
    graph: HeteroData,
    val_end: int = 2017,      # End of validation period (cumulative snapshot)
    test_start: int = 2018,   # Start of test period (for filtering test-only edges)
    test_end: int = 2024,     # End of test period (cumulative snapshot)
    node_mappings: Dict = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract Cohort A and B labels for clinical progression evaluation.
    
    Uses cumulative snapshots:
    - Validation state: All edges ≤ val_end (2020)
    - Test state: All edges ≤ test_end (2024)
    
    Cohort A: Identifies gene-disease pairs that improved between snapshots
    Cohort B: Identifies targets with efficacy failures in test period only
    
    Args:
        graph: HeteroData temporal graph
        val_end: End year of validation period (snapshot boundary)
        test_start: Start year of test period (for Cohort B filtering)
        test_end: End year of test period (snapshot boundary)
        node_mappings: Dict mapping node types to ID->index mappings
        
    Returns:
        cohort_a_df: Columns [disease_idx, target_idx, progressed (0/1), score_val, score_test]
        cohort_b_df: Columns [disease_idx, target_idx, is_regression (0/1), failure_count]
    """
    print(f"\n📊 Extracting Clinical Progression Labels")
    print(f"   Validation Snapshot: ≤{val_end}")
    print(f"   Test Snapshot: ≤{test_end}")
    print(f"   Test Period (Cohort B): {test_start}-{test_end}")
    
    # 1. Extract ALL clinical trial edges at two time points (aggregate all outcomes)
    val_edges = extract_all_clinical_edges(graph, val_end)
    test_edges = extract_all_clinical_edges(graph, test_end)
    
    print(f"   Validation snapshot (≤{val_end}): {len(val_edges):,} edges")
    print(f"   Test snapshot (≤{test_end}): {len(test_edges):,} edges")
    print(f"   Unique pairs (validation): {val_edges[['disease_idx', 'target_idx']].drop_duplicates().shape[0]:,}")
    print(f"   Unique pairs (test): {test_edges[['disease_idx', 'target_idx']].drop_duplicates().shape[0]:,}")
    
    # 2. Cohort A: Find common pairs and compare scores
    cohort_a = create_cohort_a(val_edges, test_edges)
    
    # 3. Cohort B: Extract siren targets (using historical data)
    cohort_b = create_cohort_b(graph, val_end, test_start, test_end)
    
    print(f"\n✅ Cohort A (Progression): {len(cohort_a):,} pairs")
    print(f"   Progressed: {cohort_a['progressed'].sum():,} ({100*cohort_a['progressed'].mean():.1f}%)")
    print(f"   Stagnated: {(~cohort_a['progressed'].astype(bool)).sum():,} ({100*(1-cohort_a['progressed'].mean()):.1f}%)")
    
    print(f"\n✅ Cohort B (Siren Targets): {len(cohort_b):,} pairs")
    print(f"   Siren pairs: {cohort_b['is_siren'].sum():,} ({100*cohort_b['is_siren'].mean():.1f}%)")
    print(f"   Non-siren: {(~cohort_b['is_siren'].astype(bool)).sum():,} ({100*(1-cohort_b['is_siren'].mean()):.1f}%)")
    
    return cohort_a, cohort_b


def extract_all_clinical_edges(graph: HeteroData, end_year: int) -> pd.DataFrame:
    """
    Extract ALL clinical trial edges (all outcome types) up to a specific year.
    
    Aggregates across all edge types: pos, unmet, adv, op
    
    Args:
        graph: HeteroData graph
        end_year: Include all edges ≤ this year
        
    Returns:
        DataFrame with [disease_idx, target_idx, score, year, edge_type]
    """
    all_edges = []
    
    for edge_type in ['pos', 'unmet', 'adv', 'op']:
        edges_df = extract_cumulative_edges(graph, end_year, edge_type=edge_type)
        if len(edges_df) > 0:
            edges_df['edge_type'] = edge_type
            all_edges.append(edges_df)
    
    if not all_edges:
        return pd.DataFrame(columns=['disease_idx', 'target_idx', 'score', 'year', 'edge_type'])
    
    combined = pd.concat(all_edges, ignore_index=True)
    return combined


def extract_cumulative_edges(
    graph: HeteroData, 
    end_year: int,
    edge_type: str = 'pos'
) -> pd.DataFrame:
    """
    Extract cumulative edges up to a specific year (snapshot).
    
    Args:
        graph: HeteroData graph
        end_year: Include all edges ≤ this year
        edge_type: One of ['pos', 'unmet', 'adv', 'op']
        
    Returns:
        DataFrame with [disease_idx, target_idx, score, year]
    """
    # Map edge type to full relation name
    edge_map = {
        'pos': 'clinical_trial_positive::chembl',
        'unmet': 'clinical_trial_unmet_efficacy::chembl',
        'adv': 'clinical_trial_adverse_effects::chembl',
        'op': 'clinical_trial_Unknown/Operational::chembl'
    }
    
    full_etype = ('disease', edge_map[edge_type], 'target')
    
    if full_etype not in graph.edge_types:
        print(f"   Warning: {full_etype} not found in graph")
        return pd.DataFrame(columns=['disease_idx', 'target_idx', 'score', 'year'])
    
    # Get edges
    edge_index = graph[full_etype].edge_index.cpu().numpy()
    edge_time = graph[full_etype].edge_time.cpu().numpy()
    edge_attr = graph[full_etype].edge_attr.cpu().numpy() if hasattr(graph[full_etype], 'edge_attr') else None
    
    # Filter by time (cumulative: all edges ≤ end_year)
    mask = edge_time <= end_year
    
    # Handle edge_attr (might be 2D, flatten to 1D)
    if edge_attr is not None:
        if edge_attr.ndim > 1:
            edge_attr = edge_attr.flatten() if edge_attr.shape[1] == 1 else edge_attr[:, 0]
        scores = edge_attr[mask]
    else:
        scores = np.ones(mask.sum())
    
    df = pd.DataFrame({
        'disease_idx': edge_index[0][mask],
        'target_idx': edge_index[1][mask],
        'year': edge_time[mask],
        'score': scores
    })
    
    return df


def extract_test_period_edges(
    graph: HeteroData,
    start_year: int,
    end_year: int,
    edge_type: str = 'unmet'
) -> pd.DataFrame:
    """
    Extract edges from a specific time period (for Cohort B failures).
    
    Unlike cumulative snapshots, this extracts edges that occurred
    WITHIN the specified period only.
    
    Args:
        graph: HeteroData graph
        start_year, end_year: Period boundaries
        edge_type: One of ['pos', 'unmet', 'adv', 'op']
        
    Returns:
        DataFrame with [disease_idx, target_idx, score, year]
    """
    edge_map = {
        'pos': 'clinical_trial_positive::chembl',
        'unmet': 'clinical_trial_unmet_efficacy::chembl',
        'adv': 'clinical_trial_adverse_effects::chembl',
        'op': 'clinical_trial_Unknown/Operational::chembl'
    }
    
    full_etype = ('disease', edge_map[edge_type], 'target')
    
    if full_etype not in graph.edge_types:
        return pd.DataFrame(columns=['disease_idx', 'target_idx', 'score', 'year'])
    
    edge_index = graph[full_etype].edge_index.cpu().numpy()
    edge_time = graph[full_etype].edge_time.cpu().numpy()
    edge_attr = graph[full_etype].edge_attr.cpu().numpy() if hasattr(graph[full_etype], 'edge_attr') else None
    
    # Filter by period (start_year ≤ edge_time ≤ end_year)
    mask = (edge_time >= start_year) & (edge_time <= end_year)
    
    # Handle edge_attr (might be 2D, flatten to 1D)
    if edge_attr is not None:
        if edge_attr.ndim > 1:
            edge_attr = edge_attr.flatten() if edge_attr.shape[1] == 1 else edge_attr[:, 0]
        scores = edge_attr[mask]
    else:
        scores = np.ones(mask.sum())
    
    df = pd.DataFrame({
        'disease_idx': edge_index[0][mask],
        'target_idx': edge_index[1][mask],
        'year': edge_time[mask],
        'score': scores
    })
    
    return df


def create_cohort_a(val_edges: pd.DataFrame, test_edges: pd.DataFrame) -> pd.DataFrame:
    """
    Create Cohort A: Positive Progression labels (Exact GATher Methodology).
    
    **Evaluation Setup** (Figure 3C):
    - Baseline: Pairs active in clinical pipeline ≤2017 (validation)
    - Question: "Which pairs will show positive progression by 2024?"
    
    **Positive (Winners)**:
    1. Baseline pairs that INCREASED max phase (score_test > score_val) or
    2. Baseline pairs that has SAME max phase without positive outcome but has positive outcome in test
    
    **Negative (Stagnant)**:
    - Baseline pairs that has no positive outcome in test
    
    Args:
        val_edges: Validation snapshot edges (all types, with edge_type)
        test_edges: Test snapshot edges (all types, with edge_type)
        
    Returns:
        DataFrame with progression labels
    """
    # 1. Establish BASELINE: pairs active in validation period (2017)
    val_pairs = val_edges[['disease_idx', 'target_idx']].drop_duplicates()
    val_scores = val_edges.groupby(['disease_idx', 'target_idx'])['score'].max().reset_index()
    val_scores.columns = ['disease_idx', 'target_idx', 'score_val']
    
    # 2. Get test POSITIVE outcomes only
    test_pos = test_edges[test_edges['edge_type'] == 'pos'].copy()
    test_pos_scores = test_pos.groupby(['disease_idx', 'target_idx'])['score'].max().reset_index()
    test_pos_scores.columns = ['disease_idx', 'target_idx', 'score_test']
    
    # 3. Merge baseline with test outcomes (OUTER join - include novel pairs)
    cohort_a = val_scores.merge(
        test_pos_scores,
        on=['disease_idx', 'target_idx'],
        how='outer'
    )
    
    # Fill missing scores
    cohort_a['score_val'] = cohort_a['score_val'].fillna(0)
    cohort_a['score_test'] = cohort_a['score_test'].fillna(0)
    
    # 4. Label progression:
    # Positive if:
    #   a) Novel pair with positive outcome (score_val=0, score_test>0)
    #   b) Baseline pair with positive outcome AND strict increase (score_test > score_val > 0)
    cohort_a['progressed'] = (
        (cohort_a['score_test'] > 0) &  # Has positive outcome in test
        (cohort_a['score_test'] > cohort_a['score_val'])  # Novel OR increased
    ).astype(int)
    
    # Count subcategories for reporting
    novel_positive = ((cohort_a['score_val'] == 0) & (cohort_a['score_test'] > 0)).sum()
    baseline_increased = ((cohort_a['score_val'] > 0) & (cohort_a['score_test'] > cohort_a['score_val'])).sum()
    
    print(f"\n📋 Cohort A Breakdown:")
    print(f"   Baseline pairs: {(cohort_a['score_val'] > 0).sum():,}")
    print(f"   Novel positive pairs: {novel_positive:,}")
    print(f"   Baseline pairs that increased: {baseline_increased:,}")
    
    return cohort_a[['disease_idx', 'target_idx', 'score_val', 'score_test', 'progressed']]


def create_cohort_b(
    graph: HeteroData,
    val_end: int,
    test_start: int,
    test_end: int,
    min_failures: int = 2,
    min_diseases: int = 2,
    max_phase: float = 3.0
) -> pd.DataFrame:
    """
    Create Cohort B: Siren Target labels (GATher methodology with temporal split).
    
    **Prospective Evaluation**:
    1. Identify siren targets using HISTORICAL data (≤ val_end)
    2. Evaluate on TEST period pairs (test snapshot)
    
    **Siren Target Criteria** (using historical data):
    1. **Max phase ≤ 3** as of val_end (stuck in pipeline)
    2. **≥2 efficacy failures** as of val_end
    3. **Across ≥2 different diseases**
    
    Args:
        graph: HeteroData graph
        val_end: End of validation period (for siren identification)
        test_start: Test period start
        test_end: Test period end (for evaluation pairs)
        min_failures: Minimum failure count to be siren
        min_diseases: Minimum number of diseases affected
        max_phase: Maximum phase to be considered "stuck"
        
    Returns:
        DataFrame with is_siren labels for TEST period pairs
    """
    # Step 1: Identify siren targets using HISTORICAL data (≤ val_end)
    print(f"\n🎯 Siren Target Identification (Historical ≤{val_end}):")
    
    # Get historical clinical edges for max phase
    val_edges = extract_all_clinical_edges(graph, val_end)
    target_max_phase_val = val_edges.groupby('target_idx')['score'].max().reset_index()
    target_max_phase_val.columns = ['target_idx', 'max_phase_val']
    
    # Get historical failures
    unmet_val = extract_cumulative_edges(graph, val_end, edge_type='unmet')
    adv_val = extract_cumulative_edges(graph, val_end, edge_type='adv')
    failures_val = pd.concat([unmet_val, adv_val], ignore_index=True)
    
    if len(failures_val) == 0:
        # No historical failures - all pairs are non-siren
        test_edges = extract_all_clinical_edges(graph, test_end)
        all_pairs = test_edges[['disease_idx', 'target_idx']].drop_duplicates()
        all_pairs['is_siren'] = 0
        all_pairs['siren_target'] = False
        return all_pairs
    
    # Count historical failures per target
    target_failures_val = failures_val.groupby('target_idx').agg({
        'disease_idx': 'nunique',
        'score': 'count'
    }).reset_index()
    target_failures_val.columns = ['target_idx', 'num_diseases', 'num_failures']
    
    # Merge with max phase from validation period
    target_failures_val = target_failures_val.merge(target_max_phase_val, on='target_idx', how='left')
    
    # Identify siren targets based on HISTORICAL data
    siren_targets = target_failures_val[
        (target_failures_val['num_failures'] >= min_failures) &
        (target_failures_val['num_diseases'] >= min_diseases) &
        (target_failures_val['max_phase_val'] <= max_phase)
    ]['target_idx'].tolist()
    
    print(f"   Total unique targets with failures: {len(target_failures_val):,}")
    print(f"   Targets with ≥{min_failures} failures: {(target_failures_val['num_failures'] >= min_failures).sum():,}")
    print(f"   Targets with failures across ≥{min_diseases} diseases: {(target_failures_val['num_diseases'] >= min_diseases).sum():,}")
    print(f"   Targets with max phase ≤{max_phase}: {(target_failures_val['max_phase_val'] <= max_phase).sum():,}")
    print(f"   → Siren targets identified: {len(siren_targets):,}")
    
    # Step 2: Label TEST period pairs based on historical siren targets
    test_edges = extract_all_clinical_edges(graph, test_end)
    all_pairs = test_edges[['disease_idx', 'target_idx']].drop_duplicates()
    
    # Mark siren pairs
    all_pairs['siren_target'] = all_pairs['target_idx'].isin(siren_targets)
    all_pairs['is_siren'] = all_pairs['siren_target'].astype(int)
    
    return all_pairs


def save_labels(cohort_a: pd.DataFrame, cohort_b: pd.DataFrame, output_dir: str = 'data/labels'):
    """Save cohort labels to CSV files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    cohort_a.to_csv(f'{output_dir}/cohort_a_progression.csv', index=False)
    cohort_b.to_csv(f'{output_dir}/cohort_b_siren.csv', index=False)
    
    print(f"\n💾 Saved labels to {output_dir}/")
    print(f"   - cohort_a_progression.csv")
    print(f"   - cohort_b_siren.csv")


if __name__ == '__main__':
    # Example usage
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    from src.data.temporal_loader import load_event_graph
    
    print("Loading graph...")
    graph = torch.load('output/graph/hetero_graph_with_features.pt')
    
    # Extract labels
    cohort_a, cohort_b = extract_progression_labels(graph)
    
    # Save to disk
    save_labels(cohort_a, cohort_b)
    
    print("\n✅ Clinical progression labels extracted successfully!")

#!/usr/bin/env python3
"""
Step 05 (Revised): Build TFT longitudinal dataset from pre-parsed edge parquets.

Architecture:
  1. Load clinical trial edges (for anchors only)
  2. Load all other edges (for features)
  3. Build anchor table: Phase 2+ threshold, partition by year
  4. Build long-format features: harmonic sum + novelty per source per absolute year
  5. Align and impute: merge anchors with features, fill missing years
  6. Pivot to wide format and save

Best practices:
  - Type hints, docstrings, modular functions
  - Explicit year handling (not relative_year until the end)
  - Robust null handling with imputation strategy
  - Validation & logging at each step
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
from typing import Tuple, Optional, List
from dataclasses import dataclass

# --- Import shared utilities from Step 01 pipeline ---
_PIPELINE_DIR = str(Path(__file__).resolve().parents[1] / "temporal_graph" / "pipeline")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, _PIPELINE_DIR)

from build_event_list import harmonic_sum, aggregate_scores


# ──────────────────────────────────────────────
# Constants & Config
# ──────────────────────────────────────────────

PHASE_THRESHOLDS = {1: 0.1, 2: 0.2, 3: 0.7, 4: 1.0}
ANCHOR_SCORE_MIN = PHASE_THRESHOLDS[2]  # 0.2 (Phase 2)
OUTCOME_SCORE_MIN = PHASE_THRESHOLDS[3]  # 0.7 (Phase 3)


@dataclass
class TFTConfig:
    """Configuration for TFT dataset building."""
    train_max: int = 2014
    val_max: int = 2015
    test_max: int = 2022
    outcome_max: int = 2024
    lookback: int = 10
    start_year: int = 1990
    imputation_strategy: str = 'zero'  # 'zero' or 'forward_fill'
    sample_ratio: Optional[float] = None


# ──────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────

def get_phase(score: float) -> int:
    """Map score to clinical phase using thresholds."""
    if score >= 1.0:
        return 4
    if score >= 0.7:
        return 3
    if score >= 0.2:
        return 2
    if score >= 0.1:
        return 1
    return 0


def assert_columns_exist(df: pd.DataFrame, required_cols: List[str], context: str = "") -> None:
    """Assert that required columns exist in DataFrame."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {context}. Available: {df.columns.tolist()}")


# ──────────────────────────────────────────────
# Step 1: Load Edges (separated by purpose)
# ──────────────────────────────────────────────

def load_clinical_trial_edges(
    edges_dir: str,
    sample_ratio: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load only clinical_trial_*.parquet files.
    
    Args:
        edges_dir: Directory containing edge parquet files
        sample_ratio: Optional float (0.0-1.0) to sample each file
    
    Returns:
        DataFrame with clinical trial edges
    """
    files = sorted(glob(os.path.join(edges_dir, "*clinical_trial*.parquet")))
    if not files:
        raise FileNotFoundError(f"No clinical_trial_*.parquet files found in {edges_dir}")
    
    print(f"📂 Loading {len(files)} clinical trial parquets from {edges_dir}")
    dfs = []
    for pf in tqdm(files, desc="Loading clinical trial edges"):
        try:
            df = pd.read_parquet(pf)
            if sample_ratio and 0 < sample_ratio < 1.0:
                df = df.sample(frac=sample_ratio, random_state=42)
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Error reading {Path(pf).name}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"   Loaded {len(combined):,} clinical trial edge records")
    return combined


def load_feature_edges(
    edges_dir: str,
    exclude_clinical: bool = True,
    sample_ratio: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load target-disease edge parquets (for features).
    
    Filters to edges with relation containing 'target' and 'disease',
    which represent target-disease associations from various data sources.
    
    Args:
        edges_dir: Directory containing edge parquet files
        exclude_clinical: If True, skip clinical_trial_*.parquet files
        sample_ratio: Optional float (0.0-1.0) to sample each file
    
    Returns:
        DataFrame with target-disease feature edges
    """
    all_files = sorted(glob(os.path.join(edges_dir, "*.parquet")))
    
    # Filter to files that contain 'target' and 'disease' in the filename (target-disease edges)
    files = [
        f for f in all_files
        if ('target' in os.path.basename(f).lower() and 'disease' in os.path.basename(f).lower())
    ]
    
    if exclude_clinical:
        files = [f for f in files if 'clinical_trial' not in os.path.basename(f)]
    
    if not files:
        raise FileNotFoundError(f"No target-disease parquet files found in {edges_dir}")
    
    print(f"📂 Loading {len(files)} target-disease feature parquets from {edges_dir}")
    dfs = []
    for pf in tqdm(files, desc="Loading feature edges"):
        try:
            df = pd.read_parquet(pf)
            if sample_ratio and 0 < sample_ratio < 1.0:
                df = df.sample(frac=sample_ratio, random_state=42)
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Error reading {Path(pf).name}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"   Loaded {len(combined):,} target-disease feature edge records")
    return combined


# ──────────────────────────────────────────────
# Step 2: Build Anchor Table (from clinical trials only)
# ──────────────────────────────────────────────

def build_anchor_table(
    clinical_edges: pd.DataFrame,
    config: TFTConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build anchor table from clinical trial edges.
    
    T=0 is defined as the first year a TD pair reaches Phase 2 (score ≥ 0.2).
    
    Args:
        clinical_edges: Clinical trial edges (pre-filtered)
        config: TFT configuration
    
    Returns:
        (anchors DataFrame, id_cols list)
    """
    print("\n⚓ Building anchor table from clinical trial edges...")
    
    # Validate columns
    year_col = 'year'
    score_col = 'score'
    assert_columns_exist(clinical_edges, [year_col, score_col], "clinical_edges")
    
    # Ensure year and score are numeric
    clinical = clinical_edges.copy()
    clinical[year_col] = pd.to_numeric(clinical[year_col], errors='coerce')
    clinical[score_col] = pd.to_numeric(clinical[score_col], errors='coerce')
    
    # Remove rows with missing year or score
    clinical = clinical.dropna(subset=[year_col, score_col])
    print(f"   After removing NaN year/score: {len(clinical):,} records")
    
    # Detect TD identifier columns
    id_cols = [c for c in ['sourceId', 'targetId'] if c in clinical.columns]
    if not id_cols or len(id_cols) < 2:
        raise ValueError(f"Could not detect TD identifier columns. Available: {clinical.columns.tolist()}")
    print(f"   TD identifier columns: {id_cols}")
    
    # Phase filtering: Phase 2+
    clinical['phase'] = clinical[score_col].apply(get_phase)
    phase2_plus = clinical[clinical['phase'] >= 2]
    print(f"   Records with Phase 2+ (score ≥ {ANCHOR_SCORE_MIN}): {len(phase2_plus):,}")
    
    # T=0: first year with Phase 2+
    anchors = (
        phase2_plus.groupby(id_cols)[year_col]
        .min()
        .reset_index()
        .rename(columns={year_col: 'anchor_year'})
    )
    print(f"   Unique pairs with Phase 2+: {len(anchors):,}")
    
    # Filter out pairs that reached Phase 3+ at or before anchor year
    print("   Filtering out pairs already in Phase 3+ at or before anchor year...")
    pre_success = (
        clinical[clinical['phase'] >= 3]
        .groupby(id_cols)[year_col]
        .min()
        .reset_index()
        .rename(columns={year_col: 'first_phase3_year'})
    )
    
    anchors = anchors.merge(pre_success, on=id_cols, how='left')
    anchors = anchors[~(anchors['first_phase3_year'] <= anchors['anchor_year'])].copy()
    anchors = anchors.drop('first_phase3_year', axis=1)
    print(f"   Pairs remaining after Phase 3+ pre-filter: {len(anchors):,}")
    
    # Partition by anchor year
    print(f"🗓  Partitions: Train ≤ {config.train_max}, Val [{config.train_max+1}–{config.val_max}], Test [{config.val_max+1}–{config.test_max}]")
    anchors['partition'] = 'excluded'
    anchors.loc[
        (anchors['anchor_year'] >= config.start_year) & (anchors['anchor_year'] <= config.train_max),
        'partition'
    ] = 'train'
    anchors.loc[
        (anchors['anchor_year'] > config.train_max) & (anchors['anchor_year'] <= config.val_max),
        'partition'
    ] = 'val'
    anchors.loc[
        (anchors['anchor_year'] > config.val_max) & (anchors['anchor_year'] <= config.test_max),
        'partition'
    ] = 'test'
    
    anchors = anchors[anchors['partition'] != 'excluded'].copy()
    print(f"   Partition counts:\n{anchors['partition'].value_counts()}")
    
    # Outcome: reach Phase 3+ in (anchor_year, outcome_max]
    future = (
        clinical[clinical['phase'] >= 3]
        .merge(anchors[id_cols + ['anchor_year']], on=id_cols)
    )
    future = future[
        (future[year_col] > future['anchor_year']) & 
        (future[year_col] <= config.outcome_max)
    ]
    
    positive_pairs = set(map(tuple, future[id_cols].drop_duplicates().values))
    anchors['outcome'] = anchors.apply(
        lambda row: int(tuple(row[c] for c in id_cols) in positive_pairs), axis=1
    )
    
    pos_rate = anchors['outcome'].mean() * 100
    print(f"   Overall positive rate: {pos_rate:.1f}%")
    
    return anchors, id_cols


# ──────────────────────────────────────────────
# Step 3: Build Long-Format Features (absolute year)
# ──────────────────────────────────────────────

def build_dynamic_features_long(
    edges: pd.DataFrame,
    config: TFTConfig,
) -> pd.DataFrame:
    """
    Build source-level features per (TD pair, datasource, year) in long format.
    
    Returns:
        DataFrame with columns: [sourceId, targetId, datasourceId, year, S, N, P]
        where:
        - S: harmonic association score
        - N: novelty (change from previous year, clipped ≥ 0)
        - P: max clinical phase reached
    """
    print("\n📈 Building long-format dynamic features...")
    
    # Validate columns
    year_col = 'year'
    score_col = 'score'
    source_col = 'datasourceId'
    assert_columns_exist(edges, [year_col, score_col, source_col], "feature edges")
    
    # Ensure year and score are numeric
    edges = edges.copy()
    edges[year_col] = pd.to_numeric(edges[year_col], errors='coerce')
    edges[score_col] = pd.to_numeric(edges[score_col], errors='coerce')
    
    # Remove rows with missing year or score
    edges = edges.dropna(subset=[year_col, score_col])
    print(f"   After removing NaN year/score: {len(edges):,} records")
    
    # Detect TD identifier columns
    id_cols = [c for c in ['sourceId', 'targetId'] if c in edges.columns]
    if not id_cols or len(id_cols) < 2:
        raise ValueError(f"Could not detect TD identifier columns. Available: {edges.columns.tolist()}")
    
    # Phase mapping
    edges['phase'] = edges[score_col].apply(get_phase)
    
    # For each year, aggregate cumulative evidence
    all_events = []
    year_range = range(config.start_year, config.outcome_max + 1)
    
    for year in tqdm(year_range, desc="Processing years for cumulative aggregation"):
        # Get all evidence up to and including this year (cumulative)
        cumulative = edges[edges[year_col] <= year].copy()
        
        if cumulative.empty:
            continue
        
        # Group by (TD pair, source) and aggregate
        group_cols = id_cols + [source_col]
        agg = cumulative.groupby(group_cols, as_index=False).agg({
            score_col: lambda x: harmonic_sum(x.values),
            'phase': 'max'
        })
        agg.columns = list(id_cols) + [source_col, 'S', 'P']
        agg['year'] = year
        
        all_events.append(agg)
    
    if not all_events:
        print("   ⚠️ No events generated!")
        return pd.DataFrame()
    
    # Combine all years
    features = pd.concat(all_events, ignore_index=True)
    print(f"   Generated {len(features):,} (pair, source, year) combinations")
    
    # Compute novelty N = S_t - S_{t-1}
    features = features.sort_values(id_cols + [source_col, 'year'])
    features['S_prev'] = features.groupby(id_cols + [source_col])['S'].shift(1).fillna(0)
    features['N'] = (features['S'] - features['S_prev']).clip(lower=0)
    features = features.drop('S_prev', axis=1)
    
    print(f"   Feature columns: S (score), N (novelty), P (phase)")
    
    return features


# ──────────────────────────────────────────────
# Step 4: Align and Impute Features
# ──────────────────────────────────────────────

def align_and_impute_features(
    anchors: pd.DataFrame,
    features_long: pd.DataFrame,
    id_cols: List[str],
    config: TFTConfig,
) -> pd.DataFrame:
    """
    Align features with anchors, imputing missing years.
    
    For each anchor pair:
      1. Get year range: [anchor_year - lookback, anchor_year - 1]
      2. For each year, try to fetch features from features_long
      3. If missing, impute via strategy (forward_fill or zero)
      4. Convert to relative_year
    
    Args:
        anchors: Anchor table with anchor_year
        features_long: Long-format features
        id_cols: TD identifier columns
        config: TFT configuration
    
    Returns:
        Long-format DataFrame ready for pivoting: [sourceId, targetId, relative_year, datasourceId, S, N, P]
    """
    print("\n🧩 Aligning and imputing features...")
    
    if features_long.empty:
        print("   ⚠️ No features provided. Returning empty DataFrame.")
        return pd.DataFrame()
    
    features_long = features_long.copy()
    
    # For each anchor pair, build time series
    aligned_rows = []
    missing_count = 0
    imputed_count = 0
    pairs_with_features = 0
    pairs_without_features = 0
    
    for _, anchor_row in tqdm(anchors.iterrows(), total=len(anchors), desc="Aligning features"):
        pair_key = tuple(anchor_row[c] for c in id_cols)
        anchor_year = int(anchor_row['anchor_year'])
        
        # Year range for this pair
        year_min = anchor_year - config.lookback
        year_max = anchor_year - 1
        year_range = list(range(year_min, year_max + 1))
        
        # Get all sources active for this pair (filter by both id columns)
        mask = (features_long[id_cols[0]] == pair_key[0]) & (features_long[id_cols[1]] == pair_key[1])
        pair_features = features_long[mask]
        
        if pair_features.empty:
            # No features for this pair; add zero-valued rows for grid completion
            pairs_without_features += 1
            # Don't add rows here; just skip and let the grid fill with NaN->0
            continue
        
        pairs_with_features += 1
        sources = pair_features['datasourceId'].unique()
        
        for source in sources:
            source_data = pair_features[pair_features['datasourceId'] == source].copy()
            source_data = source_data.sort_values('year')
            
            # Build a dict for fast lookup
            source_dict = dict(zip(source_data['year'].astype(int), source_data.to_dict('records')))
            
            last_val = None  # For forward-fill imputation
            
            for year in year_range:
                if year in source_dict:
                    # Feature exists
                    row = source_dict[year]
                    aligned_rows.append({
                        id_cols[0]: pair_key[0],
                        id_cols[1]: pair_key[1],
                        'year': year,
                        'relative_year': year - anchor_year,
                        'datasourceId': source,
                        'S': float(row['S']),
                        'N': float(row['N']),
                        'P': float(row['P']),
                    })
                    last_val = row
                else:
                    # Feature missing; impute
                    if config.imputation_strategy == 'forward_fill' and last_val is not None:
                        aligned_rows.append({
                            id_cols[0]: pair_key[0],
                            id_cols[1]: pair_key[1],
                            'year': year,
                            'relative_year': year - anchor_year,
                            'datasourceId': source,
                            'S': float(last_val['S']),
                            'N': 0.0,  # No novelty for imputed values
                            'P': float(last_val['P']),
                        })
                    else:
                        aligned_rows.append({
                            id_cols[0]: pair_key[0],
                            id_cols[1]: pair_key[1],
                            'year': year,
                            'relative_year': year - anchor_year,
                            'datasourceId': source,
                            'S': 0.0,
                            'N': 0.0,
                            'P': 0.0,
                        })
                    imputed_count += 1
    
    aligned = pd.DataFrame(aligned_rows)
    print(f"   Pairs with features: {pairs_with_features:,}")
    print(f"   Pairs without features: {pairs_without_features:,}")
    print(f"   Aligned {len(aligned):,} feature values ({imputed_count:,} imputed)")
    
    return aligned


# ──────────────────────────────────────────────
# Step 5: Pivot to Wide Format
# ──────────────────────────────────────────────

def pivot_to_wide(
    aligned_long: pd.DataFrame,
    id_cols: List[str],
) -> pd.DataFrame:
    """
    Pivot long-format features to wide format (sources as columns).
    
    Args:
        aligned_long: [sourceId, targetId, relative_year, datasourceId, S, N, P]
        id_cols: TD identifier columns
    
    Returns:
        Wide-format DataFrame: [sourceId, targetId, relative_year, source_1_S, source_1_N, source_1_P, ...]
    """
    print("\n📊 Pivoting to wide format...")
    
    if aligned_long.empty:
        print("   ⚠️ No features to pivot.")
        return pd.DataFrame()
    
    # Separate pivots for each metric
    s_wide = aligned_long.pivot_table(
        index=id_cols + ['relative_year'],
        columns='datasourceId',
        values='S',
        fill_value=0.0
    )
    n_wide = aligned_long.pivot_table(
        index=id_cols + ['relative_year'],
        columns='datasourceId',
        values='N',
        fill_value=0.0
    )
    p_wide = aligned_long.pivot_table(
        index=id_cols + ['relative_year'],
        columns='datasourceId',
        values='P',
        fill_value=0.0
    )
    
    # Rename columns
    s_wide.columns = [f"{c}_S" for c in s_wide.columns]
    n_wide.columns = [f"{c}_N" for c in n_wide.columns]
    p_wide.columns = [f"{c}_P" for c in p_wide.columns]
    
    # Join and reset index
    wide = s_wide.join(n_wide).join(p_wide).reset_index()
    
    print(f"   Output shape: {wide.shape}")
    print(f"   Feature columns ({len(wide.columns) - len(id_cols) - 1}): {list(wide.columns)[len(id_cols)+1:10]}...")
    
    return wide


# ──────────────────────────────────────────────
# Step 6: Merge and Final Assembly
# ──────────────────────────────────────────────

def merge_anchors_and_features(
    anchors: pd.DataFrame,
    features_wide: pd.DataFrame,
    id_cols: List[str],
    lookback: int,
) -> pd.DataFrame:
    """
    Merge anchor metadata with feature sequences.
    
    Args:
        anchors: Anchor table
        features_wide: Wide-format features
        id_cols: TD identifier columns
        lookback: Lookback window
    
    Returns:
        Final dataset: [sourceId, targetId, relative_year, anchor_year, partition, outcome, feat_1_S, ...]
    """
    print("\n🔗 Merging anchors and features...")
    
    # Create grid of all (TD pair, relative_year) combinations
    grid = anchors[id_cols + ['anchor_year', 'partition', 'outcome']].copy()
    year_range = pd.DataFrame({'relative_year': range(-lookback, 0)})
    grid = grid.assign(_key=1).merge(year_range.assign(_key=1), on='_key').drop('_key', axis=1)
    
    # Merge with features
    if not features_wide.empty:
        final = grid.merge(
            features_wide,
            on=id_cols + ['relative_year'],
            how='left'
        )
    else:
        final = grid.copy()
    
    # Fill NAs in feature columns (should be minimal if imputation worked well)
    feature_cols = [c for c in final.columns if any(c.endswith(s) for s in ['_S', '_N', '_P'])]
    final[feature_cols] = final[feature_cols].fillna(0.0)
    
    print(f"   Final dataset shape: {final.shape}")
    print(f"   Feature columns: {len(feature_cols)}")
    
    return final


# ──────────────────────────────────────────────
# Main Orchestration
# ──────────────────────────────────────────────

def build_tft_dataset_v2(
    raw_edges_dir: str,
    output_dir: str,
    config: Optional[TFTConfig] = None,
) -> None:
    """
    Orchestrate the full TFT dataset building pipeline.
    
    Args:
        raw_edges_dir: Directory with pre-parsed edge parquets
        output_dir: Output directory for TFT dataset parquets
        config: TFT configuration (uses defaults if None)
    """
    if config is None:
        config = TFTConfig()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TFT DATASET BUILDING (v2)")
    print("="*80)
    print(f"Config:")
    print(f"  Train max: {config.train_max}")
    print(f"  Val max: {config.val_max}")
    print(f"  Test max: {config.test_max}")
    print(f"  Outcome max: {config.outcome_max}")
    print(f"  Lookback: {config.lookback} years")
    print(f"  Imputation: {config.imputation_strategy}")
    print("="*80)
    
    # 1. Load clinical trial edges
    print("\n[1/6] Loading clinical trial edges...")
    clinical = load_clinical_trial_edges(raw_edges_dir, sample_ratio=config.sample_ratio)
    
    # 2. Load feature edges
    print("\n[2/6] Loading feature edges...")
    features_raw = load_feature_edges(
        raw_edges_dir,
        exclude_clinical=True,
        sample_ratio=config.sample_ratio
    )
    
    # 3. Build anchor table
    print("\n[3/6] Building anchor table...")
    anchors, id_cols = build_anchor_table(clinical, config)
    
    # 4. Build long-format features
    print("\n[4/6] Building long-format features...")
    features_long = build_dynamic_features_long(features_raw, config)
    
    # 5. Align and impute
    print("\n[5/6] Aligning and imputing features...")
    aligned = align_and_impute_features(anchors, features_long, id_cols, config)
    
    # 6. Pivot to wide and merge
    print("\n[6/6] Pivoting and merging...")
    features_wide = pivot_to_wide(aligned, id_cols)
    final = merge_anchors_and_features(anchors, features_wide, id_cols, config.lookback)
    
    # Summary
    print("\n" + "="*80)
    print("📊 DATASET SUMMARY")
    print("="*80)
    pairs = final.drop_duplicates(id_cols)
    summary = pairs.groupby(['partition', 'outcome']).size().unstack(fill_value=0)
    if 1 in summary.columns and 0 in summary.columns:
        summary['pos_rate (%)'] = (summary[1] / (summary[0] + summary[1]) * 100).round(2)
    print("\nPair Outcomes per Partition:")
    print(summary)
    
    feature_cols = [c for c in final.columns if any(c.endswith(s) for s in ['_S', '_N', '_P'])]
    print(f"\nTime steps per pair: {config.lookback}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:6]}{'...' if len(feature_cols) > 6 else ''}")
    print("="*80)
    
    # Save outputs
    print(f"\n💾 Saving outputs to {output_path}...")
    out_file = output_path / "tft_longitudinal.parquet"
    final.to_parquet(out_file, index=False)
    print(f"   ✅ Longitudinal dataset: {out_file}")
    
    anchors_file = output_path / "tft_anchors.parquet"
    anchors.to_parquet(anchors_file, index=False)
    print(f"   ✅ Anchor table: {anchors_file}")
    
    features_file = output_path / "tft_features_long.parquet"
    if not features_long.empty:
        features_long.to_parquet(features_file, index=False)
        print(f"   ✅ Long-format features: {features_file}")
    
    print("\n✅ TFT Dataset Building Complete!")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 05 (v2): Build TFT longitudinal dataset from edge parquets"
    )
    parser.add_argument(
        "--raw-edges-dir",
        default="output/evidences/edges",
        help="Directory with pre-parsed edge parquets"
    )
    parser.add_argument(
        "--output-dir",
        default="output/tft_dataset",
        help="Output directory for TFT dataset parquets"
    )
    parser.add_argument("--train-max", type=int, default=2014)
    parser.add_argument("--val-max", type=int, default=2015)
    parser.add_argument("--test-max", type=int, default=2022)
    parser.add_argument("--outcome-max", type=int, default=2024)
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--start-year", type=int, default=1990)
    parser.add_argument(
        "--imputation-strategy",
        choices=['zero', 'forward_fill'],
        default='zero',
        help="Strategy for missing years: 'zero' or 'forward_fill'"
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=None,
        help="Sample fraction of edges for testing (e.g., 0.01)"
    )
    
    args = parser.parse_args()
    
    config = TFTConfig(
        train_max=args.train_max,
        val_max=args.val_max,
        test_max=args.test_max,
        outcome_max=args.outcome_max,
        lookback=args.lookback,
        start_year=args.start_year,
        imputation_strategy=args.imputation_strategy,
        sample_ratio=args.sample_ratio,
    )
    
    build_tft_dataset_v2(args.raw_edges_dir, args.output_dir, config)


if __name__ == "__main__":
    main()

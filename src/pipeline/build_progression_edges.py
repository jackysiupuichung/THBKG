#!/usr/bin/env python3
"""
Build progression edges with harmonic sum aggregation.

Processes raw edges to create source-level progression edges with:
- Datasource-specific cutoffs
- Harmonic sum score aggregation
- Per-year output directories
- Static edge separation

Replaces build_progression_graph.py
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from tqdm import tqdm


# ============================================================
# UTILITIES
# ============================================================

def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def harmonic_sum(scores, max_harmonic=1.644):
    """
    Compute harmonic sum of top-50 scores (Open Targets standard).
    
    Args:
        scores: Array of scores
        max_harmonic: Maximum harmonic value for normalization
        
    Returns:
        Normalized harmonic sum
    """
    if len(scores) == 0:
        return 0.0
    s = np.sort(scores)[::-1][:50]
    idx = np.arange(1, len(s) + 1)
    return np.sum(s / (idx ** 2)) / max_harmonic


def load_all_edges(directory):
    """
    Load all parquet files from directory.
    
    Args:
        directory: Directory containing edge parquet files
        
    Returns:
        DataFrame with all edges
    """
    dfs = []
    parquet_files = glob(os.path.join(directory, "*.parquet"))
    
    for pq in tqdm(parquet_files, desc="Loading edges"):
        try:
            df = pd.read_parquet(pq)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {pq}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


# ============================================================
# PROCESSING
# ============================================================

def apply_cutoffs(edges, config):
    """
    Apply datasource-specific cutoffs.
    
    Args:
        edges: DataFrame with edges
        config: Configuration dict with datasource cutoffs
        
    Returns:
        Filtered DataFrame
    """
    if 'datasources' not in config:
        return edges
    
    filtered = []
    
    for datasource, params in config['datasources'].items():
        ds_edges = edges[edges['datasourceId'] == datasource].copy()
        
        if ds_edges.empty:
            continue
        
        # Apply cutoff if specified
        if 'cutoff' in params and 'score' in ds_edges.columns:
            cutoff = params['cutoff']
            ds_edges = ds_edges[ds_edges['score'] >= cutoff]
            print(f"   {datasource}: {len(ds_edges):,} edges (cutoff >= {cutoff})")
        else:
            print(f"   {datasource}: {len(ds_edges):,} edges (no cutoff)")
        
        filtered.append(ds_edges)
    
    # Also include edges from datasources not in config
    configured_datasources = set(config['datasources'].keys())
    unconfigured = edges[~edges['datasourceId'].isin(configured_datasources)]
    if not unconfigured.empty:
        print(f"   Other datasources: {len(unconfigured):,} edges")
        filtered.append(unconfigured)
    
    return pd.concat(filtered, ignore_index=True) if filtered else pd.DataFrame()


def aggregate_scores(edges):
    """
    Aggregate scores using harmonic sum.
    
    Groups by (source, target, relation, datasource, year) and aggregates scores.
    
    Args:
        edges: DataFrame with edges
        
    Returns:
        DataFrame with aggregated scores
    """
    if 'score' not in edges.columns:
        return edges
    
    # Group by key columns
    group_cols = ['sourceId', 'targetId', 'source_type', 'target_type', 
                  'relation', 'datasourceId']
    
    # Add year if present
    if 'year' in edges.columns:
        group_cols.append('year')
    
    # Aggregate
    result = edges.groupby(group_cols, as_index=False).agg({
        'score': lambda x: harmonic_sum(x.values)
    })
    
    return result


# ============================================================
# MAIN PROCESSING
# ============================================================

def build_progression_edges(
    input_dir: str,
    config_path: str,
    start_year: int,
    end_year: int,
    output_dir: str,
    include_static: bool = True
):
    """
    Build progression edges with harmonic sum aggregation.
    
    Args:
        input_dir: Directory with raw edges
        config_path: Path to progression config
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
        output_dir: Output directory for progression edges
        include_static: Whether to save static edges separately
    """
    print("\n" + "="*80)
    print("BUILDING PROGRESSION EDGES")
    print("="*80)
    
    # Load config
    print(f"\n📄 Loading config from {config_path}...")
    config = load_config(config_path)
    
    # Load raw edges
    print(f"\n📂 Loading raw edges from {input_dir}...")
    edges = load_all_edges(input_dir)
    
    if edges.empty:
        print("❌ No edges found!")
        return
    
    print(f"✅ Loaded {len(edges):,} total edges")
    
    # Separate dynamic and static
    has_year = 'year' in edges.columns
    
    if has_year:
        dynamic_edges = edges[edges['year'].notna()].copy()
        static_edges = edges[edges['year'].isna()].copy() if include_static else None
        
        print(f"\n📊 Edge breakdown:")
        print(f"   Dynamic (with year): {len(dynamic_edges):,}")
        if include_static:
            print(f"   Static (no year): {len(static_edges):,}")
    else:
        print("⚠️ No 'year' column found, treating all edges as dynamic")
        dynamic_edges = edges.copy()
        static_edges = None
    
    # Filter by year range
    if has_year:
        print(f"\n⏰ Filtering to year range: {start_year} - {end_year}")
        dynamic_edges = dynamic_edges[
            (dynamic_edges['year'] >= start_year) &
            (dynamic_edges['year'] <= end_year)
        ]
        print(f"✅ {len(dynamic_edges):,} edges in range")
    
    # Apply cutoffs
    print(f"\n✂️ Applying datasource cutoffs...")
    dynamic_edges = apply_cutoffs(dynamic_edges, config)
    print(f"✅ {len(dynamic_edges):,} edges after cutoffs")
    
    # Aggregate scores
    print(f"\n🔢 Aggregating scores (harmonic sum)...")
    dynamic_edges = aggregate_scores(dynamic_edges)
    print(f"✅ {len(dynamic_edges):,} edges after aggregation")
    
    # Save per year
    print(f"\n💾 Saving progression edges...")
    
    if has_year:
        years = sorted(dynamic_edges['year'].unique())
        print(f"   Years to save: {years}")
        
        for year in years:
            year_edges = dynamic_edges[dynamic_edges['year'] == year]
            year_dir = f"{output_dir}/{int(year)}"
            Path(year_dir).mkdir(parents=True, exist_ok=True)
            
            # Group by edge type and save
            saved_files = 0
            for (src_type, rel, dst_type, ds), group in year_edges.groupby([
                'source_type', 'relation', 'target_type', 'datasourceId'
            ]):
                filename = f"{src_type}_{rel}_{dst_type}_{ds}.parquet"
                filepath = f"{year_dir}/{filename}"
                group.to_parquet(filepath, index=False)
                saved_files += 1
            
            print(f"   {int(year)}: {len(year_edges):,} edges → {saved_files} files")
    
    # Save static edges
    if include_static and static_edges is not None and not static_edges.empty:
        print(f"\n💾 Saving static edges...")
        static_dir = f"{output_dir}/static"
        Path(static_dir).mkdir(parents=True, exist_ok=True)
        
        # Aggregate static edges
        static_edges = aggregate_scores(static_edges)
        
        saved_files = 0
        for (src_type, rel, dst_type, ds), group in static_edges.groupby([
            'source_type', 'relation', 'target_type', 'datasourceId'
        ]):
            filename = f"{src_type}_{rel}_{dst_type}_{ds}.parquet"
            filepath = f"{static_dir}/{filename}"
            group.to_parquet(filepath, index=False)
            saved_files += 1
        
        print(f"   Static: {len(static_edges):,} edges → {saved_files} files")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ PROGRESSION EDGES BUILD COMPLETE!")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Year range: {start_year} - {end_year}")
    print(f"Total dynamic edges: {len(dynamic_edges):,}")
    if include_static and static_edges is not None:
        print(f"Total static edges: {len(static_edges):,}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build progression edges with harmonic sum aggregation"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with raw edge parquet files"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to progression config YAML"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        required=True,
        help="Start year (inclusive)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        required=True,
        help="End year (inclusive)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/progression_edges",
        help="Output directory for progression edges"
    )
    parser.add_argument(
        "--include-static",
        action="store_true",
        help="Save static edges separately"
    )
    
    args = parser.parse_args()
    
    build_progression_edges(
        input_dir=args.input_dir,
        config_path=args.config,
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output_dir,
        include_static=args.include_static,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build temporal event graph from progression edges.

Outputs single event-based graph with edge_time and edge_weight attributes.
Replaces per-year snapshot approach.
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


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def harmonic_sum(scores, max_harmonic=1.644):
    """Compute harmonic sum of top-50 scores (Open Targets standard)."""
    if len(scores) == 0:
        return 0.0
    s = np.sort(scores)[::-1][:50]
    idx = np.arange(1, len(s) + 1)
    return np.sum(s / (idx ** 2)) / max_harmonic


def load_all_edges(directory):
    """Load all parquet files from directory."""
    dfs = []
    parquet_files = glob(os.path.join(directory, "*.parquet"))
    
    for pq in tqdm(parquet_files, desc="Loading edges"):
        try:
            df = pd.read_parquet(pq)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {pq}: {e}")
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def apply_cutoffs(edges, config):
    """Apply datasource-specific cutoffs."""
    if 'datasources' not in config:
        return edges
    
    filtered = []
    
    for datasource, params in config['datasources'].items():
        ds_edges = edges[edges['datasourceId'] == datasource].copy()
        
        if ds_edges.empty:
            continue
        
        if 'cutoff' in params and 'score' in ds_edges.columns:
            cutoff = params['cutoff']
            ds_edges = ds_edges[ds_edges['score'] >= cutoff]
            print(f"   {datasource}: {len(ds_edges):,} edges (cutoff >= {cutoff})")
        else:
            print(f"   {datasource}: {len(ds_edges):,} edges")
        
        filtered.append(ds_edges)
    
    # Include unconfigured datasources
    configured = set(config['datasources'].keys())
    unconfigured = edges[~edges['datasourceId'].isin(configured)]
    if not unconfigured.empty:
        print(f"   Other datasources: {len(unconfigured):,} edges")
        filtered.append(unconfigured)
    
    return pd.concat(filtered, ignore_index=True) if filtered else pd.DataFrame()


def build_event_list(
    input_dir: str,
    config_path: str,
    output_file: str,
):
    """
    Build temporal event graph from raw edges.
    
    Creates single event list with edge_time and edge_weight.
    
    Args:
        input_dir: Directory with raw edges
        config_path: Path to progression config
        output_file: Output parquet file
    """
    print("\n" + "="*80)
    print("BUILDING TEMPORAL EVENT GRAPH")
    print("="*80)
    
    # Load config
    print(f"\n📄 Loading config from {config_path}...")
    config = load_config(config_path)
    
    # Get time range from config
    if 'time_range' not in config:
        print("❌ No 'time_range' found in config!")
        return
    
    start_year = config['time_range']['first_year']
    end_year = config['time_range']['last_year']
    print(f"✅ Time range from config: {start_year} - {end_year}")
    
    # Load raw edges
    print(f"\n📂 Loading raw edges from {input_dir}...")
    edges = load_all_edges(input_dir)
    
    if edges.empty:
        print("❌ No edges found!")
        return
    
    print(f"✅ Loaded {len(edges):,} total edges")
    
    # Filter to dynamic edges only
    if 'year' not in edges.columns:
        print("❌ No 'year' column found!")
        return
    
    dynamic_edges = edges[edges['year'].notna()].copy()
    print(f"📊 Dynamic edges: {len(dynamic_edges):,}")
    
    # Apply cutoffs
    print(f"\n✂️ Applying datasource cutoffs...")
    dynamic_edges = apply_cutoffs(dynamic_edges, config)
    print(f"✅ {len(dynamic_edges):,} edges after cutoffs")
    
    # Build cumulative temporal events
    print(f"\n🔢 Building cumulative temporal events...")
    print(f"   For each year {start_year}-{end_year}: include all evidences up to that year")
    
    # Group columns (without year)
    group_cols = ['sourceId', 'targetId', 'source_type', 'target_type', 
                  'relation', 'datasourceId']
    
    # Store all year-score combinations
    all_events = []
    
    # For each year, calculate cumulative harmonic sum
    for year in tqdm(range(start_year, end_year + 1), desc="Processing years"):
        # Get all edges up to and including this year (CUMULATIVE)
        cumulative_edges = dynamic_edges[dynamic_edges['year'] <= year].copy()
        
        if cumulative_edges.empty:
            continue
        
        # Group by combination and aggregate scores with harmonic sum
        year_scores = cumulative_edges.groupby(group_cols, as_index=False).agg({
            'score': lambda x: harmonic_sum(x.values)
        })
        
        # Add year column
        year_scores['year'] = year
        
        all_events.append(year_scores)
    
    if not all_events:
        print("❌ No events generated!")
        return
    
    # Combine all years
    events = pd.concat(all_events, ignore_index=True)
    print(f"✅ Generated {len(events):,} year-score combinations")
    
    # Keep only score-change events
    # For each combination, keep first year and years where score changes
    print(f"\n🗜️ Filtering to score-change events only...")
    
    events = events.sort_values(group_cols + ['year'])
    
    # Group and filter
    compressed = []
    for combo, group in events.groupby(group_cols):
        # Keep first event and events where score changed from previous year
        keep_mask = group['score'].diff().fillna(1.0) != 0
        compressed.append(group[keep_mask])
    
    events = pd.concat(compressed, ignore_index=True)
    print(f"✅ {len(events):,} events after filtering (score changes only)")
    
    # Rename for clarity
    events = events.rename(columns={
        'year': 'edge_time',
        'score': 'edge_weight'
    })
    
    # Save
    print(f"\n💾 Saving event graph...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(output_file, index=False)
    
    print(f"✅ Saved to: {output_file}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 EVENT GRAPH SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nTime range: {int(events['edge_time'].min())} - {int(events['edge_time'].max())}")
    print(f"Total events: {len(events):,}")
    print(f"Unique node pairs: {events[['sourceId', 'targetId']].drop_duplicates().shape[0]:,}")
    
    print(f"\n📈 Events per year:")
    year_counts = events.groupby('edge_time').size()
    for year, count in sorted(year_counts.items()):
        print(f"   {int(year)}: {count:,} events")
    
    print(f"\n{'='*80}")
    print(f"✅ EVENT GRAPH BUILD COMPLETE!")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build temporal event graph"
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
        "--output",
        type=str,
        default="output/progression/events.parquet",
        help="Output parquet file"
    )
    
    args = parser.parse_args()
    
    build_event_list(
        input_dir=args.input_dir,
        config_path=args.config,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()

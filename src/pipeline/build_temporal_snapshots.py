#!/usr/bin/env python3
"""
Build snapshot-based temporal graph from progression edges.

Loads pre-processed progression edges and creates cumulative graph snapshots
for each year. Each snapshot contains all edges up to that year.

Input:  output/progression_edges/YYYY/ (per-year parquets)
        output/progression_edges/static/ (optional static edges)
Output: Single .pt file with all snapshots + metadata
"""

import os
import sys
import torch
import pandas as pd
from glob import glob
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.graph_builder import build_hetero_graph


def load_year_edges(progression_dir: str, year: int) -> pd.DataFrame:
    """
    Load all progression edges for a specific year.
    
    Args:
        progression_dir: Base progression edges directory
        year: Year to load
        
    Returns:
        DataFrame with all edges for that year
    """
    year_dir = f"{progression_dir}/{year}"
    if not os.path.exists(year_dir):
        return pd.DataFrame()
    
    dfs = []
    for parquet_file in glob(f"{year_dir}/*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {parquet_file}: {e}")
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_static_edges(progression_dir: str) -> pd.DataFrame:
    """
    Load static edges.
    
    Args:
        progression_dir: Base progression edges directory
        
    Returns:
        DataFrame with static edges
    """
    static_dir = f"{progression_dir}/static"
    if not os.path.exists(static_dir):
        return pd.DataFrame()
    
    dfs = []
    for parquet_file in glob(f"{static_dir}/*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {parquet_file}: {e}")
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def build_temporal_snapshots(
    progression_dir: str,
    include_static: bool = True,
    output_file: str = "temporal_graph.pt",
):
    """
    Build temporal graph with cumulative snapshots from progression edges.
    
    Args:
        progression_dir: Directory with progression edges (YYYY/ subdirs)
        include_static: Whether to include static edges at every timestamp
        output_file: Output file path
    """
    print("\n" + "="*80)
    print("BUILDING TEMPORAL GRAPH FROM PROGRESSION EDGES")
    print("="*80)
    
    # ============================================================
    # 1. Get Available Years
    # ============================================================
    print(f"\n📂 Scanning {progression_dir}...")
    
    years = sorted([
        int(d) for d in os.listdir(progression_dir)
        if d.isdigit() and os.path.isdir(f"{progression_dir}/{d}")
    ])
    
    if not years:
        print(f"❌ No year directories found in {progression_dir}")
        return
    
    print(f"✅ Found {len(years)} years: {years}")
    
    # ============================================================
    # 2. Load Static Edges (Once)
    # ============================================================
    static_edges = None
    if include_static:
        print(f"\n📂 Loading static edges...")
        static_edges = load_static_edges(progression_dir)
        if not static_edges.empty:
            print(f"✅ Loaded {len(static_edges):,} static edges")
        else:
            print(f"⚠️ No static edges found")
            static_edges = None
    
    # ============================================================
    # 3. Build Cumulative Snapshots
    # ============================================================
    graphs = {}
    metadata = {}
    cumulative_edges = []
    
    print(f"\n🔨 Building cumulative snapshots...")
    
    for year in years:
        print(f"\n{'='*80}")
        print(f"📅 Year {year}")
        print(f"{'='*80}")
        
        # Load this year's edges
        year_edges = load_year_edges(progression_dir, year)
        
        if year_edges.empty:
            print(f"⚠️ No edges for {year}, skipping")
            continue
        
        print(f"   New edges: {len(year_edges):,}")
        cumulative_edges.append(year_edges)
        
        # Combine all edges up to this year (CUMULATIVE)
        all_edges_up_to_year = pd.concat(cumulative_edges, ignore_index=True)
        print(f"   Cumulative edges: {len(all_edges_up_to_year):,}")
        
        # Add static edges
        if include_static and static_edges is not None:
            all_edges = pd.concat([all_edges_up_to_year, static_edges], ignore_index=True)
            print(f"   + Static edges: {len(static_edges):,}")
            print(f"   Total edges: {len(all_edges):,}")
        else:
            all_edges = all_edges_up_to_year
        
        # Build graph
        print(f"\n   🔨 Building HeteroData...")
        hetero_data, id_maps = build_hetero_graph(all_edges)
        graphs[year] = hetero_data
        
        # Collect metadata
        edge_type_counts = {}
        total_edges = 0
        
        for edge_type in hetero_data.edge_types:
            num_edges = hetero_data[edge_type].edge_index.size(1)
            edge_type_counts[str(edge_type)] = num_edges
            total_edges += num_edges
        
        metadata[year] = {
            'num_edge_types': len(hetero_data.edge_types),
            'edge_types': edge_type_counts,
            'total_edges': total_edges,
            'num_nodes': sum(data.num_nodes for data in hetero_data.node_stores),
            'node_types': {
                node_type: hetero_data[node_type].num_nodes
                for node_type in hetero_data.node_types
            }
        }
        
        print(f"   ✅ Snapshot complete:")
        print(f"      Nodes: {metadata[year]['num_nodes']:,}")
        print(f"      Edges: {metadata[year]['total_edges']:,}")
        print(f"      Edge types: {metadata[year]['num_edge_types']}")
    
    # ============================================================
    # 4. Save Temporal Graph
    # ============================================================
    print(f"\n{'='*80}")
    print(f"💾 Saving temporal graph...")
    print(f"{'='*80}")
    
    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    torch.save({
        'timestamps': years,
        'graphs': graphs,
        'metadata': metadata,
        'config': {
            'include_static': include_static,
            'mode': 'snapshot',
            'progression_dir': progression_dir,
        }
    }, output_file)
    
    print(f"✅ Saved to: {output_file}")
    
    # ============================================================
    # 5. Summary
    # ============================================================
    print(f"\n{'='*80}")
    print(f"📊 TEMPORAL GRAPH SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nYears: {years}")
    print(f"Snapshots: {len(graphs)}")
    print(f"Include static: {include_static}")
    
    print(f"\n📈 Growth Over Time:")
    print(f"{'Year':<8} {'Nodes':<10} {'Edges':<12} {'Edge Types':<12}")
    print(f"{'-'*80}")
    
    for year in years:
        meta = metadata[year]
        print(f"{year:<8} {meta['num_nodes']:<10,} "
              f"{meta['total_edges']:<12,} {meta['num_edge_types']:<12}")
    
    print(f"\n{'='*80}")
    print(f"✅ TEMPORAL GRAPH BUILD COMPLETE!")
    print(f"{'='*80}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build temporal graph from progression edges"
    )
    parser.add_argument(
        "--progression-dir",
        type=str,
        required=True,
        help="Directory with progression edges (contains YYYY/ subdirs)"
    )
    parser.add_argument(
        "--include-static",
        action="store_true",
        help="Include static edges at every timestamp"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/progression/temporal_graph.pt",
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    build_temporal_snapshots(
        progression_dir=args.progression_dir,
        include_static=args.include_static,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build event-based HeteroData graph from progression events.

Loads single event list (from build_event_list.py), builds HeteroData 
with edge_time and edge_weight, and saves to .pt file.
"""

import os
import sys
import argparse
import pandas as pd
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.graph_builder import build_hetero_graph


def build_event_graph(
    event_file: str,
    output_file: str
):
    """
    Build HeteroData from event list.
    
    Args:
        event_file: Path to events parquet file
        output_file: Output .pt file
    """
    print("\n" + "="*80)
    print("BUILDING EVENT-BASED TEMPORAL GRAPH")
    print("="*80)
    
    # Load events
    print(f"\n📂 Loading events from {event_file}...")
    if not os.path.exists(event_file):
        print(f"❌ Event file not found: {event_file}")
        return
        
    events = pd.read_parquet(event_file)
    print(f"✅ Loaded {len(events):,} events")
    
    # Check columns
    required = ['sourceId', 'targetId', 'source_type', 'target_type', 
                'relation', 'datasourceId', 'edge_time', 'edge_weight']
    
    missing = [c for c in required if c not in events.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        return
    
    # Build graph
    # build_hetero_graph now supports edge_time and edge_weight
    hetero_data, id_maps = build_hetero_graph(events)
    
    # Save
    print(f"\n💾 Saving event graph to {output_file}...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(hetero_data, output_file)
    print(f"✅ Saved HeteroData object")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"✅ EVENT GRAPH COMPLETE")
    print(f"{'='*80}")
    print(f"Nodes:")
    for nt in hetero_data.node_types:
        print(f"   {nt}: {hetero_data[nt].num_nodes:,}")
        
    print(f"\nEdges:")
    for et in hetero_data.edge_types:
        print(f"   {et}: {hetero_data[et].edge_index.size(1):,}")
        
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Build event-based temporal graph")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to events parquet file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/progression/temporal_graph.pt",
        help="Output .pt file"
    )
    
    args = parser.parse_args()
    
    build_event_graph(
        event_file=args.input,
        output_file=args.output
    )


if __name__ == "__main__":
    main()

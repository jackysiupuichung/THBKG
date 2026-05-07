#!/usr/bin/env python3
"""
Build event-based HeteroData graph from progression events.

Loads single event list (from build_event_list.py), builds HeteroData 
with edge_time and edge_weight, and saves to .pt file.
"""

# does the static edges need to be confined to the nodes that have been included in the dynamic edges?
# this depends on the ratio between static and dynamic edges

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple
from glob import glob

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




def load_edges(edge_dir: str, cutoff_year: int = None) -> pd.DataFrame:
    """
    Load all edge parquet files from directory.
    
    Args:
        edge_dir: Directory containing edge parquet files
        cutoff_year: Optional year cutoff (only include edges <= cutoff_year)
        
    Returns:
        DataFrame with all edges
    """
    dfs = []
    
    for parquet_file in glob(os.path.join(edge_dir, "*.parquet")):
        df = pd.read_parquet(parquet_file)
        
        if df.empty:
            continue
        
        # Filter by cutoff year if specified
        if cutoff_year is not None and "year" in df.columns:
            df = df[df["year"] <= cutoff_year]
        
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def load_static_edges(static_dir: str) -> pd.DataFrame:
    """
    Load static edges from directory.
    expected format: sourceId, targetId, source_type, target_type, relation, datasourceId
    
    Args:
        static_dir: Directory containing static edge parquet files
        
    Returns:
        DataFrame with static edges (with default score=1.0 if missing)
    """
    if not static_dir or not os.path.exists(static_dir):
        return pd.DataFrame()
        
    print(f"\n📂 Loading static edges from {static_dir}...")
    dfs = []
    
    for parquet_file in glob(os.path.join(static_dir, "*.parquet")):
        try:
            df = pd.read_parquet(parquet_file)
            if df.empty: continue
            
            # Ensure required columns
            required = ['sourceId', 'targetId', 'source_type', 'target_type', 'relation', 'datasourceId']
            if not all(c in df.columns for c in required):
                print(f"⚠️ Skipping {Path(parquet_file).name}: Missing required columns")
                continue
                
            # Add default score if missing
            if 'score' not in df.columns:
                df['score'] = 1.0

            # Static edges have no novelty signal — set to 1.0 (fully novel / always present)
            df['edge_novelty'] = 1.0
                
            # Ensure no temporal columns interfere (force them to be null or handled)
            # Static edges have NO edge_time
            if 'edge_time' in df.columns:
                df = df.drop(columns=['edge_time'])
                
            dfs.append(df)
            print(f"   Loaded {len(df):,} edges from {Path(parquet_file).name}")
            
        except Exception as e:
            print(f"❌ Error loading {parquet_file}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    combined = pd.concat(dfs, ignore_index=True)
    print(f"✅ Total static edges: {len(combined):,}")
    return combined


def extract_nodes_from_edges(edges: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Extract unique nodes from edges.
    
    Args:
        edges: DataFrame with edges containing sourceId, targetId, source_type, target_type
        
    Returns:
        nodes: Dictionary mapping node type to list of node IDs
        id_to_type: Dictionary mapping node ID to node type
    """
    nodes = {}
    id_to_type = {}
    
    # Extract source nodes
    for _, row in edges[['sourceId', 'source_type']].drop_duplicates().iterrows():
        node_id = str(row['sourceId'])
        node_type = row['source_type']
        
        if node_type not in nodes:
            nodes[node_type] = []
        
        if node_id not in id_to_type:
            nodes[node_type].append(node_id)
            id_to_type[node_id] = node_type
    
    # Extract target nodes
    for _, row in edges[['targetId', 'target_type']].drop_duplicates().iterrows():
        node_id = str(row['targetId'])
        node_type = row['target_type']
        
        if node_type not in nodes:
            nodes[node_type] = []
        
        if node_id not in id_to_type:
            nodes[node_type].append(node_id)
            id_to_type[node_id] = node_type
    
    # Remove duplicates and sort
    for node_type in nodes:
        nodes[node_type] = sorted(list(set(nodes[node_type])))
    
    return nodes, id_to_type



def build_hetero_graph(edges: pd.DataFrame, edge_type_mode: str = 'relation_datasource') -> Tuple[HeteroData, Dict]:
    """
    Build heterogeneous graph from edges.
    
    Supports two edge type modes:
    - 'relation_datasource': (source_type, "relation::datasource", target_type)
    - 'relation_only': (source_type, "relation", target_type)
    
    Supports temporal attributes: edge_time and edge_weight.
    
    Args:
        edges: DataFrame with edges (sourceId, targetId, source_type, target_type, 
               relation, datasourceId, score)
               Optional: edge_time (year/timestamp), edge_weight (for events)
        edge_type_mode: 'relation_datasource' or 'relation_only'
        
    Returns:
        hetero_data: HeteroData object
        mappings: Dictionary containing:
            - node_mapping: {node_type: {node_id_str: index}}
            - node_type_mapping: {node_type: type_index}
            - edge_type_mapping: {edge_type_tuple: type_index}
    """
    mode_label = "relation::datasource" if edge_type_mode == 'relation_datasource' else "relation only"
    print(f"\n🔨 Building HeteroData ({mode_label} level)...")
    
    # Extract nodes
    print("📊 Extracting nodes from edges...")
    nodes, id_to_type = extract_nodes_from_edges(edges)
    
    # Create ID mappings
    node_mapping = {}
    node_type_mapping = {nt: i for i, nt in enumerate(sorted(nodes.keys()))}
    
    for node_type, node_list in nodes.items():
        node_mapping[node_type] = {node_id: idx for idx, node_id in enumerate(node_list)}
        print(f"   {node_type}: {len(node_list)} nodes")
    
    # Build HeteroData
    hetero_data = HeteroData()
    
    # Add nodes
    print("\n🔗 Adding nodes...")
    for node_type, node_list in nodes.items():
        hetero_data[node_type].num_nodes = len(node_list)
        # Store original IDs to allow feature mapping later
        hetero_data[node_type].node_id = node_list
        print(f"✅ Added {len(node_list)} {node_type} nodes")
    
    # Build edges
    mode_label = "relation::datasource" if edge_type_mode == 'relation_datasource' else "relation only"
    print(f"\n🔗 Building edges ({mode_label} level)...")
    
    # Check for temporal attributes
    has_edge_time = 'edge_time' in edges.columns
    has_edge_weight = 'edge_weight' in edges.columns
    has_score = 'score' in edges.columns and not has_edge_weight
    # edge_novelty is always expected alongside edge_weight
    has_edge_novelty = 'edge_novelty' in edges.columns
    
    # Group by edge type based on mode
    if edge_type_mode == 'relation_datasource':
        # Group by relation AND datasource
        edge_groups = edges.groupby(['source_type', 'relation', 'target_type', 'datasourceId'])
        group_keys = ['source_type', 'relation', 'target_type', 'datasourceId']
    else:
        # Group by relation ONLY (aggregate across datasources)
        edge_groups = edges.groupby(['source_type', 'relation', 'target_type'])
        group_keys = ['source_type', 'relation', 'target_type']
    
    edge_type_mapping = {}
    edge_type_idx = 0
    
    for group_tuple, group in edge_groups:
        # Unpack based on mode
        if edge_type_mode == 'relation_datasource':
            src_type, relation, dst_type, datasource = group_tuple
            edge_type_key = (src_type, f"{relation}::{datasource}", dst_type)
        else:
            src_type, relation, dst_type = group_tuple
            edge_type_key = (src_type, relation, dst_type)
        
        # Add to mapping if new
        if edge_type_key not in edge_type_mapping:
            edge_type_mapping[edge_type_key] = edge_type_idx
            edge_type_idx += 1
        
        # Map node IDs to indices
        src_indices = [node_mapping[src_type][str(sid)] for sid in group['sourceId']]
        dst_indices = [node_mapping[dst_type][str(tid)] for tid in group['targetId']]
        
        # Create edge_index
        edge_index = torch.tensor(
            [src_indices, dst_indices],
            dtype=torch.long
        )
        
        hetero_data[edge_type_key].edge_index = edge_index
        
        # Add edge attributes — always [E, 2]: [edge_weight, edge_novelty]
        if has_edge_weight:
            w = torch.tensor(group['edge_weight'].fillna(0.0).values, dtype=torch.float).unsqueeze(-1)
            n = torch.tensor(
                group['edge_novelty'].fillna(0.0).values if has_edge_novelty else [0.0] * len(group),
                dtype=torch.float
            ).unsqueeze(-1)
            hetero_data[edge_type_key].edge_attr = torch.cat([w, n], dim=-1)
        elif has_score:
            # Snapshot-based: use score; novelty not available → 0
            w = torch.tensor(group['score'].fillna(0.0).values, dtype=torch.float).unsqueeze(-1)
            n = torch.zeros(len(group), 1, dtype=torch.float)
            hetero_data[edge_type_key].edge_attr = torch.cat([w, n], dim=-1)
        else:
            # Structural edges (ontology hierarchies etc.) — no score or weight available
            # Use 1.0, 1.0: fully confident, fully novel
            hetero_data[edge_type_key].edge_attr = torch.ones(len(group), 2, dtype=torch.float)

        # Add temporal attribute
        if has_edge_time:
            hetero_data[edge_type_key].edge_time = torch.tensor(
                group['edge_time'].values,
                dtype=torch.long  # Use long for temporal sampling compatibility
            )
        
        num_edges = edge_index.size(1)
        attrs_str = []
        if has_edge_weight or has_score:
            edge_attr_shape = tuple(hetero_data[edge_type_key].edge_attr.shape)
            attrs_str.append(f"edge_attr{list(edge_attr_shape)}")
        if has_edge_time:
            attrs_str.append("edge_time")
        
        attr_info = f" ({', '.join(attrs_str)})" if attrs_str else ""
        print(f"✅ Added {num_edges} edges for {edge_type_key}{attr_info}")
    
    mappings = {
        "node_mapping": node_mapping,
        "node_type_mapping": node_type_mapping,
        "edge_type_mapping": edge_type_mapping
    }
    
    return hetero_data, mappings

def load_advancement_edges(train_csv: str, test_csv: str) -> pd.DataFrame:
    """
    Load clinical trial advancement edges from CSV files and format them
    as graph edges compatible with build_hetero_graph.

    Args:
        train_csv: Path to train_dataset.csv
        test_csv: Path to test_dataset.csv

    Returns:
        DataFrame with columns: sourceId, targetId, source_type, target_type,
        relation, datasourceId, edge_time, edge_weight
    """
    print(f"\n📂 Loading advancement edges...")
    dfs = []
    for path, label in [(train_csv, "train"), (test_csv, "test")]:
        if not os.path.exists(path):
            print(f"   ⚠️  {label} CSV not found: {path}")
            continue
        df = pd.read_csv(path)
        print(f"   Loaded {len(df):,} rows from {label} ({Path(path).name})")
        dfs.append(df)

    if not dfs:
        print("   ❌ No advancement CSVs loaded")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    print(f"   Total advancement rows: {len(df):,}")

    adv = pd.DataFrame({
        "sourceId":     df["target_id"].astype(str),
        "targetId":     df["disease_id"].astype(str),
        "source_type":  "target",
        "target_type":  "disease",
        "relation":     "advancement",
        "datasourceId": "advancement",
        "edge_time":    df["transition_year"].astype(int),
        "edge_weight":  df["outcome"].astype(float),
        "edge_novelty": 1.0,  # no novelty decay applies to advancement labels
    })
    print(f"✅ Advancement edges ready: {len(adv):,}")
    return adv


def build_literature_sidecar(raw_edges_dir: str, output_path: str):
    """
    Build a sidecar parquet mapping (sourceId, targetId, datasourceId, year) -> PMIDs + OT evidence IDs.

    Reads raw edge parquets (which retain the 'literature' and 'id' columns from kg_pipeline),
    explodes the PMID list so each row is one (edge_key, pmid) pair, and writes the result
    alongside the graph file for later lookup via get_edge_literature().

    Args:
        raw_edges_dir: Directory containing raw edge parquets (output of kg_pipeline)
        output_path: Destination parquet path for the sidecar table
    """
    print(f"\n📚 Building literature sidecar from {raw_edges_dir}...")

    dfs = []
    for pf in glob(os.path.join(raw_edges_dir, "*.parquet")):
        df = pd.read_parquet(pf)
        keep = [c for c in ['sourceId', 'targetId', 'datasourceId', 'year', 'literature', 'id'] if c in df.columns]
        if 'literature' not in keep:
            continue
        dfs.append(df[keep])

    if not dfs:
        print("⚠️  No raw edge parquets with 'literature' column found — skipping sidecar")
        return

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined['literature'].notna()]

    # Normalise: literature may be a list, ndarray, or a single scalar
    def _to_list(v):
        if isinstance(v, (list, tuple)):
            return list(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return [v]
    combined['literature'] = combined['literature'].apply(_to_list)

    exploded = combined.explode('literature').rename(columns={'literature': 'pmid', 'id': 'evidence_id'})
    exploded['pmid'] = exploded['pmid'].astype(str)
    exploded = exploded[exploded['pmid'].notna() & (exploded['pmid'] != 'nan')].drop_duplicates()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    exploded.to_parquet(output_path, index=False)
    print(f"✅ Sidecar saved: {output_path}  ({len(exploded):,} rows, {exploded['pmid'].nunique():,} unique PMIDs)")


def build_event_graph(
    event_file: str,
    output_file: str,
    static_edges_dir: str = None,
    edge_type_mode: str = 'relation_datasource',
    advancement_train_csv: str = None,
    advancement_test_csv: str = None,
    raw_edges_dir: str = None,
):
    """
    Build HeteroData from event list.

    Args:
        event_file: Path to events parquet file
        output_file: Output .pt file
        static_edges_dir: Optional directory with static edges
        advancement_train_csv: Optional path to advancement train CSV
        advancement_test_csv: Optional path to advancement test CSV
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
    
    # Load advancement edges (injected before static edges so their nodes are registered)
    if advancement_train_csv or advancement_test_csv:
        adv_edges = load_advancement_edges(
            advancement_train_csv or "",
            advancement_test_csv or "",
        )
        if not adv_edges.empty:
            events = pd.concat([events, adv_edges], ignore_index=True)
            print(f"   Events + advancement: {len(events):,} total rows")

    # Load static edges
    static_edges = pd.DataFrame()
    if static_edges_dir:
        static_edges = load_static_edges(static_edges_dir)

    # Combine
    # Note: Static edges have NaNs for edge_time and edge_weight (unless score mapped check)
    # We should normalize columns before concat

    all_edges = events
    
    if not static_edges.empty:
        print("\n➕ Merging static edges (no endpoint filtering — static nodes may introduce new nodes)...")

        event_node_ids = set()
        event_node_ids.update(events['sourceId'].astype(str).unique())
        event_node_ids.update(events['targetId'].astype(str).unique())

        static_src = set(static_edges['sourceId'].astype(str).unique())
        static_tgt = set(static_edges['targetId'].astype(str).unique())
        new_nodes = (static_src | static_tgt) - event_node_ids

        print(f"   Unique nodes in events: {len(event_node_ids):,}")
        print(f"   Static edges: {len(static_edges):,}")
        print(f"   New nodes introduced by static edges: {len(new_nodes):,}")

        all_edges = pd.concat([events, static_edges], ignore_index=True)
        print(f"   Combined Total: {len(all_edges):,} edges")
    
    # Build literature sidecar (before graph build; raw edges dir must be provided)
    if raw_edges_dir:
        sidecar_path = output_file.replace(".pt", "_literature.parquet")
        build_literature_sidecar(raw_edges_dir, sidecar_path)

    # Build graph
    # build_hetero_graph now supports edge_time and edge_weight
    hetero_data, mappings = build_hetero_graph(all_edges, edge_type_mode=edge_type_mode)

    # Save
    print(f"\n💾 Saving event graph to {output_file}...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(hetero_data, output_file)
    print(f"✅ Saved HeteroData object")
    
    # Save mappings
    mapping_file = output_file.replace(".pt", "_mappings.pt")
    print(f"💾 Saving mappings to {mapping_file}...")
    torch.save(mappings, mapping_file)
    print(f"✅ Saved mappings object")
    
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
    parser.add_argument(
        "--static-edges",
        type=str,
        default=None,
        help="Directory containing static edge parquets"
    )
    parser.add_argument(
        "--edge-type-mode",
        type=str,
        default="relation_datasource",
        choices=["relation_datasource", "relation_only"],
        help="Edge type naming: 'relation_datasource' (e.g., clinical_trial::chembl) or 'relation_only' (e.g., clinical_trial)"
    )
    parser.add_argument(
        "--advancement-train-csv",
        type=str,
        default=None,
        help="Path to advancement train CSV (optional)"
    )
    parser.add_argument(
        "--advancement-test-csv",
        type=str,
        default=None,
        help="Path to advancement test CSV (optional)"
    )
    parser.add_argument(
        "--raw-edges",
        type=str,
        default=None,
        help="Directory of raw edge parquets (from kg_pipeline) used to build literature sidecar (optional)"
    )

    args = parser.parse_args()

    # Build graph
    build_event_graph(
        event_file=args.input,
        output_file=args.output,
        static_edges_dir=args.static_edges,
        edge_type_mode=args.edge_type_mode,
        advancement_train_csv=args.advancement_train_csv,
        advancement_test_csv=args.advancement_test_csv,
        raw_edges_dir=args.raw_edges,
    )


if __name__ == "__main__":
    main()

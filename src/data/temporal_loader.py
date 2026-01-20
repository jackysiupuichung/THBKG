#!/usr/bin/env python3
"""
Temporal graph loader utilities.

Load pre-built temporal graph snapshots by year for efficient testing.
"""

import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional
from pathlib import Path


def load_snapshot(
    filepath: str,
    year: int,
    attach_features: bool = False,
    embedding_dim: int = 128,
    seed: int = 42
) -> HeteroData:
    """
    Load a specific year snapshot from temporal graph.
    
    Args:
        filepath: Path to temporal graph file
        year: Year to load
        attach_features: Whether to attach random node features
        embedding_dim: Embedding dimension
        seed: Random seed
        
    Returns:
        HeteroData for specified year
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Temporal graph file not found: {filepath}")
    
    data = torch.load(filepath, weights_only=False)
    
    if year not in data['graphs']:
        available = data['timestamps']
        raise ValueError(
            f"Year {year} not found in temporal graph. "
            f"Available years: {available}"
        )
    
    hetero_data = data['graphs'][year]
    
    # Optionally attach features
    if attach_features:
        from .utils import attach_node_features
        # Create dummy id_maps (nodes already indexed in graph)
        id_maps = {}
        for node_type in hetero_data.node_types:
            num_nodes = hetero_data[node_type].num_nodes
            id_maps[node_type] = {str(i): i for i in range(num_nodes)}
        
        hetero_data = attach_node_features(
            hetero_data,
            id_maps,
            init_method="random",
            embedding_dim=embedding_dim,
            seed=seed
        )
    
    return hetero_data


def load_temporal_graph(filepath: str) -> Dict:
    """
    Load entire temporal graph with all snapshots.
    
    Args:
        filepath: Path to temporal graph file
        
    Returns:
        Dictionary with 'timestamps', 'graphs', 'metadata', 'config'
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Temporal graph file not found: {filepath}")
    
    return torch.load(filepath, weights_only=False)


def get_metadata(filepath: str, year: Optional[int] = None) -> Dict:
    """
    Get metadata for temporal graph.
    
    Args:
        filepath: Path to temporal graph file
        year: Optional specific year (if None, returns all metadata)
        
    Returns:
        Metadata dict or dict of metadata per year
    """
    data = torch.load(filepath, weights_only=False)
    
    if year is not None:
        if year not in data['metadata']:
            available = list(data['metadata'].keys())
            raise ValueError(
                f"Year {year} not found. Available: {available}"
            )
        return data['metadata'][year]
    else:
        return data['metadata']


def list_available_years(filepath: str) -> List[int]:
    """
    List available years in temporal graph.
    
    Args:
        filepath: Path to temporal graph file
        
    Returns:
        List of available years
    """
    data = torch.load(filepath, weights_only=False)
    return data['timestamps']


def print_temporal_summary(filepath: str):
    """
    Print summary of temporal graph.
    
    Args:
        filepath: Path to temporal graph file
    """
    data = torch.load(filepath, weights_only=False)
    
    print(f"\n📊 Temporal Graph Summary")
    print(f"{'='*80}")
    print(f"File: {filepath}")
    print(f"Years: {data['timestamps']}")
    print(f"Config: {data['config']}")
    
    print(f"\n📈 Growth Over Time:")
    print(f"{'Year':<8} {'Nodes':<10} {'Edges':<12} {'Edge Types':<12}")
    print(f"{'-'*80}")
    
    for year in data['timestamps']:
        meta = data['metadata'][year]
        print(f"{year:<8} {meta['num_nodes']:<10,} "
              f"{meta['total_edges']:<12,} {meta['num_edge_types']:<12}")

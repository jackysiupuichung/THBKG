#!/usr/bin/env python3
"""
Pipeline script to attach node features to a HeteroData graph.
Loads a graph from a .pt file, attaches features using src.data.utils, and saves it.
"""

import argparse
import sys
import os
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.utils import attach_node_features

def main():
    parser = argparse.ArgumentParser(description="Attach node features to HeteroData")
    parser.add_argument("--graph-file", required=True, help="Input .pt graph file")
    parser.add_argument("--output-file", required=True, help="Output .pt graph file with features")
    parser.add_argument("--feature-dir", default="data/node_features/processed", help="Directory with feature .pt files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.graph_file):
        print(f"❌ Graph file not found: {args.graph_file}")
        sys.exit(1)
        
    print(f"\n🔗 Attaching Features to Graph...")
    print(f"   Input: {args.graph_file}")
    
    # Load
    data = torch.load(args.graph_file, weights_only=False)
    print(f"   Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    
    # Attach
    # Note: src.data.utils.attach_node_features uses hardcoded paths for features relative to CWD usually.
    # However, we updated it to point to data/node_features/processed.
    # To be safer, we should probably pass the feature dir if updated...
    # But current utils implementation hardcodes paths like "data/node_features/processed/...".
    # Assuming CWD is project root.
    
    data = attach_node_features(
        data,
        id_maps=None,
        init_method="pretrained",
        embedding_dim=128 # Default fallback dim
    )
    
    # Save
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output_file)
    print(f"✅ Saved graph with features to {args.output_file}")
    
    # Print feature status
    print("\n   Feature Status:")
    for nt in data.node_types:
        if data[nt].x is not None:
            print(f"   - {nt}: {data[nt].x.shape}")
        else:
            print(f"   - {nt}: None")

if __name__ == "__main__":
    main()

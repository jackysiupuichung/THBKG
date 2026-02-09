import torch
import os
from pathlib import Path

def extract_mappings(graph_path, output_path):
    print(f"Loading graph from {graph_path}...")
    data = torch.load(graph_path, weights_only=False)
    
    mappings = {'node_mapping': {}}
    
    for node_type in ['disease', 'target']:
        if hasattr(data[node_type], 'node_id'):
            print(f"Extracting {node_type} IDs...")
            node_ids = data[node_type].node_id
            # Create ID -> Index mapping
            id_map = {nid: idx for idx, nid in enumerate(node_ids)}
            mappings['node_mapping'][node_type] = id_map
            print(f"  {len(id_map)} {node_type}s mapped.")
        else:
            print(f"WARNING: No node_id found for {node_type}")
            
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mappings, output_path)
    print(f"Saved mappings to {output_path}")

if __name__ == "__main__":
    extract_mappings(
        'output/graph/hetero_graph_with_features.pt',
        'output/graph/hetero_mappings.pt'
    )

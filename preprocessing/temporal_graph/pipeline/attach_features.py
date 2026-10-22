#!/usr/bin/env python3
"""
Attach node features to a HeteroData graph.
Loads features from processed directory and attaches them to graph nodes.
"""

import argparse
import sys
import torch
from pathlib import Path


def load_feature_embeddings(feature_dir: Path, node_type: str):
    """
    Load feature embeddings for a specific node type.
    
    Args:
        feature_dir: Directory containing processed feature files
        node_type: Node type (disease, target, go, reactome, molecule)
    
    Returns:
        dict: {node_id: tensor} mapping, or None if not found
    """
    # Map node types to their feature files
    feature_files = {
        'disease': 'disease_embeddings.pt',
        'target': 'integrated_target_features.pt',
        'go': 'go_embeddings.pt',
        'reactome': 'reactome_embeddings.pt',
        'molecule': 'molecule_morgan_fingerprints.pt'
    }
    
    if node_type not in feature_files:
        print(f"   ⚠️  Unknown node type: {node_type}")
        return None

    feature_path = feature_dir / feature_files[node_type]
    
    if not feature_path.exists():
        print(f"   ⚠️  Feature file not found: {feature_path}")
        return None
    
    print(f"   Loading {node_type} features from {feature_path.name}")
    embeddings = torch.load(feature_path, weights_only=False)
    print(f"      Loaded {len(embeddings):,} embeddings")
    
    return embeddings


def attach_modalities(store, path: Path) -> None:
    """Attach per-modality tensors (molecule_multimodal.py) to a node store, by node_id.

    Sets x_<modality>, has_<modality> and drug_type, row-aligned with the store's
    node_id list; no ``x`` (models project each modality separately, see
    src/models/node_input.py). Every graph node must be covered.
    """
    mod = torch.load(path, weights_only=False)
    row = {nid: i for i, nid in enumerate(mod["node_id"])}
    missing = [nid for nid in store.node_id if nid not in row]
    if missing:
        raise ValueError(f"{len(missing)} graph nodes missing from {path.name}, e.g. {missing[:3]}")
    idx = torch.tensor([row[nid] for nid in store.node_id], dtype=torch.long)
    # explicit: without ``x`` PyG would infer the count from edge indices, and
    # samplers would then not slice the modality tensors as node attributes
    store.num_nodes = len(store.node_id)
    for key, val in mod.items():
        if torch.is_tensor(val):
            store[key] = val[idx]
    if "x" in store:
        del store.x
    shapes = ", ".join(f"{k} {list(v.shape)}" for k, v in mod.items() if torch.is_tensor(v))
    print(f"      ✅ Attached modalities: {shapes}")


def attach_features_to_graph(data, feature_dir: Path):
    """
    Attach node features to HeteroData graph.
    
    Args:
        data: HeteroData graph
        feature_dir: Directory with processed feature .pt files
    
    Returns:
        HeteroData: Graph with features attached
    """
    print(f"\n📦 Attaching features from {feature_dir}")
    
    for node_type in data.node_types:
        print(f"\n   Processing {node_type}...")

        # Per-modality molecule features (molecule_multimodal.py) supersede the
        # Morgan-only matrix when present
        if node_type == 'molecule' and (feature_dir / 'molecule_modalities.pt').exists():
            attach_modalities(data[node_type], feature_dir / 'molecule_modalities.pt')
            continue

        # Load embeddings
        embeddings = load_feature_embeddings(feature_dir, node_type)
        
        # Fatal rather than x=None / zero rows: a silent gap here trains models
        # on a node type without features
        if embeddings is None:
            raise FileNotFoundError(f"no feature file for node type {node_type!r} in {feature_dir}")
        if not hasattr(data[node_type], 'node_id'):
            raise ValueError(f"{node_type!r} has no node_id to align features with")
        
        node_ids = data[node_type].node_id
        num_nodes = len(node_ids)
        
        # Build feature matrix
        missing = [nid for nid in node_ids if nid not in embeddings]
        if missing:
            raise ValueError(f"{len(missing):,} of {num_nodes:,} {node_type} nodes have no features "
                             f"(e.g. {missing[:3]}); rebuild the feature file")
        feature_list = [embeddings[nid] for nid in node_ids]
        
        # Stack into tensor
        feature_matrix = torch.stack(feature_list)
        data[node_type].x = feature_matrix
        
        print(f"      ✅ Attached features: {feature_matrix.shape}")
    
    return data


def main():
    parser = argparse.ArgumentParser(description="Attach node features to HeteroData graph")
    parser.add_argument("--graph-file", required=True, help="Input .pt graph file")
    parser.add_argument("--output-file", required=True, help="Output .pt graph file with features")
    parser.add_argument("--feature-dir", required=True, help="Directory with processed feature .pt files")
    
    args = parser.parse_args()
    
    # Validate inputs
    graph_path = Path(args.graph_file)
    feature_dir = Path(args.feature_dir)
    output_path = Path(args.output_file)
    
    if not graph_path.exists():
        print(f"❌ Graph file not found: {graph_path}")
        sys.exit(1)
    
    if not feature_dir.exists():
        print(f"❌ Feature directory not found: {feature_dir}")
        sys.exit(1)
    
    print(f"\n🔗 Attaching Features to Graph")
    print(f"{'='*60}")
    print(f"Graph file: {graph_path}")
    print(f"Feature directory: {feature_dir}")
    print(f"Output file: {output_path}")
    
    # Load graph
    print(f"\n📊 Loading graph...")
    data = torch.load(graph_path, weights_only=False)
    print(f"   Node types: {data.node_types}")
    print(f"   Edge types: {data.edge_types}")
    print(f"   Total nodes: {data.num_nodes:,}")
    print(f"   Total edges: {data.num_edges:,}")
    
    # Attach features
    data = attach_features_to_graph(data, feature_dir)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    print(f"\n✅ Saved graph with features to {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("FEATURE SUMMARY")
    print(f"{'='*60}")
    for node_type in data.node_types:
        store = data[node_type]
        if 'x' in store and store.x is not None:
            print(f"   {node_type:15s}: {str(store.x.shape):20s} ✅")
        elif any(k.startswith('x_') for k in store.keys()):
            print(f"   {node_type:15s}: " + ", ".join(f"{k} {list(store[k].shape)}" for k in store.keys()
                                                     if k.startswith('x_')) + " ✅")
        else:
            print(f"   {node_type:15s}: {'None':20s} ❌")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

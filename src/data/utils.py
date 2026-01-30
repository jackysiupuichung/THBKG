#!/usr/bin/env python3
"""
Data utilities for node feature attachment.
"""

import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, Optional


def load_integrated_target_features(
    target_ids: list[str],
    feature_dir: str = "data/node_features/processed"
) -> torch.Tensor:
    """
    Load pre-integrated features for targets.
    
    Args:
        target_ids: List of target IDs (ENSG...) to align with.
        feature_dir: Directory containing .pt feature files.
        
    Returns:
        Tensor of shape (num_targets, combined_dim)
    """
    import os
    print(f"\n🧬 Loading Integrated Target Features...")
    
    path = os.path.join(feature_dir, "integrated_target_features.pt")
    if not os.path.exists(path):
        print(f"   ⚠️ Integrated features not found at {path}. Returning random.")
        return torch.randn(len(target_ids), 128)
        
    print(f"   Loading {path}...")
    features_dict = torch.load(path, weights_only=False)
    
    # Determine dimension from first item
    if not features_dict:
        print("   ⚠️ Feature dictionary is empty. Returning random.")
        return torch.randn(len(target_ids), 128)
        
    dim = next(iter(features_dict.values())).shape[0]
    print(f"   Feature dimension: {dim}")
    
    # Align
    aligned_features = []
    missing_count = 0
    zero_vec = torch.zeros(dim)
    
    for tid in target_ids:
        if tid in features_dict:
            aligned_features.append(features_dict[tid])
        else:
            aligned_features.append(zero_vec)
            missing_count += 1
            
    print(f"   Aligned {len(target_ids)} targets.")
    print(f"   Missing features: {missing_count} ({missing_count/len(target_ids):.1%})")
    
    return torch.stack(aligned_features)


def attach_node_features(
    data: HeteroData,
    id_maps: Dict[str, Dict[str, int]],
    init_method: str = "random",
    embedding_dim: int = 128,
    pretrained_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    seed: int = 42,
) -> HeteroData:
    """
    Initialize node features for heterogeneous graph.
    
    Args:
        data: HeteroData object
        id_maps: Node ID to index mappings (not used if node_id attribute exists)
        init_method: "random" or "pretrained"
        embedding_dim: Dimension for random initialization
        pretrained_embeddings: Optional dict of {node_type: tensor}
        seed: Random seed
        
    Returns:
        HeteroData with .x attributes attached
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        
        # Check if we have IDs available in the data object
        node_ids = getattr(data[node_type], 'node_id', None)
        

        if init_method == "pretrained" and node_type == "target" and node_ids is not None:
             # Try loading integrated features
             features = load_integrated_target_features(node_ids)
             data[node_type].x = features
             print(f"✅ Loaded integrated features for {node_type}: {data[node_type].x.shape}")
             
        elif init_method == "pretrained" and node_type == "disease" and node_ids is not None:
             # Try loading disease embeddings
             feature_path = "data/node_features/processed/disease_embeddings.pt"
             map_loc = "cuda" if torch.cuda.is_available() else "cpu"
             
             import os
             if os.path.exists(feature_path):
                 print(f"Loading disease embeddings from {feature_path}...")
                 emb_dict = torch.load(feature_path, map_location=map_loc, weights_only=False)
                 
                 # Align
                 aligned = []
                 missing = 0
                 dim = embedding_dim
                 if len(emb_dict) > 0:
                     dim = next(iter(emb_dict.values())).shape[0]
                     
                 zero_vec = torch.zeros(dim)
                 
                 for did in node_ids:
                     if did in emb_dict:
                         aligned.append(emb_dict[did])
                     else:
                         aligned.append(zero_vec)
                         missing += 1
                 
                 data[node_type].x = torch.stack(aligned)
                 print(f"✅ Loaded disease features: {data[node_type].x.shape} (Missing: {missing})")
             else:
                 print(f"⚠️ Disease embeddings not found at {feature_path}. Using random.")
                 data[node_type].x = torch.randn(num_nodes, embedding_dim)

        elif init_method == "pretrained" and node_type == "molecule" and node_ids is not None:
             # Try loading Morgan fingerprints
             feature_path = "data/node_features/processed/molecule_fingerprints.pt"
             map_loc = "cuda" if torch.cuda.is_available() else "cpu"
             
             import os
             if os.path.exists(feature_path):
                 print(f"Loading molecule fingerprints from {feature_path}...")
                 emb_dict = torch.load(feature_path, map_location=map_loc, weights_only=False)
                 
                 # Align
                 aligned = []
                 missing = 0
                 dim = embedding_dim
                 if len(emb_dict) > 0:
                     dim = next(iter(emb_dict.values())).shape[0]
                     
                 zero_vec = torch.zeros(dim)
                 
                 for mid in node_ids:
                     if mid in emb_dict:
                         aligned.append(emb_dict[mid].float())
                     else:
                         aligned.append(zero_vec)
                         missing += 1
                 
                 data[node_type].x = torch.stack(aligned)
                 print(f"✅ Loaded molecule features: {data[node_type].x.shape} (Missing: {missing})")
             else:
                 print(f"⚠️ Molecule features not found at {feature_path}. Using random.")
                 data[node_type].x = torch.randn(num_nodes, embedding_dim)

        elif init_method == "pretrained" and pretrained_embeddings and node_type in pretrained_embeddings:
            data[node_type].x = pretrained_embeddings[node_type]
            print(f"✅ Loaded pretrained features for {node_type}: {data[node_type].x.shape}")
            
        else:
            # Fallback to random
            if init_method == "pretrained":
                 print(f"⚠️ Pretrained requested for {node_type} but not found (or no node_ids). Using random.")
            
            data[node_type].x = torch.randn(num_nodes, embedding_dim)
            print(f"✅ Initialized {node_type} with random features: {data[node_type].x.shape}")
            
    return data

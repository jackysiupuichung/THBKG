#!/usr/bin/env python3
"""
Homogeneous Graph Autoencoder for learning GO and Reactome embeddings.

This script trains a GNN autoencoder on the GO and Reactome hierarchical graphs
(is_subtype_of, is_subpathway_of relationships) to learn low-dimensional embeddings.
These embeddings can then be used as static node features for targets by aggregating
their associated GO/pathway embeddings.

Architecture:
    - Encoder: GCN layers to encode nodes based on hierarchy
    - Decoder: Inner product decoder (built into GAE)
    - Loss: Binary cross-entropy for link prediction
"""

import os
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import negative_sampling, to_undirected, train_test_split_edges
from tqdm import tqdm


class PathwayEncoder(torch.nn.Module):
    """
    GCN-based encoder for pathway/GO embeddings.
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def load_homogeneous_graph(
    edges_file: str,
    make_undirected: bool = True,
) -> Tuple[Data, Dict[str, int]]:
    """
    Load homogeneous graph from edge file.
    
    Args:
        edges_file: Path to parquet file with edges
        make_undirected: Whether to make graph undirected
        
    Returns:
        data: PyG Data object
        node_mapping: {node_id: index}
    """
    print(f"\n📊 Loading graph from {edges_file}...")
    
    df = pd.read_parquet(edges_file)
    print(f"   Loaded {len(df):,} edges")
    
    # Create node mapping
    all_nodes = pd.concat([df['sourceId'], df['targetId']]).unique()
    node_mapping = {nid: i for i, nid in enumerate(all_nodes)}
    
    print(f"   Unique nodes: {len(node_mapping):,}")
    
    # Create edge index
    src_indices = [node_mapping[nid] for nid in df['sourceId']]
    dst_indices = [node_mapping[nid] for nid in df['targetId']]
    
    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    
    # Make undirected if requested
    if make_undirected:
        edge_index = to_undirected(edge_index)
        print(f"   Made undirected: {edge_index.size(1):,} edges")
    
    # Create Data object
    data = Data(
        num_nodes=len(node_mapping),
        edge_index=edge_index,
    )
    
    return data, node_mapping


def train_autoencoder(
    data: Data,
    hidden_dim: int = 128,
    embedding_dim: int = 64,
    num_epochs: int = 100,
    lr: float = 0.001,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Train graph autoencoder to learn node embeddings.
    
    Args:
        data: PyG Data object
        hidden_dim: Hidden dimension
        embedding_dim: Output embedding dimension
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Learned node embeddings [num_nodes, embedding_dim]
    """
    print(f"\n🔧 Training autoencoder...")
    print(f"   Nodes: {data.num_nodes:,}")
    print(f"   Edges: {data.edge_index.size(1):,}")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   Embedding dim: {embedding_dim}")
    
    # Initialize node features (identity matrix or random)
    data.x = torch.eye(data.num_nodes, hidden_dim)  # One-hot-like initialization
    data = data.to(device)
    
    # Create encoder
    encoder = PathwayEncoder(
        in_dim=hidden_dim,
        hidden_dim=hidden_dim,
        out_dim=embedding_dim
    )
    
    # Create GAE model
    model = GAE(encoder).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Encode
        z = model.encode(data.x, data.edge_index)
        
        # Compute loss (recon_loss handles both positive and negative sampling)
        loss = model.recon_loss(z, data.edge_index)
        
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}/{num_epochs} | Loss: {loss.item():.4f} | Best Loss: {best_loss:.4f}")
    
    # Extract final embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index).cpu()
    
    print(f"   ✅ Training complete! Learned embeddings: {embeddings.shape}")
    
    return embeddings


def save_embeddings(
    embeddings: torch.Tensor,
    node_mapping: Dict[str, int],
    output_path: str,
    entity_type: str,
):
    """Save learned embeddings to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create dictionary {node_id: embedding}
    embedding_dict = {}
    for node_id, idx in node_mapping.items():
        embedding_dict[node_id] = embeddings[idx]
    
    # Save as PyTorch tensor dict
    torch.save(embedding_dict, output_path)
    
    print(f"   💾 Saved {len(embedding_dict):,} {entity_type} embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train GNN autoencoder for GO/Reactome embeddings"
    )
    
    parser.add_argument("--edges-file", required=True, 
                        help="Path to static edge parquet file")
    parser.add_argument("--output-path", required=True, 
                        help="Output path for embeddings (.pt file)")
    parser.add_argument("--entity-type", required=True, 
                        choices=["go", "reactome"], 
                        help="Entity type (for logging)")
    
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--make-undirected", action="store_true", default=True,
                        help="Make graph undirected")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"TRAINING {args.entity_type.upper()} AUTOENCODER")
    print("="*80)
    
    # Load graph
    data, node_mapping = load_homogeneous_graph(
        edges_file=args.edges_file,
        make_undirected=args.make_undirected,
    )
    
    # Train autoencoder
    embeddings = train_autoencoder(
        data=data,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
    )
    
    # Save embeddings
    save_embeddings(
        embeddings=embeddings,
        node_mapping=node_mapping,
        output_path=args.output_path,
        entity_type=args.entity_type,
    )
    
    print(f"\n✅ Done! {args.entity_type.capitalize()} embeddings saved to {args.output_path}")
    print("="*80)


if __name__ == "__main__":
    main()

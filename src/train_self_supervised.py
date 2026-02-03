#!/usr/bin/env python3
"""
Time-agnostic self-supervised pretraining via multi-task link prediction.
Excludes clinical trial edges (reserved for fine-tuning).
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.temporal_loader import load_event_graph, to_time_agnostic
from src.data import init_wandb
from src.models.utils import build_model


# Edge type configuration
CLINICAL_TRIAL_KEYWORDS = ['clinical_trial']


def is_clinical_trial_edge(edge_type):
    """Check if edge type is clinical trial (exclude from pretraining)."""
    return any(kw in edge_type[1] for kw in CLINICAL_TRIAL_KEYWORDS)


def get_edge_loss_type(graph, edge_type):
    """Determine if edge uses BCE (binary) or MSE (continuous) loss."""
    edge_store = graph[edge_type]
    if 'edge_attr' not in edge_store:
        return 'bce'  # Default to binary
    
    scores = edge_store['edge_attr'].flatten()
    unique_vals = torch.unique(scores)
    is_binary = set(unique_vals.tolist()).issubset({0.0, 1.0})
    return 'bce' if is_binary else 'mse'


def mask_edges(graph, mask_rate=0.15):
    """
    Randomly mask edges per type for reconstruction.
    Excludes clinical trial edges.
    
    Returns:
        masked_graph: Graph with masked edges removed
        masked_edges: Dict of masked edges per type for reconstruction
    """
    masked_graph = graph.clone()
    masked_edges = {}
    
    for etype in graph.edge_types:
        if is_clinical_trial_edge(etype):
            continue  # Skip clinical trial edges
        
        edge_index = graph[etype].edge_index
        num_edges = edge_index.size(1)
        
        if num_edges == 0:
            continue
        
        num_mask = int(num_edges * mask_rate)
        
        if num_mask == 0:
            continue
        
        # Random masking
        perm = torch.randperm(num_edges)
        mask_idx = perm[:num_mask]
        keep_idx = perm[num_mask:]
        
        # Store masked edges for reconstruction
        masked_edges[etype] = {
            'edge_index': edge_index[:, mask_idx],
            'edge_attr': graph[etype].edge_attr[mask_idx] if 'edge_attr' in graph[etype] else None
        }
        
        # Update graph with kept edges only
        masked_graph[etype].edge_index = edge_index[:, keep_idx]
        if 'edge_attr' in masked_graph[etype]:
            masked_graph[etype].edge_attr = graph[etype].edge_attr[keep_idx]
    
    return masked_graph, masked_edges


def train_one_epoch(model, graph, masked_edges, optimizer, device, edge_loss_config):
    """Train one epoch of self-supervised pretraining."""
    model.train()
    total_loss = 0
    loss_breakdown = {'bce': 0, 'mse': 0}
    num_edge_types = {'bce': 0, 'mse': 0}
    
    for etype, masked_data in masked_edges.items():
        if masked_data['edge_index'].size(1) == 0:
            continue
        
        src_type, rel, dst_type = etype
        edge_index = masked_data['edge_index'].to(device)
        
        # Forward pass
        pred_scores = model(
            {k: v.to(device) for k, v in graph.x_dict.items()},
            {k: v.to(device) for k, v in graph.edge_index_dict.items()},
            edge_index,
            src_type,
            dst_type
        )
        
        # Compute loss based on edge type
        loss_type = edge_loss_config[etype]
        
        if loss_type == 'bce':
            # Binary: all masked edges are positive (score=1)
            targets = torch.ones(edge_index.size(1), device=device)
            loss = F.binary_cross_entropy_with_logits(pred_scores, targets)
            loss_breakdown['bce'] += loss.item()
            num_edge_types['bce'] += 1
        else:  # mse
            # Continuous: use actual edge scores
            targets = masked_data['edge_attr'].flatten().to(device)
            loss = F.mse_loss(pred_scores, targets)
            loss_breakdown['mse'] += loss.item()
            num_edge_types['mse'] += 1
        
        total_loss += loss
    
    # Count total edge types processed
    total_types = sum(num_edge_types.values())
    
    if total_types == 0:
        return 0.0, loss_breakdown
    
    # Backward pass
    avg_loss = total_loss / total_types
    optimizer.zero_grad()
    avg_loss.backward()
    optimizer.step()
    
    # Average loss breakdown
    if num_edge_types['bce'] > 0:
        loss_breakdown['bce'] /= num_edge_types['bce']
    if num_edge_types['mse'] > 0:
        loss_breakdown['mse'] /= num_edge_types['mse']
    
    return avg_loss.item(), loss_breakdown


def main(config_path):
    print("\n" + "="*80)
    print("TIME-AGNOSTIC SELF-SUPERVISED PRETRAINING")
    print("="*80 + "\n")
    
    # Load config
    cfg = OmegaConf.load(config_path)
    
    # Initialize WandB
    if cfg.wandb.enabled:
        init_wandb(cfg)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load full graph (time-agnostic)
    print(f"\n📊 Loading graph from {cfg.data.graph_file}")
    graph = load_event_graph(cfg.data.graph_file, to_undirected=True)
    
    print("   Collapsing temporal graph to static view...")
    graph = to_time_agnostic(graph)  # Collapse temporal to static
    graph = graph.to(device)
    
    # Build edge loss configuration
    edge_loss_config = {}
    excluded_count = 0
    
    for etype in graph.edge_types:
        if is_clinical_trial_edge(etype):
            excluded_count += 1
            continue
        edge_loss_config[etype] = get_edge_loss_type(graph, etype)
    
    bce_count = sum(1 for v in edge_loss_config.values() if v == 'bce')
    mse_count = sum(1 for v in edge_loss_config.values() if v == 'mse')
    
    print(f"\n🎯 Edge Type Configuration:")
    print(f"   Total edge types: {len(graph.edge_types)}")
    print(f"   Excluded (clinical trials): {excluded_count}")
    print(f"   Pretraining on: {len(edge_loss_config)} edge types")
    print(f"     - BCE (binary): {bce_count}")
    print(f"     - MSE (continuous): {mse_count}")
    
    # Build model
    print(f"\n🏗️  Building {cfg.model.name} model...")
    model = build_model(
        model_name=cfg.model.name,
        data=graph,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.pretrain.lr,
        weight_decay=cfg.pretrain.get('weight_decay', 0.0)
    )
    
    # Create output directory
    output_dir = Path(cfg.pretrain.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n🔄 Starting Training...")
    print(f"   Epochs: {cfg.pretrain.num_epochs}")
    print(f"   Mask rate: {cfg.pretrain.mask_rate}")
    print(f"   Learning rate: {cfg.pretrain.lr}")
    
    best_loss = float('inf')
    
    for epoch in range(cfg.pretrain.num_epochs):
        # Mask edges (different masking each epoch)
        masked_graph, masked_edges = mask_edges(graph, cfg.pretrain.mask_rate)
        
        # Train
        loss, loss_breakdown = train_one_epoch(
            model, masked_graph, masked_edges, 
            optimizer, device, edge_loss_config
        )
        
        # Logging
        print(f"Epoch {epoch+1:03d}/{cfg.pretrain.num_epochs} | "
              f"Loss: {loss:.4f} | BCE: {loss_breakdown['bce']:.4f} | MSE: {loss_breakdown['mse']:.4f}")
        
        if cfg.wandb.enabled:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": loss,
                "bce_loss": loss_breakdown['bce'],
                "mse_loss": loss_breakdown['mse']
            })
        
        # Save best model
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), output_dir / "pretrained.pt")
    
    print(f"\n✅ Pretraining Complete!")
    print(f"   Best Loss: {best_loss:.4f}")
    print(f"   Model saved to: {output_dir / 'pretrained.pt'}")
    
    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-supervised pretraining")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)

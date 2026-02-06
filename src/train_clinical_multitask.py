#!/usr/bin/env python3
"""
Fine-tuning script for multi-task clinical trial phase regression.

Supports:
- Frozen pretrained encoder or direct node features
- Negative sampling (1:15 ratio)
- Temporal train/val/test splits
- Multi-task evaluation metrics
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.temporal_loader import load_event_graph
from src.models.multitask_mlp import MultiTaskClinicalMLP, WeightedMultiTaskLoss, compute_task_weights
from src.models.utils import build_model


def load_labels(label_path):
    """Load clinical trial labels from parquet file."""
    df = pd.read_parquet(label_path)
    return df


def create_negative_samples(positive_pairs, all_disease_ids, all_target_ids, ratio=15, seed=42):
    """
    Create negative samples for training.
    
    Args:
        positive_pairs: Set of (disease_id, target_id) tuples with trials
        all_disease_ids: List of all disease IDs
        all_target_ids: List of all target IDs
        ratio: Negative to positive ratio
        seed: Random seed
    
    Returns:
        List of negative (disease_id, target_id) tuples
    """
    np.random.seed(seed)
    
    num_negatives = len(positive_pairs) * ratio
    negatives = []
    
    while len(negatives) < num_negatives:
        disease_id = np.random.choice(all_disease_ids)
        target_id = np.random.choice(all_target_ids)
        
        if (disease_id, target_id) not in positive_pairs:
            negatives.append((disease_id, target_id))
    
    return negatives


def prepare_training_data(labels_df, negative_ratio=15):
    """
    Prepare training data with negative sampling.
    
    Args:
        labels_df: DataFrame with labels
        negative_ratio: Negative to positive ratio
    
    Returns:
        DataFrame with positive and negative samples
    """
    # Positive samples: any pair with at least one trial
    positive_mask = (labels_df[['y_pos', 'y_unmet', 'y_adv', 'y_op']].sum(axis=1) > 0)
    positives = labels_df[positive_mask].copy()
    
    # Create negative samples
    positive_pairs = set(zip(positives['disease_id'], positives['target_id']))
    all_disease_ids = labels_df['disease_id'].unique()
    all_target_ids = labels_df['target_id'].unique()
    
    negative_pairs = create_negative_samples(
        positive_pairs, all_disease_ids, all_target_ids, ratio=negative_ratio
    )
    
    # Create negative DataFrame
    negatives = pd.DataFrame(negative_pairs, columns=['disease_id', 'target_id'])
    negatives['y_pos'] = 0.0
    negatives['y_unmet'] = 0.0
    negatives['y_adv'] = 0.0
    negatives['y_op'] = 0.0
    
    # Combine
    combined = pd.concat([positives, negatives], ignore_index=True)
    
    print(f"   Positives: {len(positives):,}")
    print(f"   Negatives: {len(negatives):,}")
    print(f"   Total: {len(combined):,}")
    
    return combined


def extract_embeddings(graph, encoder, device, freeze_encoder=True):
    """
    Extract node embeddings using encoder or direct features.
    
    Args:
        graph: HeteroData graph
        encoder: Encoder model (None for direct features)
        device: Device
        freeze_encoder: Freeze encoder weights
    
    Returns:
        Dictionary of node embeddings by type
    """
    if encoder is None:
        # Use direct node features
        print("   Using direct node features (no encoder)")
        embeddings = {}
        for node_type in ['disease', 'target']:
            if hasattr(graph[node_type], 'x'):
                embeddings[node_type] = graph[node_type].x.to(device)
            else:
                raise ValueError(f"No features found for {node_type}")
        return embeddings
    
    # Use encoder
    if freeze_encoder:
        encoder.eval()
        print("   Encoder frozen")
    else:
        encoder.train()
        print("   Encoder trainable")
    
    with torch.set_grad_enabled(not freeze_encoder):
        # Forward pass through encoder
        x_dict = {k: v.to(device) for k, v in graph.x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}
        
        embeddings = encoder.encode(x_dict, edge_index_dict)
    
    return embeddings


def train_epoch(model, data_df, embeddings, node_mappings, loss_fn, optimizer, device):
    """Train one epoch."""
    model.train()
    
    # Shuffle data
    data_df = data_df.sample(frac=1).reset_index(drop=True)
    
    total_loss = 0
    task_losses = {'pos': 0, 'unmet': 0, 'adv': 0, 'op': 0}
    
    batch_size = 512
    num_batches = (len(data_df) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        batch_df = data_df.iloc[i*batch_size:(i+1)*batch_size]
        
        # Get node indices
        disease_indices = torch.tensor([
            node_mappings['disease'][did] for did in batch_df['disease_id']
        ], dtype=torch.long, device=device)
        
        target_indices = torch.tensor([
            node_mappings['target'][tid] for tid in batch_df['target_id']
        ], dtype=torch.long, device=device)
        
        # Get embeddings
        disease_emb = embeddings['disease'][disease_indices]
        target_emb = embeddings['target'][target_indices]
        
        # Forward pass
        predictions = model(disease_emb, target_emb)
        
        # Prepare targets
        targets = {
            'pos': torch.tensor(batch_df['y_pos'].values, dtype=torch.float32, device=device),
            'unmet': torch.tensor(batch_df['y_unmet'].values, dtype=torch.float32, device=device),
            'adv': torch.tensor(batch_df['y_adv'].values, dtype=torch.float32, device=device),
            'op': torch.tensor(batch_df['y_op'].values, dtype=torch.float32, device=device)
        }
        
        # Compute loss
        loss, batch_task_losses = loss_fn(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        for task in task_losses:
            task_losses[task] += batch_task_losses[task]
    
    # Average losses
    avg_loss = total_loss / num_batches
    for task in task_losses:
        task_losses[task] /= num_batches
    
    return avg_loss, task_losses


@torch.no_grad()
def evaluate(model, data_df, embeddings, node_mappings, device):
    """Evaluate model."""
    model.eval()
    
    # Get all predictions
    disease_indices = torch.tensor([
        node_mappings['disease'][did] for did in data_df['disease_id']
    ], dtype=torch.long, device=device)
    
    target_indices = torch.tensor([
        node_mappings['target'][tid] for tid in data_df['target_id']
    ], dtype=torch.long, device=device)
    
    disease_emb = embeddings['disease'][disease_indices]
    target_emb = embeddings['target'][target_indices]
    
    # Logits
    predictions = model(disease_emb, target_emb)
    
    # Compute RMSE per task
    metrics = {}
    probs = {} # Store probabilities for saving
    
    for task in ['pos', 'unmet', 'adv', 'op']:
        logit = predictions[task]
        prob = torch.sigmoid(logit).cpu().numpy()
        probs[task] = torch.tensor(prob) # keep as tensor? no numpy is fine for df
        
        target = data_df[f'y_{task}'].values
        
        mse = np.mean((prob - target) ** 2)
        rmse = np.sqrt(mse)
        
        metrics[f'{task}_mse'] = mse
        metrics[f'{task}_rmse'] = rmse
    
    return metrics, probs


def main(config_path):
    print("\n" + "="*80)
    print("MULTI-TASK CLINICAL TRIAL PHASE REGRESSION")
    print("="*80 + "\n")
    
    # Load config
    cfg = OmegaConf.load(config_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load graph
    print(f"\n📂 Loading graph: {cfg.data.graph_file}")
    graph = load_event_graph(cfg.data.graph_file, to_undirected=False)
    
    # Load mappings
    print(f"📂 Loading mappings: {cfg.data.mappings_file}")
    mappings = torch.load(cfg.data.mappings_file)
    node_mappings = mappings['node_mapping']
    
    # Load labels
    print(f"\n📊 Loading labels:")
    train_labels = load_labels(cfg.data.train_labels)
    val_labels = load_labels(cfg.data.val_labels)
    test_labels = load_labels(cfg.data.test_labels)
    
    # Get years from config for logging
    train_year = cfg.data.temporal_split.train[1] if hasattr(cfg.data, 'temporal_split') else "2015"
    val_year = cfg.data.temporal_split.val[1] if hasattr(cfg.data, 'temporal_split') else "2017"
    test_year = cfg.data.temporal_split.test[1] if hasattr(cfg.data, 'temporal_split') else "2024"
    
    print(f"   Train ({train_year}): {len(train_labels):,} pairs")
    print(f"   Val ({val_year}): {len(val_labels):,} pairs")
    print(f"   Test ({test_year}): {len(test_labels):,} pairs")
    
    # Prepare training data with negatives
    print(f"\n🔄 Preparing training data (negative ratio: 1:{cfg.train.negative_ratio})")
    train_data = prepare_training_data(train_labels, negative_ratio=cfg.train.negative_ratio)
    
    # Compute task weights
    task_weights = compute_task_weights(train_labels, inverse_freq=True)
    print(f"\n⚖️  Task weights:")
    for task, weight in task_weights.items():
        print(f"   {task}: {weight:.3f}")
    
    # Build encoder or initialize random embeddings
    encoder = None
    use_random_features = (cfg.model.get('node_features', {}).get('init_method') == 'random')
    
    if use_random_features:
        print(f"\n🎲 Using RANDOM node features (testing mode)")
        # Create random embeddings directly
        # Will be handled in extract_embeddings or we can just override it here
        # Actually extract_embeddings expects an encoder or graph features.
        # If we want random, we should probably generate them here.
        pass
    elif cfg.model.get('use_encoder', False):
        print(f"\n🏗️  Building encoder: {cfg.model.encoder.name}")
        encoder = build_model(
            model_name=cfg.model.encoder.name,
            data=graph,
            hidden_dim=cfg.model.encoder.hidden_dim,
            num_heads=cfg.model.encoder.num_heads,
            num_layers=cfg.model.encoder.num_layers,
            dropout=cfg.model.encoder.dropout
        ).to(device)
        
        # Load pretrained weights if specified
        if cfg.model.encoder.get('pretrained_checkpoint'):
            print(f"   Loading pretrained weights: {cfg.model.encoder.pretrained_checkpoint}")
            encoder.load_state_dict(torch.load(cfg.model.encoder.pretrained_checkpoint, map_location=device))
    
    # Extract embeddings
    # Extract embeddings
    print(f"\n🔧 Extracting embeddings...")
    graph = graph.to(device)
    
    if use_random_features:
        embeddings = {}
        emb_dim = cfg.model.node_features.get('embedding_dim', 128)
        for node_type in ['disease', 'target']:
            num_nodes = graph[node_type].num_nodes
            emb = torch.empty(num_nodes, emb_dim, device=device)
            nn.init.xavier_uniform_(emb)
            embeddings[node_type] = emb
    else:
        embeddings = extract_embeddings(graph, encoder, device, freeze_encoder=cfg.model.get('freeze_encoder', True))
    
    # Get embedding dimensions
    disease_dim = embeddings['disease'].size(1)
    target_dim = embeddings['target'].size(1)
    input_dim = disease_dim + target_dim
    
    print(f"   Disease embedding dim: {disease_dim}")
    print(f"   Target embedding dim: {target_dim}")
    print(f"   Total input dim: {input_dim}")
    
    # Build MLP decoder
    print(f"\n🏗️  Building MLP decoder...")
    model = MultiTaskClinicalMLP(
        input_dim=input_dim,
        hidden_dim=cfg.model.decoder.hidden_dim,
        dropout=cfg.model.decoder.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # Loss and optimizer
    loss_fn = WeightedMultiTaskLoss(weights=task_weights, use_huber=cfg.train.get('use_huber', False))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    
    # Create output directory
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n🔄 Starting Training...")
    print(f"   Epochs: {cfg.train.num_epochs}")
    print(f"   Learning rate: {cfg.train.lr}")
    
    best_val_rmse = float('inf')
    
    for epoch in range(cfg.train.num_epochs):
        # Train
        train_loss, train_task_losses = train_epoch(
            model, train_data, embeddings, node_mappings, loss_fn, optimizer, device
        )
        
        # Evaluate
        val_metrics, _ = evaluate(model, val_labels, embeddings, node_mappings, device)
        
        # Average RMSE across tasks
        avg_val_rmse = np.mean([val_metrics[f'{task}_rmse'] for task in ['pos', 'unmet', 'adv', 'op']])
        
        print(f"Epoch {epoch+1:03d}/{cfg.train.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val RMSE: {avg_val_rmse:.4f}")
        
        # Save best model
        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            torch.save(model.state_dict(), output_dir / "best_decoder.pt")
    
    # Final evaluation on test set
    print(f"\n📊 Final Evaluation on Test Set:")
    model.load_state_dict(torch.load(output_dir / "best_decoder.pt"))
    test_metrics, test_predictions = evaluate(model, test_labels, embeddings, node_mappings, device)
    
    print(f"\n   Per-Task Metrics:")
    for task in ['pos', 'unmet', 'adv', 'op']:
        print(f"   {task:6s}: RMSE={test_metrics[f'{task}_rmse']:.4f}, MSE={test_metrics[f'{task}_mse']:.4f}")
    
    # Save predictions
    # Save predictions
    for task in ['pos', 'unmet', 'adv', 'op']:
        # Ensure we convert to numpy if it's a tensor
        vals = test_predictions[task]
        if isinstance(vals, torch.Tensor):
            vals = vals.cpu().numpy()
        test_labels[f'pred_{task}'] = vals
    
    test_labels.to_parquet(output_dir / "test_predictions.parquet", index=False)
    print(f"\n✅ Saved predictions to {output_dir / 'test_predictions.parquet'}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-task clinical trial regression")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)

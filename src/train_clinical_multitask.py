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
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.temporal_loader import (
    load_event_graph, 
    remove_clinical_trial_edges, 
    filter_graph_by_time, 
    to_time_agnostic
)
from src.models.multitask_mlp import MultiTaskClinicalMLP, WeightedMultiTaskLoss, compute_task_weights
from src.models.utils import build_model


def extract_labels_from_graph(graph, split_year, node_mappings):
    """
    Dynamically extract labels from graph edges up to a specific year.
    
    Args:
        graph: HeteroData object
        split_year: Max year to include (inclusive)
        node_mappings: Dictionary of node mappings
        
    Returns:
        DataFrame with columns: disease_id, target_id, disease_idx, target_idx, y_pos, y_unmet, y_adv, y_op
    """
    print(f"   Extracting labels up to year {split_year}...")
    
    # Task to edge type mapping
    task_edge_map = {
        'pos': 'clinical_trial_positive::chembl',
        'unmet': 'clinical_trial_unmet_efficacy::chembl',
        'adv': 'clinical_trial_adverse_effects::chembl',
        'op': 'clinical_trial_Unknown/Operational::chembl'
    }
    
    # Store data in dictionary: (disease_idx, target_idx) -> {task: score}
    label_data = {}
    
    # Pre-fetch node mappings for reverse lookup if needed (though we use indices primarily)
    # We will store indices directly.
    
    for task, edge_type_name in task_edge_map.items():
        etype = ('disease', edge_type_name, 'target')
        
        if etype not in graph.edge_types:
            print(f"   ⚠️ Warning: Edge type {etype} not found in graph!")
            continue
            
        edge_store = graph[etype]
        edge_index = edge_store.edge_index
        edge_attr = edge_store.edge_attr
        edge_time = edge_store.edge_time if hasattr(edge_store, 'edge_time') else None
        
        if edge_time is None:
             print(f"   ⚠️ Warning: No connection time found for {etype}, using all edges.")
             mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        else:
             # Filter by time
             mask = edge_time <= int(split_year)
        
        # Apply mask
        filtered_indices = edge_index[:, mask]
        filtered_attr = edge_attr[mask] if edge_attr is not None else torch.ones(mask.sum(), 1)
        
        num_edges = filtered_indices.size(1)
        print(f"      {task}: {num_edges:,} edges")
        
        # Aggregate stats
        # We need to map (src, dst) -> max(score)
        # Convert to numpy for easier dict processing
        src_indices = filtered_indices[0].cpu().numpy()
        dst_indices = filtered_indices[1].cpu().numpy()
        scores = filtered_attr.squeeze().cpu().numpy()
        if scores.ndim == 0: scores = np.array([scores]) # handle single scalar case
        
        for i in range(num_edges):
            d_idx = int(src_indices[i])
            t_idx = int(dst_indices[i])
            score = float(scores[i])
            
            key = (d_idx, t_idx)
            if key not in label_data:
                label_data[key] = {t: 0.0 for t in task_edge_map.keys()}
                
            # Max aggregation (if multiple edges exist for same pair)
            label_data[key][task] = max(label_data[key][task], score)

    # Convert to DataFrame
    rows = []
    inv_disease_map = {v: k for k, v in node_mappings['disease'].items()}
    inv_target_map = {v: k for k, v in node_mappings['target'].items()}

    for (d_idx, t_idx), scores in label_data.items():
        row = {
            'disease_idx': d_idx,
            'target_idx': t_idx,
            'disease_id': inv_disease_map.get(d_idx, f"idx_{d_idx}"),
            'target_id': inv_target_map.get(t_idx, f"idx_{t_idx}")
        }
        for task in task_edge_map:
            row[f'y_{task}'] = scores[task]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def prepare_training_data(positives_df, num_disease_nodes, num_target_nodes, negative_ratio=15, seed=42):
    """
    Prepare training data with negative sampling using indices.
    
    Args:
        positives_df: DataFrame with positive samples
        num_disease_nodes: Total number of disease nodes
        num_target_nodes: Total number of target nodes
        negative_ratio: Negative to positive ratio
    
    Returns:
        DataFrame with positive and negative samples
    """
    # Positive samples are already in the DF
    positives = positives_df.copy()
    
    # Create negative samples
    # We use indices directly for efficiency
    existing_pairs = set(zip(positives['disease_idx'], positives['target_idx']))
    
    np.random.seed(seed)
    num_negatives = len(positives) * negative_ratio
    negatives = []
    
    # Batch generation for speed
    batch_size = num_negatives * 2 # Generate more to account for collisions
    
    while len(negatives) < num_negatives:
        d_idxs = np.random.randint(0, num_disease_nodes, batch_size)
        t_idxs = np.random.randint(0, num_target_nodes, batch_size)
        
        for d, t in zip(d_idxs, t_idxs):
            if (d, t) not in existing_pairs:
                negatives.append((d, t))
                if len(negatives) >= num_negatives:
                    break
    
    # Create negative DataFrame
    neg_df = pd.DataFrame(negatives, columns=['disease_idx', 'target_idx'])
    for task in ['pos', 'unmet', 'adv', 'op']:
        neg_df[f'y_{task}'] = 0.0
    
    # Combine
    combined = pd.concat([positives, neg_df], ignore_index=True)
    
    print(f"   Positives: {len(positives):,}")
    print(f"   Negatives: {len(neg_df):,}")
    print(f"   Total: {len(combined):,}")
    
    return combined



class WeightedMultiTaskLoss(nn.Module):
    """
    Weighted loss for multi-task learning.
    
    Supports:
    - 'mse': Mean Squared Error (default)
    - 'huber': Huber Loss
    - 'bce': Binary Cross Entropy
    """
    def __init__(self, weights, loss_type='mse', huber_delta=1.0):
        super().__init__()
        self.weights = weights
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict of task -> logits
            targets: Dict of task -> scalar targets
        """
        total_loss = 0
        task_losses = {}
        
        for task in predictions:
            pred = predictions[task]
            target = targets[task]
            
            # Apply task weight
            weight = self.weights.get(task, 1.0)
            
            # Compute loss based on type
            if self.loss_type == 'bce':
                # BCEWithLogits takes logits
                loss = F.binary_cross_entropy_with_logits(pred, target)
            else:
                # MSE and Huber typically operate on probabilities (0-1) for this task
                # So we apply sigmoid first
                prob = torch.sigmoid(pred)
                
                if self.loss_type == 'huber':
                    loss = F.huber_loss(prob, target, delta=self.huber_delta)
                else: # 'mse'
                    loss = F.mse_loss(prob, target)
            
            weighted_loss = loss * weight
            total_loss += weighted_loss
            task_losses[task] = loss.item() # Log raw unweighted loss
            
        return total_loss, task_losses


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
        
        # Get node indices (directly from dataframe)
        disease_indices = torch.tensor(batch_df['disease_idx'].values, dtype=torch.long, device=device)
        target_indices = torch.tensor(batch_df['target_idx'].values, dtype=torch.long, device=device)
        
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
    disease_indices = torch.tensor(data_df['disease_idx'].values, dtype=torch.long, device=device)
    target_indices = torch.tensor(data_df['target_idx'].values, dtype=torch.long, device=device)
    
    disease_emb = embeddings['disease'][disease_indices]
    target_emb = embeddings['target'][target_indices]
    
    # Logits
    predictions = model(disease_emb, target_emb)
    
    # Compute metrics per task
    metrics = {}
    probs = {} # Store probabilities for saving
    
    for task in ['pos', 'unmet', 'adv', 'op']:
        logit = predictions[task]
        prob = torch.sigmoid(logit).cpu().numpy()
        probs[task] = torch.tensor(prob)
        
        target = data_df[f'y_{task}'].values
        
        # Regression Metrics
        mse = np.mean((prob - target) ** 2)
        rmse = np.sqrt(mse)
        
        metrics[f'{task}_mse'] = mse
        metrics[f'{task}_rmse'] = rmse

        # Classification Metrics
        # Threshold at 0.0 (any evidence) for binary metrics since labels are soft [0.025, 1.0]
        # For 'unmet'/'adv', max score is 0.5, so >0.5 would be all zeros.
        preds_binary = (prob > 0.5).astype(int) # Predictions are sigmoid, 0.5 is fair
        target_binary = (target > 0.0).astype(int) # Targets are soft scores, >0 implies positive
        
        try:
            # Need at least two classes for AUC
            if len(np.unique(target_binary)) < 2:
                metrics[f'{task}_roc_auc'] = float('nan')
            else:
                metrics[f'{task}_roc_auc'] = roc_auc_score(target_binary, prob)
                
            metrics[f'{task}_prec'] = precision_score(target_binary, preds_binary, zero_division=0)
            metrics[f'{task}_rec'] = recall_score(target_binary, preds_binary, zero_division=0)
            metrics[f'{task}_f1'] = f1_score(target_binary, preds_binary, zero_division=0)
            metrics[f'{task}_mcc'] = matthews_corrcoef(target_binary, preds_binary)
            
            # P-value (using Pearson correlation as proxy for significance of relationship)
            # This tests against the null hypothesis of no correlation
            if len(prob) > 1 and np.std(target) > 0 and np.std(prob) > 0:
                 _, p_val = stats.pearsonr(prob, target)
                 metrics[f'{task}_pval'] = p_val
            else:
                 metrics[f'{task}_pval'] = 1.0
            
        except ValueError as e:
            # Handle cases with only one class present
            metrics[f'{task}_roc_auc'] = float('nan')
            metrics[f'{task}_prec'] = 0.0
            metrics[f'{task}_rec'] = 0.0
            metrics[f'{task}_f1'] = 0.0
            metrics[f'{task}_mcc'] = 0.0
            metrics[f'{task}_pval'] = 1.0
    
    return metrics, probs


def main(cfg):
    # Load config if passed as path
    if isinstance(cfg, (str, Path)):
        cfg = OmegaConf.load(cfg)
        
    print("\n" + "="*80)
    print("MULTI-TASK CLINICAL TRIAL PHASE REGRESSION")
    print("="*80 + "\n")
    
    # Debug override
    if cfg.get('debug', False):
        print("\n🐞 DEBUG MODE ENABLED: Truncating datasets to 100 samples")
        cfg.train.num_epochs = 2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = Path(cfg.finetune.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / "config.yaml", 'w') as f:
        OmegaConf.save(cfg, f)
    print(f"✅ Saved config to {output_dir / 'config.yaml'}")
    
    # Load graph
    # IMPORTANT: Must match pretraining directionality for checkpoint compatibility
    # Pretrained model used to_undirected=True, resulting in ~124 relations (66*2 - 8 clinical trial edges)
    print(f"\n📂 Loading graph: {cfg.data.graph_file}")
    graph = load_event_graph(cfg.data.graph_file, to_undirected=True)
    
    # Load mappings
    print(f"📂 Loading mappings: {cfg.data.mappings_file}")
    mappings = torch.load(cfg.data.mappings_file)
    node_mappings = mappings['node_mapping']
    
    # Get years from config for logging and splitting
    # Try finetune split first, then global
    if hasattr(cfg.data, 'temporal_split') and hasattr(cfg.data.temporal_split, 'finetune'):
         split_cfg = cfg.data.temporal_split.finetune
    elif hasattr(cfg.data, 'temporal_split'):
         split_cfg = cfg.data.temporal_split
    else:
         split_cfg = None

    train_year = split_cfg.train[1] if split_cfg else "2015"
    val_year = split_cfg.val[1] if split_cfg else "2017"
    test_year = split_cfg.test[1] if split_cfg else "2024"
    
    # Extract labels from graph dynamically
    print(f"\n📊 Extracting labels from graph:")
    print(f"   Train (<= {train_year})")
    train_labels = extract_labels_from_graph(graph, train_year, node_mappings)
    
    print(f"   Val (<= {val_year})")
    val_labels = extract_labels_from_graph(graph, val_year, node_mappings)
    
    print(f"   Test (<= {test_year})")
    test_labels = extract_labels_from_graph(graph, test_year, node_mappings)
    
    # Truncate for debug
    if cfg.get('debug', False):
        print("\n🐞 DEBUG MODE: Truncating datasets to 100 samples")
        train_labels = train_labels.head(100)
        val_labels = val_labels.head(100)
        test_labels = test_labels.head(100)
    
    print(f"   Train: {len(train_labels):,} pairs")
    print(f"   Val:   {len(val_labels):,} pairs")
    print(f"   Test:  {len(test_labels):,} pairs")
    
    # Prepare training data with negatives
    print(f"\n🔄 Preparing training data (negative ratio: 1:{cfg.finetune.negative_ratio})")
    num_disease_nodes = graph['disease'].num_nodes
    num_target_nodes = graph['target'].num_nodes
    
    train_data = prepare_training_data(
        train_labels, 
        num_disease_nodes=num_disease_nodes,
        num_target_nodes=num_target_nodes,
        negative_ratio=cfg.finetune.negative_ratio
    )
    
    # Compute task weights
    task_weights = compute_task_weights(train_labels, inverse_freq=True)
    print(f"\n⚖️  Task weights:")
    for task, weight in task_weights.items():
        print(f"   {task}: {weight:.3f}")
    
    # Build encoder
    encoder = None
    train_context = None  # Will store train-only graph for encoder
    
    # Check if use_encoder is implied or explicit (default True for this script)
    use_encoder = cfg.model.get('use_encoder', True) 
    
    if use_encoder:
        # ========================================================================
        # Create temporal context for encoder (ONLY train split)
        # ========================================================================
        # Prevent temporal leakage: encoder should only see edges up to train year
        # NOTE: We keep clinical trial edges in the context to match pretrain setup
        #       They are excluded from SUPERVISION, not from MESSAGE PASSING
        print(f"\n🏗️  Building encoder: {cfg.model.name}")
        print(f"   Creating train context (≤ {train_year}) to prevent temporal leakage...")
        
        # Filter to train year
        train_temporal = filter_graph_by_time(graph, train_year)
        # Collapse to static (deduplicate, max aggregation)
        train_context = to_time_agnostic(train_temporal)
        
        # ========================================================================
        # STEP 3: Build encoder using TRAIN CONTEXT metadata
        # ========================================================================
        print(f"   Step 3: Building encoder model...")
        encoder = build_model(
            model_name=cfg.model.name,
            data=train_context,  # CRITICAL: Use train_context, not full graph
            hidden_dim=cfg.model.hidden_dim,
            num_heads=cfg.model.num_heads,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout
        ).to(device)
        
        # Load pretrained weights if specified in finetune.model
        pretrained_checkpoint = cfg.finetune.get('model', {}).get('pretrained_checkpoint')
        if pretrained_checkpoint:
             print(f"   Loading pretrained weights: {pretrained_checkpoint}")
             checkpoint = torch.load(pretrained_checkpoint, map_location=device)
             # Use strict=False to allow for decoder mismatches (we only need the encoder convs)
             missing_keys, unexpected_keys = encoder.load_state_dict(checkpoint, strict=False)
             
             print(f"   Pretrained weights loaded (strict=False).")
             if missing_keys:
                 print(f"   Missing keys (expected): {len(missing_keys)}")
                 # Verify that convs are NOT missing
                 convs_missing = [k for k in missing_keys if 'convs' in k]
                 if convs_missing:
                     print(f"   ⚠️ WARNING: Some CONV layers are missing! {convs_missing[:5]}...")
                 else:
                     print(f"   ✅ All encoder (convs) layers loaded.")
             
             if unexpected_keys:
                  print(f"   Unexpected keys (ignored): {len(unexpected_keys)}")
     
    # Extract embeddings
    print(f"\n🔧 Extracting embeddings...")
    
    if use_encoder and train_context is not None:
        # Use train context for encoder (temporal isolation)
        train_context = train_context.to(device)
        freeze_encoder = cfg.finetune.get('model', {}).get('freeze_encoder', True)
        embeddings = extract_embeddings(train_context, encoder, device, freeze_encoder=freeze_encoder)
    else:
        # Direct features (no encoder)
        graph = graph.to(device)
        embeddings = extract_embeddings(graph, encoder, device, freeze_encoder=False)
    
    # Get embedding dimensions
    disease_dim = embeddings['disease'].size(1)
    target_dim = embeddings['target'].size(1)
    input_dim = disease_dim + target_dim
    
    print(f"   Disease embedding dim: {disease_dim}")
    print(f"   Target embedding dim: {target_dim}")
    print(f"   Total input dim: {input_dim}")
    
    # Build MLP decoder
    print(f"\n🏗️  Building MLP decoder...")
    
    # Get decoder config from finetune.model.decoder
    decoder_cfg = cfg.finetune.get('model', {}).get('decoder', {})
    hidden_dim = decoder_cfg.get('hidden_dim', 32)
    dropout = decoder_cfg.get('dropout', 0.1)

    model = MultiTaskClinicalMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # Loss and optimizer
    # Determine loss type
    # Loss and optimizer
    # Determine loss type
    if cfg.finetune.get('use_huber', False):
        loss_type = 'huber'
    elif cfg.finetune.get('use_bce_loss', False): # Optional explicit flag for BCE
        loss_type = 'bce'
    else:
        loss_type = 'mse' # Default
        
    print(f"\n📉 Loss Function: {loss_type.upper()}")
    
    loss_fn = WeightedMultiTaskLoss(
        weights=task_weights, 
        loss_type=loss_type
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.finetune.lr, weight_decay=cfg.finetune.weight_decay)
    
    # Training loop
    print(f"\n🔄 Starting Training...")
    print(f"   Epochs: {cfg.finetune.num_epochs}")
    print(f"   Learning rate: {cfg.finetune.lr}")
    
    best_val_metric = float('inf')
    
    # Determine reporting metric (MSE vs RMSE)
    # Default to MSE as requested
    report_mse = True 
    metric_key = 'mse'
    
    for epoch in range(cfg.finetune.num_epochs):
        # Train
        train_loss, train_task_losses = train_epoch(
            model, train_data, embeddings, node_mappings, loss_fn, optimizer, device
        )
        
        # Evaluate
        val_metrics, _ = evaluate(model, val_labels, embeddings, node_mappings, device)
        
        # Average Metric across tasks
        avg_val_metric = np.mean([val_metrics[f'{task}_{metric_key}'] for task in ['pos', 'unmet', 'adv', 'op']])
        
        print(f"Epoch {epoch+1:03d}/{cfg.finetune.num_epochs} | "
              f"Train Loss ({loss_type}): {train_loss:.4f} | Val {metric_key.upper()}: {avg_val_metric:.4f}")
        
        # Save best model
        if avg_val_metric < best_val_metric:
            best_val_metric = avg_val_metric
            torch.save(model.state_dict(), output_dir / "best_decoder.pt")
    
    # Final evaluation on test set
    print(f"\n📊 Final Evaluation on Test Set:")
    model.load_state_dict(torch.load(output_dir / "best_decoder.pt"))
    test_metrics, test_predictions = evaluate(model, test_labels, embeddings, node_mappings, device)
    
    # Format per-task metrics as a table
    print(f"\n   Per-Task Metrics:")
    metric_rows = []
    for task in ['pos', 'unmet', 'adv', 'op']:
        row = {
            'Task': task,
            'MSE': f"{test_metrics[f'{task}_mse']:.4f}",
            'RMSE': f"{test_metrics[f'{task}_rmse']:.4f}",
            'AUC': f"{test_metrics[f'{task}_roc_auc']:.4f}",
            'F1': f"{test_metrics[f'{task}_f1']:.4f}",
            'MCC': f"{test_metrics[f'{task}_mcc']:.4f}",
            'Prec': f"{test_metrics[f'{task}_prec']:.4f}",
            'Rec': f"{test_metrics[f'{task}_rec']:.4f}",
            'P-Val': f"{test_metrics[f'{task}_pval']:.2e}"
        }
        metric_rows.append(row)
    
    metrics_df = pd.DataFrame(metric_rows)
    print(metrics_df.to_string(index=False))
    
    # Save predictions ONLY if configured
    if cfg.finetune.get('save_predictions', False):
        for task in ['pos', 'unmet', 'adv', 'op']:
            vals = test_predictions[task]
            if isinstance(vals, torch.Tensor):
                vals = vals.cpu().numpy()
            test_labels[f'pred_{task}'] = vals
        
        test_labels.to_parquet(output_dir / "test_predictions.parquet", index=False)
        print(f"\n✅ Saved predictions to {output_dir / 'test_predictions.parquet'}")
    else:
        print("\nℹ️  Skipping raw prediction file (save_predictions=False)")

    # Save results to yaml
    results = {
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'config': OmegaConf.to_container(cfg, resolve=True)
    }
    
    # --- Integration of Ranking Evaluation ---
    print("\n" + "="*80)
    print("🚀 Running Ranking Evaluation (Novel Target Prioritization)")
    print("="*80)
    
    # 1. Load Validation Diseases
    val_diseases_path = Path("data/validation_diseases.csv")
    if val_diseases_path.exists():
        print(f"📂 Loading validation diseases from {val_diseases_path}")
        val_df = pd.read_csv(val_diseases_path)
        # Filter for valid indices (graph_node_idx != -1)
        valid_diseases = val_df[val_df['graph_node_idx'] != -1]['graph_node_idx'].unique()
        valid_diseases = set(valid_diseases)
        print(f"   Found {len(valid_diseases)} valid disease indices for evaluation.")
    else:
        print(f"⚠️  Validation disease file not found at {val_diseases_path}")
        print("   Skipping specific validation subset filtering (evaluating on all test diseases).")
        valid_diseases = None

    # 2. Prepare Data for Ranking
    # We need to construct:
    # - History: All edges in Train + Val (to mask known edges)
    # - Test Targets: Edges in Test (to evaluate against)
    
    # Filter strictly novel: In Test BUT NOT in History
    # Combine Train + Val for history masking
    history_pairs = set()
    for _, row in train_labels.iterrows():
        history_pairs.add((int(row['disease_idx']), int(row['target_idx'])))
    for _, row in val_labels.iterrows():
        history_pairs.add((int(row['disease_idx']), int(row['target_idx'])))
        
    print(f"   History Size (Train + Val): {len(history_pairs):,} pairs")
    
    # Test Pairs (Ground Truth)
    # Filter strictly novel: In Test BUT NOT in History
    test_pairs = {}
    test_novel_count = 0
    for _, row in test_labels.iterrows():
        d_idx = int(row['disease_idx'])
        t_idx = int(row['target_idx'])
        
        # Check novelty
        if (d_idx, t_idx) in history_pairs:
            continue
            
        test_novel_count += 1
        scores = {col: row[col] for col in ['y_pos', 'y_unmet', 'y_adv', 'y_op']}
        
        # Structure matches evaluate_clinical_ranking: (d, t) -> scores
        test_pairs[(d_idx, t_idx)] = scores

    print(f"   Test Pairs (Total): {len(test_labels):,}")
    print(f"   Test Pairs (Novel): {test_novel_count:,}")
    
    if test_novel_count > 0:
        # 3. Run Ranking Evaluation for Custom Scoring Strategies
        ranking_results = evaluate_ranking(
            model=model,
            embeddings=embeddings,
            history_pairs=history_pairs,
            test_pairs=test_pairs,
            num_disease_nodes=num_disease_nodes,
            num_target_nodes=num_target_nodes,
            device=device,
            k_values=[100, 200, 500],
            valid_disease_indices=valid_diseases
        )
        
        results['ranking_metrics'] = ranking_results
    else:
        print("⚠️  No novel test pairs found. Skipping ranking evaluation.")

    # -----------------------------------------

    import yaml
    with open(output_dir / "results.yaml", 'w') as f:
        yaml.dump(results, f)
    print(f"✅ Saved results to {output_dir / 'results.yaml'}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)


def evaluate_ranking(
    model, 
    embeddings, 
    history_pairs, 
    test_pairs, 
    num_disease_nodes, 
    num_target_nodes, 
    device, 
    k_values=[100, 200, 500],
    valid_disease_indices=None
):
    """
    Evaluate ranking metrics per disease using custom scoring strategies.
    Strategies:
    1. Positive Outcome (Pos)
    2. Pos / Unmet
    3. Pos / Max(Unmet, Adv)
    """
    model.eval()
    
    # All target indices (candidates)
    target_emb_all = embeddings['target'].to(device) # [N_t, dim]
    
    # Pre-organize history: disease -> set(target_indices)
    history_map = {}
    for (d, t) in history_pairs:
        if d not in history_map: history_map[d] = set()
        history_map[d].add(t)

    all_task_results = {}
    
    # Define Ground Truth: Relevant = (Pos > 0 or Op > 0)
    # We want to retrieve successful trials.
    # We explicitly exclude pure 'unmet' or 'adv' (failures) from the "Relevant" set for these metrics.
    test_ground_truth = {}
    for (d, t), scores in test_pairs.items():
        if scores['y_pos'] > 0 or scores['y_op'] > 0:
            if d not in test_ground_truth: test_ground_truth[d] = set()
            test_ground_truth[d].add(t)
            
    test_diseases = set(test_ground_truth.keys())

    # Filter by validation set if provided
    if valid_disease_indices is not None:
        original_count = len(test_diseases)
        test_diseases = test_diseases.intersection(valid_disease_indices)
        print(f"   Filtered diseases: {original_count} -> {len(test_diseases)} (based on validation list)")
    
    if len(test_diseases) == 0:
        print(f"   ⚠️  No diseases with relevant targets left. Skipping.")
        return {}

    # Define Strategies
    strategies = ['Score_Pos', 'Score_Pos_div_Unmet', 'Score_Pos_div_MaxUnmetAdv']
    
    for strategy in strategies:
        print(f"\n   --- Ranking Strategy: {strategy} ---")
        
        # Pre-compute metrics storage
        metrics = {k: {'precision': [], 'recall': [], 'hits': [], 'mrr': [], 'ndcg': []} for k in k_values}
        
        # Loop over diseases
        for d_idx in tqdm(test_diseases, desc=f"Ranking ({strategy})"):
            true_targets = test_ground_truth.get(d_idx, set())
            history = history_map.get(d_idx, set())
            
            # Prepare inputs
            d_emb = embeddings['disease'][d_idx].unsqueeze(0).to(device) # [1, dim]
            d_emb_expanded = d_emb.expand(num_target_nodes, -1)
            
            # Forward pass
            with torch.no_grad():
                logits = model(d_emb_expanded, target_emb_all) # returns dict
                
                # Get Probabilities
                p_pos = torch.sigmoid(logits['pos'])
                p_unmet = torch.sigmoid(logits['unmet'])
                p_adv = torch.sigmoid(logits['adv'])
                # p_op = torch.sigmoid(logits['op']) # Not used in scoring formula, but used in GT
                
                eps = 1e-6
                
                if strategy == 'Score_Pos':
                    scores = p_pos
                elif strategy == 'Score_Pos_div_Unmet':
                    scores = p_pos / (p_unmet + eps)
                elif strategy == 'Score_Pos_div_MaxUnmetAdv':
                    denom = torch.max(p_unmet, p_adv) + eps
                    scores = p_pos / denom
                else:
                    scores = p_pos # Default
                
            # Mask history
            if history:
                hist_indices = torch.tensor(list(history), device=device, dtype=torch.long)
                scores[hist_indices] = -1.0 # Set to lowest possible
                
            # Ranking
            max_k = max(k_values)
            top_k_scores, top_k_indices = torch.topk(scores, max_k)
            top_k_indices = top_k_indices.cpu().tolist()
            
            # Metrics
            for k in k_values:
                curr_top = top_k_indices[:k]
                intersects = len(set(curr_top) & true_targets)
                
                # Recall@K
                if len(true_targets) > 0:
                    recall = intersects / len(true_targets)
                else:
                    recall = 0.0
                metrics[k]['recall'].append(recall)
                
                # Precision@K
                precision = intersects / k
                metrics[k]['precision'].append(precision)
                
                # Hits@K
                metrics[k]['hits'].append(1.0 if intersects > 0 else 0.0)
                
                # MRR
                rr = 0.0
                for rank, t_idx in enumerate(curr_top):
                    if t_idx in true_targets:
                        rr = 1.0 / (rank + 1)
                        break
                metrics[k]['mrr'].append(rr)
                
                # NDCG
                dcg = 0.0
                idcg = 0.0
                
                # DCG
                for i, t_idx in enumerate(curr_top):
                    if t_idx in true_targets:
                        dcg += 1.0 / np.log2(i + 2)
                
                # IDCG
                num_relevant = min(k, len(true_targets))
                for i in range(num_relevant):
                    idcg += 1.0 / np.log2(i + 2)
                    
                metrics[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0)

        # Average metrics for this strategy
        task_final_results = {}
        ranking_rows = []
        
        for k in k_values:
            avg_rec = np.mean(metrics[k]['recall']) if metrics[k]['recall'] else 0.0
            avg_prec = np.mean(metrics[k]['precision']) if metrics[k]['precision'] else 0.0
            avg_mrr = np.mean(metrics[k]['mrr']) if metrics[k]['mrr'] else 0.0
            avg_ndcg = np.mean(metrics[k]['ndcg']) if metrics[k]['ndcg'] else 0.0
            
            task_final_results[f'Recall@{k}'] = float(avg_rec)
            task_final_results[f'Precision@{k}'] = float(avg_prec)
            task_final_results[f'MRR@{k}'] = float(avg_mrr)
            task_final_results[f'NDCG@{k}'] = float(avg_ndcg)
            
            ranking_rows.append({
                'K': k,
                'Recall': f"{avg_rec:.4f}",
                'Precision': f"{avg_prec:.4f}",
                'MRR': f"{avg_mrr:.4f}",
                'NDCG': f"{avg_ndcg:.4f}"
            })
            
        print(f"\n📊 Ranking Results for {strategy}:")
        df_ranking = pd.DataFrame(ranking_rows)
        print(df_ranking.to_string(index=False))
        
        all_task_results[strategy] = task_final_results
        
    return all_task_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-task clinical trial regression")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)

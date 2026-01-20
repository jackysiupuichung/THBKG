#!/usr/bin/env python3
"""
Main training script for Temporal HGT link prediction.

Uses LinkNeighborLoader with Snapshot + Filter strategy for valid temporal training
without requiring pyg-lib's optimized temporal sampler.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.nn import MIPSKNNIndex

from data.temporal_loader import load_event_graph, get_temporal_masks, filter_graph_by_time
from models.utils import build_hgt_model, count_parameters


def main(config_path: str):
    """
    Main training pipeline.
    
    Args:
        config_path: Path to configuration file
    """
    print("\n" + "="*80)
    print("TEMPORAL HGT LINK PREDICTION (Snapshot Strategy)")
    print("="*80 + "\n")
    
    # ============================================================
    # 1. Load Configuration
    # ============================================================
    print(f"📄 Loading config from {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # Load base config if using experiments
    if "defaults" in cfg:
        project_root = os.path.dirname(os.path.dirname(__file__))
        base_config_path = os.path.join(project_root, "config/benchmark_config.yaml")
        base_cfg = OmegaConf.load(base_config_path)
        cfg = OmegaConf.merge(base_cfg, cfg)
        
    print(f"✅ Configuration loaded")
    print(f"   Experiment: {cfg.get('experiment_name', 'default')}")
    print(f"   Train year: {cfg.data.temporal_split.train_year}")
    print(f"   Val year:   {cfg.data.temporal_split.val_year}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # ============================================================
    # 2. Load Event Graph
    # ============================================================
    project_root = os.path.dirname(os.path.dirname(__file__))
    temporal_graph_path = os.path.join(project_root, cfg.data.temporal_graph_file)
    
    print(f"\n📂 Loading event graph from {temporal_graph_path}...")
    
    hetero_data = load_event_graph(
        temporal_graph_path,
        attach_features=True,
        embedding_dim=cfg.model.node_features.embedding_dim,
        seed=cfg.train.seed
    )
    
    print(f"✅ Loaded HeteroData (Full):")
    print(f"   Total nodes: {sum(d.num_nodes for d in hetero_data.node_stores):,}")
    
    # ============================================================
    # 3. Create Train Snapshot (Snapshot strategy)
    # ============================================================
    print(f"\n⏰ Creating training snapshot (<= {cfg.data.temporal_split.train_year})...")
    
    # Filter graph to train_year to prevent leakage during training
    train_data = filter_graph_by_time(hetero_data, cfg.data.temporal_split.train_year)
    
    # Get supervision edge type
    src_type = cfg.data.graph.supervision.src_type
    dst_type = cfg.data.graph.supervision.dst_type
    relation = cfg.data.graph.supervision.relation
    
    supervision_edge_type = None
    for et in train_data.edge_types:
        if (et[0] == src_type and et[2] == dst_type and relation in et[1]):
            supervision_edge_type = et
            break
            
    if supervision_edge_type is None:
        raise ValueError(f"Supervision edge type not found: {src_type}->{relation}->{dst_type}")
        
    print(f"🎯 Supervision edge: {supervision_edge_type}")
    
    # In train_data, ALL edges are valid training edges (since we filtered)
    # But we might want to split last year of train_data for internal validation?
    # For now, let's train on ALL edges in train_data
    
    train_edge_index = train_data[supervision_edge_type].edge_index
    # Handle scores: train_data uses edge_attr
    if 'edge_attr' in train_data[supervision_edge_type]:
        train_edge_label = train_data[supervision_edge_type].edge_attr.squeeze()
    else:
        # Fallback if no scores
        train_edge_label = torch.ones(train_edge_index.size(1))
        
    print(f"   Training edges: {train_edge_index.size(1):,} (All <= {cfg.data.temporal_split.train_year})")
    
    # ============================================================
    # 4. Create Training Loader
    # ============================================================
    print(f"\n🚚 Creating LinkNeighborLoader (Snapshot Mode)...")
    
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        edge_label_index=(supervision_edge_type, train_edge_index),
        edge_label=train_edge_label,
        # NO time_attr passed here -> Standard sampling on static snapshot
        neg_sampling=dict(
            mode='triplet',
            amount=1.0
        ),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # ============================================================
    # 5. Build Model
    # ============================================================
    print(f"\n🏗️ Building HGT model...")
    
    model = build_hgt_model(
        train_data,
        hidden_dim=cfg.model.hgt.hidden_dim,
        num_heads=cfg.model.hgt.num_heads,
        num_layers=cfg.model.hgt.num_layers,
        dropout=cfg.model.hgt.dropout,
    )
    
    model = model.to(device)
    num_params = count_parameters(model)
    print(f"✅ Model: {num_params:,} parameters")
    
    # ============================================================
    # 6. Training Loop
    # ============================================================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )
    
    print(f"\n{'='*80}")
    print(f"TRAINING (MSE Loss)")
    print(f"{'='*80}\n")
    
    for epoch in range(cfg.train.num_epochs):
        model.train()
        total_loss = 0
        total_examples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.num_epochs}")
        
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            pred_scores = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch[supervision_edge_type].edge_label_index,
                src_type,
                dst_type
            )
            
            # Prepare targets
            num_pos = batch[supervision_edge_type].edge_label.size(0)
            full_batch_size = batch[supervision_edge_type].edge_label_index.size(1)
            num_neg = full_batch_size - num_pos
            
            pos_targets = batch[supervision_edge_type].edge_label.float()
            neg_targets = torch.zeros(num_neg, device=device)
            
            targets = torch.cat([pos_targets, neg_targets])
            
            # Loss
            curr_pred = pred_scores[:targets.size(0)]
            loss = F.mse_loss(curr_pred, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * full_batch_size
            total_examples += full_batch_size
            
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = total_loss / total_examples
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
        
    # ============================================================
    # 7. Evaluation (Exhaustive Ranking)
    # ============================================================
    print(f"\n{'='*80}")
    print(f"EVALUATION (Exhaustive Ranking)")
    print(f"{'='*80}\n")
    
    # Get validation edges from FULL graph (edges in val_year)
    # Logic: edges > train_year AND edges <= val_year
    val_year = cfg.data.temporal_split.val_year
    train_year = cfg.data.temporal_split.train_year
    
    masks = get_temporal_masks(hetero_data, train_year, val_year)
    _, val_mask, _ = masks[supervision_edge_type]
    
    # Exclusion mask should include ALL edges seen so far (<= val_year or <= train_year?)
    # Standard recommendation: Exclude training edges from candidates?
    # Yes, exclude anything <= train_year for sure.
    # What about edges in val_year that overlap? (e.g. if we are predicting one, we shouldn't rank others?)
    # Usually we exclude ALL known positives in the history <= train_year.
    exclusion_mask = masks[supervision_edge_type][0] # train_mask (<= train_year)
    
    evaluate_ranking(
        model, 
        train_data,            # Use TRAIN SNAPSHOT for creating embeddings (No leakage)
        hetero_data,           # Use FULL DATA to find val edges
        supervision_edge_type, 
        val_mask,             # Edges to predict
        exclusion_mask,       # Edges to exclude from ranking
        device, 
        k_values=cfg.eval.k_values
    )
    
    # Save model
    exp_name = cfg.get('experiment_name', 'default')
    output_dir = f"runs/{exp_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = f"{output_dir}/model.pt"
    
    torch.save(model.state_dict(), model_path)
    print(f"\n💾 Saved model to {model_path}")


def evaluate_ranking(
    model, 
    inference_data, 
    ground_truth_data, 
    edge_type, 
    eval_mask, 
    exclusion_mask, 
    device, 
    k_values=[10, 50, 100]
):
    """
    Evaluate with exhaustive ranking using MIPS.
    
    Args:
        model: Trained model
        inference_data: Graph snapshot to use for generating embeddings
        ground_truth_data: Full graph containing ground truth edges
        edge_type: Edge type to evaluate
        eval_mask: Mask for ground truth edges in ground_truth_data
        exclusion_mask: Mask for edges to exclude in ground_truth_data
    """
    print(f"🔍 Evaluating on {eval_mask.sum()} edges...")
    
    src_type, _, dst_type = edge_type
    
    # 1. Build Ground Truth Dict
    eval_edge_index = ground_truth_data[edge_type].edge_index[:, eval_mask]
    eval_dict = {}
    
    src_indices = eval_edge_index[0].tolist()
    dst_indices = eval_edge_index[1].tolist()
    
    unique_src_nodes = set()
    
    for src, dst in zip(src_indices, dst_indices):
        if src not in eval_dict:
            eval_dict[src] = []
        eval_dict[src].append(dst)
        unique_src_nodes.add(src)
        
    print(f"   Evaluating {len(eval_dict)} unique source nodes")
    
    # 2. Build Exclusion Index
    # Edges that should NOT be ranked (training edges)
    # We use ground_truth_data with exclusion_mask
    exclude_edge_index = ground_truth_data[edge_type].edge_index[:, exclusion_mask].to(device)
    
    from torch_geometric import EdgeIndex
    num_src = ground_truth_data[src_type].num_nodes
    num_dst = ground_truth_data[dst_type].num_nodes
    
    # Sparse index for fast lookup
    exclude_links = EdgeIndex(
        exclude_edge_index,
        sparse_size=(num_src, num_dst)
    ).sort_by('row')[0]
    
    # 3. Generate Embeddings (using inference_data)
    print("   Generating embeddings using inference snapshot...")
    model.eval()
    
    with torch.no_grad():
        x_dict = {k: v.to(device) for k, v in inference_data.x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in inference_data.edge_index_dict.items()}
        
        # This assumes graph fits in GPU memory. 
        # For larger graphs, use NeighborLoader to generate embeddings node-by-node.
        try:
            out_dict = model.encode(x_dict, edge_index_dict)
            src_emb = out_dict[src_type]
            dst_emb = out_dict[dst_type]
        except RuntimeError:
            print("   ⚠️ Full graph inference failed (OOM). Switching to CPU...")
            model = model.cpu()
            x_dict = {k: v.cpu() for k, v in inference_data.x_dict.items()}
            edge_index_dict = {k: v.cpu() for k, v in inference_data.edge_index_dict.items()}
            out_dict = model.encode(x_dict, edge_index_dict)
            src_emb = out_dict[src_type]
            dst_emb = out_dict[dst_type]
            model = model.to(device) # Move back
            
    # 4. MIPS Search
    print("   Indexing candidates...")
    # Move embeddings to CPU for large-scale MIPS if needed, or keep on GPU
    # MIPS using PyG
    mips = MIPSKNNIndex(dst_emb)
    
    metrics = {k: {'p': [], 'r': [], 'ndcg': []} for k in k_values}
    max_k = max(k_values)
    
    print("   Ranking...")
    for src_id in tqdm(list(unique_src_nodes), desc="Ranking sources"):
        true_dsts = set(eval_dict[src_id])
        
        # Get exclusion for this source
        # sparse_narrow returns (row, col) indices but normalized
        # We need the column indices of neighbors
        start = exclude_links.indptr[src_id]
        end = exclude_links.indptr[src_id+1]
        exclude_dsts = exclude_links.col[start:end]
        
        # Search
        src_vec = src_emb[src_id:src_id+1]
        _, pred_indices = mips.search(src_vec, max_k, exclude_dsts)
        
        top_k = pred_indices[0].tolist()
        
        for k in k_values:
            curr_top = top_k[:k]
            hits = len(set(curr_top) & true_dsts)
            
            metrics[k]['p'].append(hits / k)
            metrics[k]['r'].append(hits / len(true_dsts))
            
            # NDCG
            dcg = 0.0
            idcg = 0.0
            
            for i, t in enumerate(curr_top):
                if t in true_dsts:
                    dcg += 1.0 / np.log2(i + 2)
            
            for i in range(min(k, len(true_dsts))):
                idcg += 1.0 / np.log2(i + 2)
                
            metrics[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0)
            
    # 5. Report
    print(f"\n✅ Results:")
    for k in k_values:
        p = np.mean(metrics[k]['p'])
        r = np.mean(metrics[k]['r'])
        n = np.mean(metrics[k]['ndcg'])
        print(f"   k={k:<3}: P={p:.4f}, R={r:.4f}, NDCG={n:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/benchmark_config.yaml")
    args = parser.parse_args()
    main(args.config)

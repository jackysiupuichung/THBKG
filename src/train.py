#!/usr/bin/env python3
import argparse
import os
import datetime
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf

from src.data.split import temporal_and_cold_split
from src.data.dataset import InteractionDataset
from src.models.base_lightning import BaseRecLightning
from src.models.ncf import NCF

from src.models.utils import initialise_model, collate_variable, collect_predictions

def create_datasets(train_df, valid_df, test_df, user_map, item_map, all_interactions, cfg):
    """Create train, validation, and test datasets."""
    
    train_ds = InteractionDataset(
        train_df,
        user_map,
        item_map,
        num_neg=cfg.model.num_neg if cfg.model.loss_type == "bpr" else 0,
        dynamic=True
    )

    valid_ds = InteractionDataset(
        valid_df,
        user_map,
        item_map,
        exhaustive_eval=True,
        all_interactions=all_interactions
    )

    test_ds = InteractionDataset(
        test_df,
        user_map,
        item_map,
        exhaustive_eval=True,
        all_interactions=all_interactions
    )

    return train_ds, valid_ds, test_ds

def create_dataloaders(train_ds, valid_ds, test_ds, cfg):
    """Create dataloaders for train, validation, and test datasets."""
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.train.batch_size,
        num_workers=4,
        collate_fn=collate_variable
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        num_workers=4,
        collate_fn=collate_variable
    )

    return train_loader, valid_loader, test_loader


def main(cfg):
    # -----------------------
    # Step 0: Create run directory
    # -----------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{cfg.model.name}_{cfg.model.loss_type}_{cfg.data.cutoff}_{cfg.data.horizon}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save a copy of config into run_dir for reproducibility
    OmegaConf.save(config=cfg, f=os.path.join(run_dir, "config.yaml"))

    print(f"🚀 Starting run → {run_dir}")

    # -----------------------
    # Step 1: Custom split
    # -----------------------
    cold_start_diseases = []
    if cfg.data.cold_start_file and os.path.exists(cfg.data.cold_start_file):
        print(f"✅ Loaded cold start diseases from {cfg.data.cold_start_file}")
        cold_start_df = pd.read_csv(cfg.data.cold_start_file)
        cold_start_diseases = cold_start_df.iloc[:, 0].dropna().astype(str).tolist()
            

    train_df, valid_df, test_df = temporal_and_cold_split(
        cfg.data.parquet,
        cutoff=cfg.data.cutoff,
        horizon=cfg.data.horizon,
        cold_start_diseases=cold_start_diseases,
        out_dir=run_dir
    )

    # -----------------------
    # Step 2: Build global maps
    # -----------------------
    # Load node files for targets (users) and diseases (items)
    disease_nodes = pd.read_parquet(cfg.data.disease_nodes)
    target_nodes = pd.read_parquet(cfg.data.target_nodes)

    # Check overlap between disease_nodes and cold_start_diseases
    disease_node_ids = set(disease_nodes["id"].astype(str).unique())
    cold_start_set = set(cold_start_diseases)
    overlap = disease_node_ids & cold_start_set
    print(f"✅ Overlap between disease nodes and cold start diseases: {len(overlap)}")
    print(f"❗️ Cold start diseases not in disease nodes: {cold_start_set - overlap}")
    print(f"❗️ Disease nodes not in cold start diseases: {len(disease_node_ids - cold_start_set)}")

    all_users = disease_nodes["id"].astype(str).unique()
    all_items = target_nodes["id"].astype(str).unique()
    print(f"✅ Loaded {len(all_users)} users (diseases) and {len(all_items)} items (targets)")

    user_map = {u: idx for idx, u in enumerate(all_users)}
    item_map = {i: idx for idx, i in enumerate(all_items)}

    # Build all_interactions for exhaustive eval
    all_interactions = {}
    for df in [train_df, valid_df, test_df]:
        for u, i in zip(df["user_id"].astype(str), df["item_id"].astype(str)):
            uid = user_map[u]
            iid = item_map[i]
            all_interactions.setdefault(uid, set()).add(iid)


    train_ds, valid_ds, test_ds = create_datasets(train_df, valid_df, test_df, user_map, item_map, all_interactions, cfg)

    # -----------------------
    # Step 3: Build train interactions dict (for ranking exclusion)
    # -----------------------
    train_interactions = {}
    for u, i in zip(train_ds.user.tolist(), train_ds.item.tolist()):
        train_interactions.setdefault(int(u), set()).add(int(i))

    # -----------------------
    # Step 4: Select model
    # -----------------------
    pretrained_embeddings = None  # load here if you want to pass Word2Vec, etc.
    hetero_data = None  # load here if using graph-based model
    # add assert statement to ensure graph object is created under the same config as the training process

    model = initialise_model(cfg, user_map=user_map, item_map=item_map, hetero_data=hetero_data, pretrained_embeddings=pretrained_embeddings)

    lightning_model = BaseRecLightning(
        model,
        lr=cfg.train.lr,
        k=cfg.eval.topk,
        train_interactions=train_interactions,
        loss_type=cfg.model.loss_type,
    )

    # -----------------------
    # Step 5: Dynamic monitor
    # -----------------------
    if cfg.model.loss_type in ["mse", "bce"]:
        monitor_metric, mode = "val_loss", "min"
    else:  # ranking losses
        monitor_metric, mode = f"val_{cfg.eval.valid_metric}", "max"

    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir,
        filename="best_model",
        save_top_k=1,
        monitor=monitor_metric,
        mode=mode,
    )
    earlystop_cb = EarlyStopping(monitor=monitor_metric, patience=3, mode=mode)

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices=1,
        default_root_dir=run_dir,
        log_every_n_steps=10,
        callbacks=[checkpoint_cb, earlystop_cb],
    )

    # -----------------------
    # Step 6: Train
    # -----------------------

    train_loader, valid_loader, test_loader = create_dataloaders(train_ds, valid_ds, test_ds, cfg)
    trainer.fit(lightning_model, train_loader, valid_loader)

    # -----------------------
    # Step 7: Reload best model
    # -----------------------
    best_model_path = checkpoint_cb.best_model_path
    print(f"✅ Best model saved at: {best_model_path}")

    best_model = BaseRecLightning.load_from_checkpoint(
        best_model_path,
        model=model,
        lr=cfg.train.lr,
        k=cfg.eval.topk,
        train_interactions=train_interactions,
        loss_type=cfg.model.loss_type,
    )

    # -----------------------
    # Step 8: Collect predictions
    # -----------------------
    valid_preds = BaseRecLightning.serialise(best_model, valid_loader, run_dir, user_map, item_map, stage="val")
    test_preds = BaseRecLightning.serialise(best_model, test_loader, run_dir, user_map, item_map, stage="test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg)

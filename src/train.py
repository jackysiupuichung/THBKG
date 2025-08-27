#!/usr/bin/env python3
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

def collate_variable(batch):
        collated = {}
        for key in batch[0]:
            if key == "neg_items":
                # keep as list of tensors
                collated[key] = [d[key] for d in batch]
            else:
                collated[key] = torch.stack([d[key] for d in batch])
        return collated

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
    cold_start_targets = []
    if cfg.data.cold_start_file and os.path.exists(cfg.data.cold_start_file):
        with open(cfg.data.cold_start_file) as f:
            cold_start_targets = [line.strip() for line in f if line.strip()]
            

    train_df, valid_df, test_df = temporal_and_cold_split(
        cfg.data.parquet,
        cutoff=cfg.data.cutoff,
        horizon=cfg.data.horizon,
        cold_start_targets=cold_start_targets,
        out_dir=run_dir
    )

    # -----------------------
    # Step 2: Build global maps
    # -----------------------
    all_users = pd.concat([train_df["user_id"], valid_df["user_id"], test_df["user_id"]]).astype(str).unique()
    all_items = pd.concat([train_df["item_id"], valid_df["item_id"], test_df["item_id"]]).astype(str).unique()
    user_map = {u: idx for idx, u in enumerate(all_users)}
    item_map = {i: idx for idx, i in enumerate(all_items)}

    # Build all_interactions for exhaustive eval
    all_interactions = {}
    for df in [train_df, valid_df, test_df]:
        for u, i in zip(df["user_id"].astype(str), df["item_id"].astype(str)):
            uid = user_map[u]
            iid = item_map[i]
            all_interactions.setdefault(uid, set()).add(iid)


    train_ds = InteractionDataset(
        os.path.join(run_dir, "dataframe", "train.csv"),
        user_map,
        item_map,
        num_neg=cfg.model.num_neg if cfg.model.loss_type == "bpr" else 0,
        dynamic=True
    )

    valid_ds = InteractionDataset(
        os.path.join(run_dir, "dataframe", "valid.csv"),
        user_map,
        item_map,
        exhaustive_eval=True,
        all_interactions=all_interactions
    )

    test_ds = InteractionDataset(
        os.path.join(run_dir, "dataframe", "test.csv"),
        user_map,
        item_map,
        exhaustive_eval=True,
        all_interactions=all_interactions
    )

    # -----------------------
    # Step 3: Build train interactions dict (for ranking exclusion)
    # -----------------------
    train_interactions = {}
    for u, i in zip(train_ds.user.tolist(), train_ds.item.tolist()):
        train_interactions.setdefault(int(u), set()).add(int(i))

    # -----------------------
    # Step 4: Select model
    # -----------------------
    if cfg.model.name.lower() == "ncf":
        model = NCF(num_users=train_ds.num_users, num_items=train_ds.num_items, embed_dim=cfg.model.embed_dim)
    elif cfg.model.name.lower() == "graph":
        if not cfg.data.graph or not os.path.exists(cfg.data.graph):
            raise ValueError("Graph model requires a valid graph path in config")
        hetero_data = torch.load(cfg.data.graph)
        print(f"✅ Loaded graph object from {cfg.data.graph}")
        raise NotImplementedError("Graph model integration placeholder")
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

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
    def collect_predictions(dataloader, stage="val"):
        preds, users, items, labels = [], [], [], []
        best_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        best_model.to(device)

        with torch.no_grad():
            for batch in dataloader:
                u, i, l = batch["user_id"].to(device), batch["item_id"].to(device), batch["label"].to(device)
                p = best_model(u, i).squeeze().cpu()
                preds.extend(p.tolist())
                users.extend(u.cpu().tolist())
                items.extend(i.cpu().tolist())
                labels.extend(l.cpu().tolist())

        df = pd.DataFrame({"user_id": users, "item_id": items, "label": labels, "pred": preds})
        out_path = os.path.join(run_dir, f"{stage}_predictions.csv")
        df.to_csv(out_path, index=False)
        print(f"💾 {stage} predictions saved to {out_path}")
        return df

    collect_predictions(valid_loader, stage="val")
    collect_predictions(test_loader, stage="test")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg)

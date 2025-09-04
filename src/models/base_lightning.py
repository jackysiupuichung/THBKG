import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import pandas as pd
import os


# -----------------------
# Shared Base Class
# -----------------------
class BaseRecLightning(pl.LightningModule):
    def __init__(self, model, lr=1e-3, k=[10, 50], loss_type="bce", train_interactions=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.k = k
        self.loss_type = loss_type
        self.train_interactions = train_interactions or {}
        self.val_outputs, self.test_outputs = [], []
        self.save_hyperparameters(ignore=["model", "train_interactions"])

    # -----------------------
    # Loss
    # -----------------------
    def _compute_loss(self, preds, labels, user=None, item=None, neg_items=None):
        if self.loss_type == "mse":
            return F.mse_loss(torch.sigmoid(preds), labels.float())
        elif self.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(preds, labels.float())
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    # -----------------------
    # Metrics
    # -----------------------
    def _ranking_eval(self, outputs, stage="val", num_items=None, forward_fn=None):
        """Compute Recall@K and NDCG@K per user."""
        user_to_gt = {}
        for out in outputs:
            users, items, labels = out["user"], out["item"], out["label"]
            for u, i, l in zip(users.tolist(), items.tolist(), labels.tolist()):
                if l > 0:  # only positives
                    user_to_gt.setdefault(int(u), set()).add(int(i))

        # Assert: must have at least one positive in eval
        assert len(user_to_gt) > 0, f"[{stage}] No positives found in dataset! Check your split or labels."

        recalls = {K: [] for K in self.k}
        ndcgs = {K: [] for K in self.k}

        for u, gt_items in user_to_gt.items():
            all_items = torch.arange(num_items, device=self.device)

            # exclude training items
            exclude = self.train_interactions.get(u, set())
            mask = torch.ones(num_items, dtype=torch.bool, device=self.device)
            if exclude:
                mask[list(exclude)] = False
            candidate_items = all_items[mask]

            user_tensor = torch.full((len(candidate_items),), u, device=self.device, dtype=torch.long)
            scores = forward_fn(user_tensor, candidate_items).squeeze()

            _, topk_idx = torch.topk(scores, max(self.k))
            topk_items = candidate_items[topk_idx].cpu().tolist()

            for K in self.k:
                hits = sum([1 for i in topk_items[:K] if i in gt_items])
                recalls[K].append(hits / len(gt_items))

                dcg = sum(
                    1.0 / torch.log2(torch.tensor(rank + 1.0))
                    for rank, i in enumerate(topk_items[:K], start=1)
                    if i in gt_items
                )
                idcg = sum(
                    1.0 / torch.log2(torch.tensor(r + 1.0))
                    for r in range(1, min(len(gt_items), K) + 1)
                )
                ndcgs[K].append((dcg / idcg).item() if idcg > 0 else 0.0)

        for K in self.k:
            self.log(f"{stage}_Recall@{K}", torch.tensor(recalls[K]).mean(), prog_bar=True)
            self.log(f"{stage}_NDCG@{K}", torch.tensor(ndcgs[K]).mean(), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_train_epoch_start(self):
        """Resample negatives in the training dataset at the start of every epoch."""
        train_loader = self.trainer.train_dataloader
        if train_loader is not None:
            ds = getattr(train_loader, "dataset", None)
            if ds is not None and hasattr(ds, "resample"):
                ds.resample()
                print("🔄 Resampled negatives for new epoch")


# -----------------------
# NCF Wrapper
# -----------------------
class NCFRecLightning(BaseRecLightning):
    def forward(self, user, item):
        return self.model(user, item)

    def training_step(self, batch, batch_idx):
        user, item, label = batch["user_id"], batch["item_id"], batch["label"]
        preds = self(user, item).squeeze()
        loss = self._compute_loss(preds, label, user=user, item=item, neg_items=batch.get("neg_items"))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        user, item, label = batch["user_id"], batch["item_id"], batch["label"]
        preds = self(user, item).squeeze()
        loss = self._compute_loss(preds, label, user=user, item=item, neg_items=batch.get("neg_items"))
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.val_outputs.append({"user": user.cpu(), "item": item.cpu(), "label": label.cpu()})
        return loss

    def test_step(self, batch, batch_idx):
        self.test_outputs.append(
            {"user": batch["user_id"].cpu(), "item": batch["item_id"].cpu(), "label": batch["label"].cpu()}
        )
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Called by trainer.predict"""
        user = batch["user_id"].to(self.device)
        item = batch["item_id"].to(self.device)
        return self(user, item).squeeze()

    def on_validation_epoch_end(self):
        num_items = self.model.item_emb.num_embeddings
        self._ranking_eval(self.val_outputs, stage="val", num_items=num_items, forward_fn=self.forward)
        self.val_outputs.clear()

    def on_test_epoch_end(self):
        num_items = self.model.item_emb.num_embeddings
        self._ranking_eval(self.test_outputs, stage="test", num_items=num_items, forward_fn=self.forward)
        self.test_outputs.clear()


# -----------------------
# Graph Wrapper
# -----------------------
class GraphRecLightning(BaseRecLightning):
    def __init__(self, model, lr=1e-3,
                 supervision_src_type=None,
                 supervision_relation_type=None,
                 supervision_dst_type=None,
                 k=[10, 50],
                 **kwargs):
        super().__init__(model, lr=lr, k=k, **kwargs)

        self.supervision_etype = (
            supervision_src_type,
            supervision_relation_type,
            supervision_dst_type,
        )
        self.val_outputs = []
        self.test_outputs = []

    # -----------------------
    # Helpers
    # -----------------------
    def _parse_batch(self, batch):
        etype = self.supervision_etype
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict

        # ✅ Build edge_attr_dict safely
        edge_attr_dict = {
            et: batch[et].edge_attr
            for et in batch.edge_types
            if hasattr(batch[et], "edge_attr")
        }
        print(f"[DEBUG] edge_attr_dict keys: {list(edge_attr_dict.keys())}")

        # ✅ Ensure all node types have features, else error
        for ntype in batch.node_types:
            if ntype not in x_dict or x_dict[ntype] is None:
                print(f"[ERROR] Missing features for node type '{ntype}'")
                raise ValueError(
                    f"❌ Missing features for node type '{ntype}'. "
                    f"Expected batch['{ntype}'].x to exist."
                )
            else:
                print(f"[DEBUG] Features found for node type '{ntype}', shape: {x_dict[ntype].shape}")

        src_ids, dst_ids = batch[etype].edge_label_index
        labels = batch[etype].edge_label.float()
        print(f"[DEBUG] src_ids shape: {src_ids.shape}, dst_ids shape: {dst_ids.shape}, labels shape: {labels.shape}")
        return x_dict, edge_index_dict, edge_attr_dict, (src_ids, dst_ids), labels

    # -----------------------
    # Forward
    # -----------------------
    def forward(self, batch):
        x_dict, edge_index_dict, edge_attr_dict, pairs, _ = self._parse_batch(batch)
        return self.model(x_dict, edge_index_dict, pairs=pairs, edge_attr_dict=edge_attr_dict)

    # -----------------------
    # Training
    # -----------------------
    def training_step(self, batch, batch_idx):
        preds = self(batch).squeeze()
        _, _, _, _, labels = self._parse_batch(batch)
        loss = self._compute_loss(preds, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # -----------------------
    # Validation
    # -----------------------
    def validation_step(self, batch, batch_idx):
        preds = self(batch).squeeze()
        _, _, _, pairs, labels = self._parse_batch(batch)
        loss = self._compute_loss(preds, labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.val_outputs.append(
            {"user": pairs[0].cpu(), "item": pairs[1].cpu(), "label": labels.cpu()}
        )
        return loss

    # -----------------------
    # Test
    # -----------------------
    def test_step(self, batch, batch_idx):
        _, _, _, pairs, labels = self._parse_batch(batch)
        self.test_outputs.append(
            {"user": pairs[0].cpu(), "item": pairs[1].cpu(), "label": labels.cpu()}
        )

    # -----------------------
    # Prediction
    # -----------------------
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch = batch.to(self.device)
        return self(batch).squeeze()

    # -----------------------
    # Epoch End Hooks
    # -----------------------
    def on_validation_epoch_end(self):
        num_items = self.model.embeddings[self.model.pair_dst_type].num_embeddings
        self._ranking_eval(self.val_outputs, stage="val", num_items=num_items, forward_fn=self.forward)
        self.val_outputs.clear()

    def on_test_epoch_end(self):
        num_items = self.model.embeddings[self.model.pair_dst_type].num_embeddings
        self._ranking_eval(self.test_outputs, stage="test", num_items=num_items, forward_fn=self.forward)
        self.test_outputs.clear()

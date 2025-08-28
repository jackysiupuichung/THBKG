import pytorch_lightning as pl
import torch
import pandas as pd
import os
import torch.nn.functional as F


class BaseRecLightning(pl.LightningModule):
    def __init__(self, model, lr=1e-3, k=[10, 50], train_interactions=None, loss_type="bce"):
        super().__init__()
        self.model = model
        self.lr = lr
        self.k = k
        self.loss_type = loss_type
        self.train_interactions = train_interactions or {}
        self.val_outputs = []
        self.test_outputs = []
        self.save_hyperparameters(ignore=["model", "train_interactions"])

    def forward(self, user, item):
        return self.model(user, item)
    
    def on_train_epoch_start(self):
        if hasattr(self.trainer, "train_dataloader"):
            train_loader = self.trainer.train_dataloader
            if hasattr(train_loader.dataset, "resample"):
                train_loader.dataset.resample()

    def training_step(self, batch, batch_idx):
        user, item, label = batch["user_id"], batch["item_id"], batch["label"]

        if self.loss_type in ["bce", "mse"]:
            preds = self(user, item).squeeze()
            if self.loss_type == "mse":
                loss = F.mse_loss(torch.sigmoid(preds), label.float())
            else:  # BCE
                loss = F.binary_cross_entropy_with_logits(preds, label.float())

        elif self.loss_type == "bpr":
            if "neg_items" not in batch:
                raise ValueError("❌ Dataset must provide `neg_items` for BPR loss")

            pos_preds = self(user, item).unsqueeze(1)  # [batch, 1]
            neg_items = batch["neg_items"]  # [batch, num_neg]

            # Expand users for negatives
            user_exp = user.unsqueeze(1).expand(-1, neg_items.size(1))  # [batch, num_neg]
            neg_preds = self(user_exp.reshape(-1), neg_items.reshape(-1)).view(user.size(0), -1)

            loss = -torch.log(torch.sigmoid(pos_preds - neg_preds) + 1e-8).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        self.log("train_loss", loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        user, item, label = batch["user_id"], batch["item_id"], batch["label"]

        if self.loss_type in ["bce", "mse"]:
            preds = self(user, item).squeeze()
            if self.loss_type == "mse":
                loss = F.mse_loss(torch.sigmoid(preds), label.float())
            else:  # BCE
                loss = F.binary_cross_entropy_with_logits(preds, label.float())

        elif self.loss_type == "bpr":
            # Pairwise validation loss: sample negatives per user
            pos_preds = self(user, item).squeeze()
            neg_items = torch.randint(0, self.model.item_emb.num_embeddings, item.shape, device=item.device)
            neg_preds = self(user, neg_items).squeeze()
            loss = -torch.log(torch.sigmoid(pos_preds - neg_preds) + 1e-8).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Always log val_loss so callbacks work
        self.log("val_loss", loss, prog_bar=False, on_epoch=True, on_step=False)

        # Still collect outputs for ranking evaluation
        self.val_outputs.append({
            "user": user.detach().cpu(),
            "item": item.detach().cpu(),
            "label": label.detach().cpu(),
        })


    def on_validation_epoch_end(self):
        self._ranking_eval(self.val_outputs, stage="val")
        self.val_outputs.clear()

    def test_step(self, batch, batch_idx):
        self.test_outputs.append({
            "user": batch["user_id"].detach().cpu(),
            "item": batch["item_id"].detach().cpu(),
            "label": batch["label"].detach().cpu(),
        })

    def on_test_epoch_end(self):
        self._ranking_eval(self.test_outputs, stage="test")
        self.test_outputs.clear()

    def _ranking_eval(self, outputs, stage="val"):
        """Compute Recall@K and NDCG@K per user."""
        user_to_gt = {}
        for out in outputs:
            users = out["user"].tolist()
            items = out["item"].tolist()
            labels = out["label"].tolist()
            for u, i, l in zip(users, items, labels):
                if l > 0:  # only positives
                    user_to_gt.setdefault(int(u), set()).add(int(i))

        recalls = {K: [] for K in self.k}
        ndcgs = {K: [] for K in self.k}

        num_items = self.model.item_emb.num_embeddings

        for u, gt_items in user_to_gt.items():
            all_items = torch.arange(num_items, device=self.device)

            # Exclude training items
            exclude = self.train_interactions.get(u, set())
            mask = torch.ones(num_items, dtype=torch.bool, device=self.device)
            if exclude:
                mask[list(exclude)] = False
            candidate_items = all_items[mask]

            user_tensor = torch.full((len(candidate_items),), u, device=self.device, dtype=torch.long)
            scores = self(user_tensor, candidate_items).squeeze()

            _, topk_idx = torch.topk(scores, max(self.k))
            topk_items = candidate_items[topk_idx].cpu().tolist()

            for K in self.k:
                hits = sum([1 for i in topk_items[:K] if i in gt_items])
                recalls[K].append(hits / len(gt_items))

                dcg = 0.0
                for rank, i in enumerate(topk_items[:K], start=1):
                    if i in gt_items:
                        dcg += 1.0 / torch.log2(torch.tensor(rank + 1.0))
                idcg = sum(
                    1.0 / torch.log2(torch.tensor(r + 1.0)) for r in range(1, min(len(gt_items), K) + 1)
                )
                ndcgs[K].append((dcg / idcg).item() if idcg > 0 else 0.0)

        for K in self.k:
            self.log(f"{stage}_Recall@{K}", torch.tensor(recalls[K]).mean(), prog_bar=True)
            self.log(f"{stage}_NDCG@{K}", torch.tensor(ndcgs[K]).mean(), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    


    @staticmethod
    def predict(best_model, dataloader):
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
        return preds, users, items, labels

    @staticmethod
    def serialise(best_model, dataloader, run_dir, user_map, item_map, stage):
        preds, users, items, labels = BaseRecLightning.predict(best_model, dataloader)
        rev_user_map = {v: k for k, v in user_map.items()}
        rev_item_map = {v: k for k, v in item_map.items()}

        # Map user/item ids back to names
        user_names = [rev_user_map.get(uid, uid) for uid in users]
        item_names = [rev_item_map.get(iid, iid) for iid in items]

        df = pd.DataFrame({"user_id": user_names, "item_id": item_names, "label": labels, "pred": preds})
        pred_dir = os.path.join(run_dir, f"predictions")
        os.makedirs(pred_dir, exist_ok=True)
        out_path = os.path.join(pred_dir, f"{stage}_predictions.csv")
        df.to_csv(out_path, index=False)
        print(f"💾 {stage} predictions saved to {out_path}")
        return df

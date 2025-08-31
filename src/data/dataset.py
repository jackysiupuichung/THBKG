import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch_geometric.loader import LinkNeighborLoader

from src.models.utils import collate_variable

class UniformNegSampler:
    """Uniformly sample negatives from item space (targets)."""

    def __init__(self, num_items, num_neg=1, seed=42):
        self.num_items = num_items
        self.num_neg = num_neg
        self.rng = np.random.default_rng(seed)

    def sample(self, positives=None):
        """Sample negatives excluding given positives if provided."""
        if positives is not None and len(positives) > 0:
            candidates = list(set(range(self.num_items)) - set(positives))
            return self.rng.choice(
                candidates,
                size=self.num_neg,
                replace=len(candidates) < self.num_neg
            )
        return self.rng.integers(low=0, high=self.num_items, size=self.num_neg)


class InteractionDataset(Dataset):
    """
    Dataset for recommendation (NCF or Graph).
    Supports per-epoch dynamic negative resampling and exhaustive eval.

    Args:
        df: DataFrame with ["user_id", "item_id", "label"]
        user_map: dict {raw_user_id -> index}
        item_map: dict {raw_item_id -> index}
        num_neg: negatives per positive (for training)
        dynamic: resample negatives each epoch
        exhaustive_eval: build all negatives for each user (valid/test)
        all_interactions: dict {user_idx: set(item_idx)} of all known positives
        seed: random seed
    """

    def __init__(self, df, user_map, item_map,
                 num_neg=0, dynamic=False, exhaustive_eval=False,
                 all_interactions=None, seed=42):

        # === Base interactions (positives) ===
        self.df = df.reset_index(drop=True)
        labels = np.array(pd.to_numeric(self.df["label"], errors="coerce"))

        self.user = torch.tensor([user_map[u] for u in self.df["user_id"].astype(str)], dtype=torch.long)
        self.item = torch.tensor([item_map[i] for i in self.df["item_id"].astype(str)], dtype=torch.long)
        self.label = torch.tensor(labels, dtype=torch.float)

        self.user_map = user_map
        self.item_map = item_map
        self.num_users = len(user_map)
        self.num_items = len(item_map)

        self.num_neg = num_neg
        self.dynamic = dynamic
        self.exhaustive_eval = exhaustive_eval
        self.rng = np.random.default_rng(seed)

        self.all_interactions = all_interactions if all_interactions else {}
        self.sampler = UniformNegSampler(self.num_items, num_neg, seed) if num_neg > 0 else None
        self.neg_items = None

        if self.exhaustive_eval:
            self._build_exhaustive_negatives()
        elif self.sampler:
            self.resample()

        print("✅ InteractionDataset built:", self.dataset_description())

    # -----------------------
    # Negatives
    # -----------------------
    def resample(self):
        """Resample negatives for each positive user (train only)."""
        if not self.sampler:
            return
        negs = []
        for u in self.user.tolist():
            positives = self.all_interactions.get(u, set()) if self.all_interactions else None
            sampled = self.sampler.sample(positives)
            negs.append(sampled)
        self.neg_items = torch.tensor(negs, dtype=torch.long)

    def _build_exhaustive_negatives(self, exclude_positives=True):
        """Build full candidate set for evaluation."""
        all_items = list(self.item_map.values())
        negs = []
        for u in self.user.tolist():
            positives = self.all_interactions.get(u, set())
            if exclude_positives:
                candidates = [i for i in all_items if i not in positives]
            else:
                candidates = all_items
            negs.append(torch.tensor(candidates, dtype=torch.long))
        self.neg_items = negs

    # -----------------------
    # Dataset API
    # -----------------------
    def __len__(self):
        return len(self.user)

    def __getitem__(self, idx):
        batch = {
            "user_id": self.user[idx],
            "item_id": self.item[idx],
            "label": self.label[idx],
        }
        if self.num_neg > 0 and self.neg_items is not None and not self.exhaustive_eval:
            batch["neg_items"] = self.neg_items[idx]
        elif self.exhaustive_eval:
            batch["neg_items"] = self.neg_items[idx]
        return batch

    # -----------------------
    # Loader Builders
    # -----------------------
    def build_ncf_loader(self, batch_size=512, shuffle=True):
        """Standard PyTorch DataLoader for NCF training/eval."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_variable)

    def build_graph_loader(self, hetero_graph, batch_size=1024, num_neighbors=[15, 10], shuffle=True):
        """Build LinkNeighborLoader from this dataset (positives + negatives)."""
        edge_type = ("diseases", "clinical_trial", "targets")

        users = self.user.tolist()
        items = self.item.tolist()
        labels = self.label.tolist()

        if hasattr(self, "neg_items") and self.neg_items is not None:
            new_users, new_items, new_labels = [], [], []
            for u, i, l, negs in zip(users, items, labels, self.neg_items):
                new_users.append(u)
                new_items.append(i)
                new_labels.append(1.0)  # positive
                if isinstance(negs, torch.Tensor):
                    for n in negs.tolist():
                        new_users.append(u)
                        new_items.append(n)
                        new_labels.append(0.0)
                elif isinstance(negs, list):  # exhaustive mode
                    for n in negs:
                        new_users.append(u)
                        new_items.append(int(n))
                        new_labels.append(0.0)
            users, items, labels = new_users, new_items, new_labels

        edge_label_index = torch.tensor([users, items], dtype=torch.long)
        edge_label = torch.tensor(labels, dtype=torch.float)

        return LinkNeighborLoader(
            data=hetero_graph,
            edge_label_index=(edge_type, edge_label_index),
            edge_label=edge_label,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
            neg_sampling=False,  # already handled in dataset
            collate_fn=collate_variable
        )

    # -----------------------
    # Info
    # -----------------------
    def dataset_description(self):
        desc = {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "num_positive_interactions": len(self.df),
            "num_unique_users": self.df["user_id"].nunique(),
            "num_unique_items": self.df["item_id"].nunique(),
            "num_neg_per_pos": self.num_neg,
            "dynamic_neg_sampling": self.dynamic,
            "exhaustive_eval": self.exhaustive_eval,
        }
        return desc

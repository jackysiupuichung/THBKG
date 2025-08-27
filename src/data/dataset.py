import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class UniformNegSampler:
    def __init__(self, num_items, num_neg=1, seed=42):
        self.num_items = num_items
        self.num_neg = num_neg
        self.rng = np.random.default_rng(seed)

    def sample(self, user, positives=None):
        """Sample negatives for one user, excluding positives if provided"""
        if positives:
            candidates = list(set(range(self.num_items)) - set(positives))
            return self.rng.choice(candidates, size=self.num_neg, replace=len(candidates) < self.num_neg)
        return self.rng.integers(low=0, high=self.num_items, size=self.num_neg)



class InteractionDataset(Dataset):
    def __init__(self, csv_path, user_map, item_map, num_neg=0, dynamic=False, exhaustive_eval=False, all_interactions=None, seed=42):
        """
        Args:
            csv_path: path to user-item interactions
            user_map: dict {user_id -> int index}
            item_map: dict {item_id -> int index}
            num_neg: number of negatives per positive (for training)
            dynamic: if True, resample negatives each epoch (train only)
            exhaustive_eval: if True, build full user->all_items minus positives set (valid/test only)
            all_interactions: dict {user: set(items)} containing ALL known positives (train+valid+test) for exclusion
        """
        df = pd.read_csv(csv_path)

        # Ensure numeric labels
        labels = pd.to_numeric(df["label"], errors="coerce")
        if labels.isna().any():
            bad_rows = df[labels.isna()]
            raise ValueError(f"❌ Found NaN in label column in {csv_path} ({len(bad_rows)} rows).")

        self.user = torch.tensor([user_map[u] for u in df["user_id"].astype(str)], dtype=torch.long)
        self.item = torch.tensor([item_map[i] for i in df["item_id"].astype(str)], dtype=torch.long)
        self.label = torch.tensor(labels.astype(float).values, dtype=torch.float)

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

    def resample(self):
        if not self.sampler:
            return
        negs = []
        for u in self.user.tolist():
            positives = self.all_interactions.get(u, set()) if self.all_interactions else None
            sampled = self.sampler.sample(u, positives)
            negs.append(sampled)
        self.neg_items = torch.tensor(negs, dtype=torch.long)

    def _build_exhaustive_negatives(self):
        """For validation/test: negatives = all non-positive items."""
        negs = []
        for u in self.user.tolist():
            positives = self.all_interactions.get(u, set())
            # all items not in positives
            negatives = [i for i in range(self.num_items) if i not in positives]
            negs.append(torch.tensor(negatives, dtype=torch.long))
        self.neg_items = negs  # list of tensors, variable size per user

    def __len__(self):
        return len(self.user)

    def __getitem__(self, idx):
        batch = {
            "user_id": self.user[idx],
            "item_id": self.item[idx],
            "label": self.label[idx],
        }
        if self.num_neg > 0 and self.neg_items is not None and not self.exhaustive_eval:
            # training mode (sampled negatives)
            batch["neg_items"] = self.neg_items[idx]
        elif self.exhaustive_eval:
            # validation/test mode (all negatives)
            batch["neg_items"] = self.neg_items[idx]  # list of all non-positive items
        return batch

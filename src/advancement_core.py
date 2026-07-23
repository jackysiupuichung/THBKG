#!/usr/bin/env python3
"""
Train HGT for clinical trial advancement link prediction.

- Context graph: all non-advancement edges, full temporal structure (no collapse by default)
- Train/val split: chronological by year (train: <= 2010, val: 2011–2015)
- Test: original test_dataset rows (transition_year >= 2016)
- Task: binary link prediction (outcome 0/1), BCE loss
- Output: best_model.pt, results.yaml, test_predictions.parquet
"""

import os
import sys
from pathlib import Path

# Allow MPS to fall back to CPU for unsupported ops (e.g. scatter_reduce)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, brier_score_loss, log_loss,
    precision_recall_curve,
)
from torch_geometric.loader import LinkNeighborLoader
import wandb


class TeeLogger:
    """Mirrors stdout to both the terminal and a log file."""
    def __init__(self, log_path):
        self._terminal = sys.stdout
        self._log = open(log_path, "a", buffering=1)  # line-buffered

    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        self._log.close()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.temporal_loader import (
    ADV_ETYPE,
    TRAIN_YEAR_MAX,
    TEST_YEAR_MIN,
    build_context_graph,
    build_edge_time_dict as _build_edge_time_dict,
    load_event_graph,
    split_advancement_edges,
    to_time_agnostic,
)
from src.models.utils import build_model


def focal_loss(logits, labels, pos_weight=None, gamma=2.0):
    """Binary focal loss with optional pos_weight for class imbalance."""
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * labels + (1 - probs) * (1 - labels)
    loss = bce * (1 - p_t) ** gamma
    return loss.mean()


def run_epoch(model, loader, optimizer, device, train=True, edge_feat_cols=(0, 1), pos_weight=None, focal_gamma=None):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            edge_time_dict = _build_edge_time_dict(batch, ADV_ETYPE)
            out = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch[ADV_ETYPE].edge_label_index,
                src_type="target",
                dst_type="disease",
                edge_time_dict=edge_time_dict,
                edge_feat_dict={
                    et: batch[et].edge_attr[:, edge_feat_cols]
                    for et in batch.edge_types
                    if et != ADV_ETYPE and hasattr(batch[et], 'edge_attr')
                    and batch[et].edge_attr is not None
                },
            )
            labels = batch[ADV_ETYPE].edge_label.float()
            logits = out
            pw = pos_weight.to(device) if pos_weight is not None else None
            if focal_gamma is not None:
                loss = focal_loss(logits, labels, pos_weight=pw, gamma=focal_gamma)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """
    Compute the full evaluation metric suite.

    Parameters
    ----------
    labels : int array of shape [N]  (0 / 1)
    scores : float array of shape [N]  (predicted probabilities)
    """
    n_samples   = len(labels)
    n_positives = int(labels.sum())
    balance     = n_positives / n_samples if n_samples > 0 else float("nan")

    if n_positives == 0 or n_positives == n_samples:
        nan = float("nan")
        return {k: nan for k in [
            "n_samples", "n_positives", "balance",
            "precision", "recall", "f1", "mcc",
            "roc_auc", "average_precision", "brier", "balanced_mae", "log_loss",
            "precision@10", "precision@30", "precision@50",
            "average_precision@10", "average_precision@30", "average_precision@50", "average_precision@100",
            "recall@10", "recall@30", "recall@50",
            "rs@10", "rs@20", "rs@30", "rs@50", "rs@90", "rs@100",
        ]} | {"n_samples": n_samples, "n_positives": n_positives, "balance": balance}

    preds = (scores >= 0.5).astype(int)

    # Threshold-based
    precision   = precision_score(labels, preds, zero_division=0)
    recall      = recall_score(labels, preds, zero_division=0)
    f1          = f1_score(labels, preds, zero_division=0)
    mcc         = matthews_corrcoef(labels, preds)

    # Ranking / probabilistic
    roc_auc            = roc_auc_score(labels, scores)
    avg_precision      = average_precision_score(labels, scores)
    brier              = brier_score_loss(labels, scores)
    ll                 = log_loss(labels, scores)
    # Balanced MAE: mean |score - label| weighted so pos/neg classes contribute equally
    pos_mae = np.abs(scores[labels == 1] - 1).mean() if n_positives > 0 else float("nan")
    neg_mae = np.abs(scores[labels == 0] - 0).mean() if (n_samples - n_positives) > 0 else float("nan")
    balanced_mae = (pos_mae + neg_mae) / 2

    # Rank-based @K metrics
    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]

    def _precision_at_k(k):
        top = labels_sorted[:k]
        return top.sum() / k if k > 0 else float("nan")

    def _recall_at_k(k):
        top = labels_sorted[:k]
        return top.sum() / n_positives if n_positives > 0 else float("nan")

    def _ap_at_k(k):
        top_labels  = labels_sorted[:k]
        top_scores  = scores[order][:k]
        if top_labels.sum() == 0:
            return 0.0
        return average_precision_score(top_labels, top_scores)

    def _rs_at_k(k):
        # Rank-based top-k, matching metrics.rs_ta_mean_at_k: exposed is
        # exactly k items via a stable descending sort, ties broken by
        # original index. A threshold-mask (scores >= s[k-1]) would silently
        # expand exposed past k whenever the k-th score ties with anything
        # below it, distorting the metric (and the ES signal that uses rs@100).
        k = min(k, n_samples)
        rank_order   = np.argsort(-scores, kind="stable")
        exposed_mask = np.zeros(n_samples, dtype=bool)
        exposed_mask[rank_order[:k]] = True
        control_mask = ~exposed_mask
        if control_mask.sum() == 0:
            return float("nan")
        p_exposed = labels[exposed_mask].sum() / k
        p_control = labels[control_mask].sum() / control_mask.sum()
        if p_control == 0:                          # undefined when no positives in control
            return float("nan")
        return float(p_exposed / p_control)

    metrics = {
        "n_samples":           n_samples,
        "n_positives":         n_positives,
        "balance":             balance,
        "precision":           precision,
        "recall":              recall,
        "f1":                  f1,
        "mcc":                 mcc,
        "roc_auc":             roc_auc,
        "average_precision":   avg_precision,
        "brier":               brier,
        "balanced_mae":        balanced_mae,
        "log_loss":            ll,
        "precision@10":        _precision_at_k(10),
        "precision@30":        _precision_at_k(30),
        "precision@50":        _precision_at_k(50),
        "average_precision@10":  _ap_at_k(10),
        "average_precision@30":  _ap_at_k(30),
        "average_precision@50":  _ap_at_k(50),
        "average_precision@100": _ap_at_k(100),
        "recall@10":           _recall_at_k(10),
        "recall@30":           _recall_at_k(30),
        "recall@50":           _recall_at_k(50),
        "rs@10":               _rs_at_k(10),
        "rs@20":               _rs_at_k(20),
        "rs@30":               _rs_at_k(30),
        "rs@50":               _rs_at_k(50),
        "rs@90":               _rs_at_k(90),
        "rs@100":              _rs_at_k(100),
    }
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, edge_feat_cols=(0, 1), pos_weight=None, focal_gamma=None):
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        edge_time_dict = _build_edge_time_dict(batch, ADV_ETYPE)
        out = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch[ADV_ETYPE].edge_label_index,
            src_type="target",
            dst_type="disease",
            edge_time_dict=edge_time_dict,
            edge_feat_dict={
                et: batch[et].edge_attr[:, edge_feat_cols]
                for et in batch.edge_types
                if et != ADV_ETYPE and hasattr(batch[et], 'edge_attr')
                and batch[et].edge_attr is not None
            },
        )
        all_logits.append(out.cpu())
        all_labels.append(batch[ADV_ETYPE].edge_label.cpu())

    logits_t = torch.cat(all_logits)
    labels_t = torch.cat(all_labels).float()

    pw = pos_weight.cpu() if pos_weight is not None else None
    if focal_gamma is not None:
        val_loss = focal_loss(logits_t, labels_t, pos_weight=pw, gamma=focal_gamma).item()
    else:
        val_loss = F.binary_cross_entropy_with_logits(logits_t, labels_t, pos_weight=pw).item()

    logits = logits_t.numpy()
    labels = (labels_t > 0).numpy().astype(int)
    nan_mask = np.isnan(logits)
    if nan_mask.any():
        print(f"WARNING: {nan_mask.sum()} NaN logits detected, dropping them.")
        logits = logits[~nan_mask]
        labels = labels[~nan_mask]
    scores = expit(logits)

    metrics = compute_metrics(labels, scores)
    metrics["val_loss"] = val_loss
    return metrics


@torch.no_grad()
def predict_test(model, context, edge_index, edge_labels, edge_times, num_neighbors, batch_size, device, edge_feat_cols=(0, 1), num_workers=0):
    """Score test edges using temporally-constrained subgraphs."""
    model.eval()
    loader = LinkNeighborLoader(
        data=context,
        num_neighbors=num_neighbors,
        edge_label_index=(ADV_ETYPE, edge_index),
        edge_label=edge_labels,
        edge_label_time=edge_times,
        time_attr="edge_time",
        temporal_strategy="last",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        edge_time_dict = _build_edge_time_dict(batch, ADV_ETYPE)
        out = model(
            batch.x_dict, batch.edge_index_dict,
            batch[ADV_ETYPE].edge_label_index,
            src_type="target", dst_type="disease",
            edge_time_dict=edge_time_dict,
            edge_feat_dict={
                et: batch[et].edge_attr[:, edge_feat_cols]
                for et in batch.edge_types
                if et != ADV_ETYPE and hasattr(batch[et], 'edge_attr')
                and batch[et].edge_attr is not None
            },
            edge_label_time=getattr(batch[ADV_ETYPE], "edge_label_time", None),
        )
        all_logits.append(out.cpu())
        all_labels.append(batch[ADV_ETYPE].edge_label.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = (torch.cat(all_labels) > 0).numpy().astype(int)
    scores = expit(logits)
    return scores, labels

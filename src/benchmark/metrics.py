#!/usr/bin/env python3
"""
Evaluation metrics for link prediction / recommendation.

Implements ranking metrics: Recall@k, Precision@k, NDCG@k, MRR.
"""

import torch
import numpy as np
from typing import Dict, List, Optional


def recall_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> float:
    """
    Compute Recall@k.
    
    Args:
        scores: Predicted scores [num_items]
        labels: Binary labels [num_items]
        k: Top-k
        
    Returns:
        Recall@k score
    """
    if labels.sum() == 0:
        return 0.0
    
    # Get top-k indices
    _, top_k_indices = torch.topk(scores, k=min(k, len(scores)))
    
    # Count relevant items in top-k
    relevant_in_top_k = labels[top_k_indices].sum().item()
    total_relevant = labels.sum().item()
    
    return relevant_in_top_k / total_relevant


def precision_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> float:
    """
    Compute Precision@k.
    
    Args:
        scores: Predicted scores [num_items]
        labels: Binary labels [num_items]
        k: Top-k
        
    Returns:
        Precision@k score
    """
    # Get top-k indices
    _, top_k_indices = torch.topk(scores, k=min(k, len(scores)))
    
    # Count relevant items in top-k
    relevant_in_top_k = labels[top_k_indices].sum().item()
    
    return relevant_in_top_k / k


def ndcg_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> float:
    """
    Compute NDCG@k (Normalized Discounted Cumulative Gain).
    
    Args:
        scores: Predicted scores [num_items]
        labels: Binary labels [num_items]
        k: Top-k
        
    Returns:
        NDCG@k score
    """
    if labels.sum() == 0:
        return 0.0
    
    # Get top-k indices
    _, top_k_indices = torch.topk(scores, k=min(k, len(scores)))
    
    # DCG: sum of (relevance / log2(rank + 1))
    relevance = labels[top_k_indices].float()
    ranks = torch.arange(1, len(top_k_indices) + 1, dtype=torch.float32)
    dcg = (relevance / torch.log2(ranks + 1)).sum().item()
    
    # IDCG: ideal DCG (all relevant items at top)
    ideal_relevance, _ = torch.sort(labels.float(), descending=True)
    ideal_relevance = ideal_relevance[:k]
    ideal_ranks = torch.arange(1, len(ideal_relevance) + 1, dtype=torch.float32)
    idcg = (ideal_relevance / torch.log2(ideal_ranks + 1)).sum().item()
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def ndcg_ta_mean_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    ta_per_item: List[List[str]],
    k: int,
    primary_tas: Optional[List[str]] = None,
) -> float:
    """
    TA-grouped NDCG@k. For each therapeutic area, restrict to the items
    belonging to that TA, compute NDCG@k on that subset, then average across
    TAs (mean-of-ratios over therapeutic areas, mirroring the eval-time RR).

    Items can belong to multiple TAs (one disease → many TAs); each TA gets
    counted once. TAs with zero positives in the val set are skipped.

    Args:
        scores: [N] predicted scores
        labels: [N] binary labels
        ta_per_item: list of length N; each entry a list of TA names
        k: top-k cutoff
        primary_tas: optional whitelist of TAs to average over. If None, uses
            all TAs that appear in `ta_per_item`.

    Returns:
        Mean NDCG@k across TAs, or 0.0 if no TA had any positives.
    """
    if labels.sum() == 0:
        return 0.0

    by_ta_idx: Dict[str, List[int]] = {}
    for i, tas in enumerate(ta_per_item):
        for ta in tas:
            if primary_tas is not None and ta not in primary_tas:
                continue
            by_ta_idx.setdefault(ta, []).append(i)

    if not by_ta_idx:
        return 0.0

    vals = []
    for ta, idx_list in by_ta_idx.items():
        idx = torch.as_tensor(idx_list, dtype=torch.long)
        ta_scores = scores[idx]
        ta_labels = labels[idx]
        if ta_labels.sum() == 0:
            continue
        vals.append(ndcg_at_k(ta_scores, ta_labels, k))

    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _rs_ta_values_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    ta_per_item: List[List[str]],
    k: int,
    primary_tas: Optional[List[str]] = None,
) -> List[float]:
    """Per-TA relative-success-at-K values (the list aggregated by mean/median).

    Within each TA, take the top-k items by score, then
    RS = P(pos | exposed) / P(pos | control). Skips TAs with fewer than k items
    or zero control positives (RS undefined). Items can belong to multiple TAs.
    Returns the list of qualifying per-TA RS values (possibly empty).
    """
    if labels.sum() == 0:
        return []

    by_ta_idx: Dict[str, List[int]] = {}
    for i, tas in enumerate(ta_per_item):
        for ta in tas:
            if primary_tas is not None and ta not in primary_tas:
                continue
            by_ta_idx.setdefault(ta, []).append(i)

    if not by_ta_idx:
        return []

    scores_np = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)

    vals = []
    for ta, idx_list in by_ta_idx.items():
        if len(idx_list) <= k:
            continue
        idx = np.asarray(idx_list, dtype=np.int64)
        ta_scores = scores_np[idx]
        ta_labels = labels_np[idx]

        # Rank-based top-k (standard recommender-systems convention): ties broken
        # by original index via stable sort, so exposed is exactly k items
        # regardless of score ties. A threshold-mask (scores >= s[k-1]) would
        # silently expand exposed past k whenever the k-th score ties with
        # anything below it, collapsing control to empty and skipping the TA.
        order = np.argsort(-ta_scores, kind="stable")
        exposed_mask = np.zeros(len(ta_scores), dtype=bool)
        exposed_mask[order[:k]] = True
        control_mask = ~exposed_mask
        p_exposed = ta_labels[exposed_mask].sum() / k
        p_control = ta_labels[control_mask].sum() / control_mask.sum()
        if p_control == 0:
            continue
        vals.append(float(p_exposed / p_control))

    return vals


def rs_ta_mean_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    ta_per_item: List[List[str]],
    k: int,
    primary_tas: Optional[List[str]] = None,
) -> float:
    """TA-grouped relative-success-at-K, mean across therapeutic areas.

    See `_rs_ta_values_at_k` for the per-TA RS definition. Returns the mean
    across qualifying TAs, or 0.0 if none qualify. NOTE: the mean is inflated
    by a single high-RS TA (concentration); compare against
    `rs_ta_median_at_k` to gauge how broad performance is across TAs.
    """
    vals = _rs_ta_values_at_k(scores, labels, ta_per_item, k, primary_tas)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def rs_ta_median_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    ta_per_item: List[List[str]],
    k: int,
    primary_tas: Optional[List[str]] = None,
) -> float:
    """TA-grouped relative-success-at-K, MEDIAN across therapeutic areas.

    Robust to per-TA concentration: unlike the mean, a single spiked TA cannot
    inflate it, so a high median indicates broad competence across TAs. A large
    mean - median gap signals collapse into a few TAs.
    Returns the median across qualifying TAs, or 0.0 if none qualify.
    """
    vals = _rs_ta_values_at_k(scores, labels, ta_per_item, k, primary_tas)
    if not vals:
        return 0.0
    return float(np.median(vals))


def mean_reciprocal_rank(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        scores: Predicted scores [num_items]
        labels: Binary labels [num_items]
        
    Returns:
        MRR score
    """
    if labels.sum() == 0:
        return 0.0
    
    # Sort by scores (descending)
    _, sorted_indices = torch.sort(scores, descending=True)
    sorted_labels = labels[sorted_indices]
    
    # Find rank of first relevant item (1-indexed)
    relevant_ranks = torch.where(sorted_labels == 1)[0]
    if len(relevant_ranks) == 0:
        return 0.0
    
    first_relevant_rank = relevant_ranks[0].item() + 1  # 1-indexed
    
    return 1.0 / first_relevant_rank


def compute_ranking_metrics(
    scores_dict: Dict[int, torch.Tensor],
    labels_dict: Dict[int, torch.Tensor],
    k_values: List[int] = [10, 20, 50, 100],
) -> Dict[str, float]:
    """
    Compute ranking metrics for all users.
    
    Args:
        scores_dict: Dict mapping user index to scores
        labels_dict: Dict mapping user index to labels
        k_values: List of k values for top-k metrics
        
    Returns:
        Dictionary of aggregated metrics
    """
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"precision@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    metrics["mrr"] = []
    
    for user_idx in scores_dict.keys():
        scores = scores_dict[user_idx]
        labels = labels_dict[user_idx]
        
        # Skip users with no relevant items
        if labels.sum() == 0:
            continue
        
        # Compute metrics for each k
        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(scores, labels, k))
            metrics[f"precision@{k}"].append(precision_at_k(scores, labels, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(scores, labels, k))
        
        # MRR
        metrics["mrr"].append(mean_reciprocal_rank(scores, labels))
    
    # Average across users
    aggregated = {}
    for metric_name, values in metrics.items():
        if values:
            aggregated[metric_name] = np.mean(values)
        else:
            aggregated[metric_name] = 0.0
    
    return aggregated

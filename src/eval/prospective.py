"""Prospective target discovery evaluation.

Treats the trained advancement model as a recommender. For each user-supplied
disease EFO ID:
  1. Build a candidate pool of (target, disease) pairs with NO clinical-trial
     precedence at the cutoff year.
  2. Score every candidate with the trained model.
  3. Label as positive any pair whose first trial-related edge appears strictly
     after the cutoff (clinical_trial_* or advancement edges).
  4. Compute precision@K and recall@K per disease, then macro-average.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from src.train_advancement_hgt import ADV_ETYPE, predict_test


def _trial_etypes(data) -> list[tuple[str, str, str]]:
    """Edge types whose name contains 'clinical_trial' (any phase/outcome)."""
    return [et for et in data.edge_types if "clinical_trial" in et[1]]


def _pairs_from_etype(data, etype, time_mask) -> set[tuple[int, int]]:
    """Return (target_idx, disease_idx) pairs from `etype` filtered by mask.

    Handles both edge orientations: (target, _, disease) and (disease, _, target).
    """
    src, _, dst = etype
    ei = data[etype].edge_index
    et_time = data[etype].edge_time
    mask = time_mask(et_time)
    if mask.sum().item() == 0:
        return set()
    sel = ei[:, mask]
    if src == "target" and dst == "disease":
        t_idx, d_idx = sel[0], sel[1]
    elif src == "disease" and dst == "target":
        d_idx, t_idx = sel[0], sel[1]
    else:
        raise ValueError(f"Unexpected etype orientation: {etype}")
    return set(zip(t_idx.tolist(), d_idx.tolist()))


def build_prior_precedence_set(data, cutoff_year: int) -> set[tuple[int, int]]:
    """Pairs with any clinical_trial edge at edge_time <= cutoff."""
    out: set[tuple[int, int]] = set()
    for et in _trial_etypes(data):
        out |= _pairs_from_etype(data, et, lambda t: t <= cutoff_year)
    return out


def build_future_positive_set(
    data, cutoff_year: int, prior_set: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    """Pairs with a clinical_trial OR advancement edge at edge_time > cutoff,
    excluding any pair that already had precedence at edge_time <= cutoff."""
    fut: set[tuple[int, int]] = set()
    for et in _trial_etypes(data):
        fut |= _pairs_from_etype(data, et, lambda t: t > cutoff_year)
    fut |= _pairs_from_etype(data, ADV_ETYPE, lambda t: t > cutoff_year)
    return fut - prior_set


def build_candidate_pool(
    disease_idx: int, num_targets: int, prior_set: set[tuple[int, int]]
) -> torch.Tensor:
    """edge_label_index [2, N] for LinkNeighborLoader: row 0 targets, row 1 disease."""
    excluded = {t for (t, d) in prior_set if d == disease_idx}
    candidate_targets = [t for t in range(num_targets) if t not in excluded]
    t_tensor = torch.tensor(candidate_targets, dtype=torch.long)
    d_tensor = torch.full((len(candidate_targets),), disease_idx, dtype=torch.long)
    return torch.stack([t_tensor, d_tensor], dim=0)


def _precision_recall_at_k(
    scores: np.ndarray, labels: np.ndarray, ks: Iterable[int]
) -> dict[int, tuple[float, float]]:
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    n_pos = int(labels.sum())
    out: dict[int, tuple[float, float]] = {}
    for k in ks:
        k_eff = min(k, len(sorted_labels))
        top_hits = int(sorted_labels[:k_eff].sum())
        precision = top_hits / k_eff if k_eff > 0 else 0.0
        recall = top_hits / n_pos if n_pos > 0 else 0.0
        out[int(k)] = (precision, recall)
    return out


def run_prospective_eval(
    model,
    data,
    context,
    mappings,
    cfg,
    output_dir: Path,
) -> None:
    """Compute per-disease prospective P@K / R@K and write CSVs + parquet."""
    p_cfg = cfg.eval.prospective
    diseases: list[str] = list(p_cfg.get("diseases", []) or [])
    if not diseases:
        print("Prospective eval: no diseases configured, skipping.")
        return

    cutoff_year: int = int(p_cfg.get("cutoff_year", 2015))
    ks: list[int] = [int(k) for k in p_cfg.get("ks", [100, 200, 500])]

    device = next(model.parameters()).device
    num_neighbors = list(cfg.train.num_neighbors)
    # Inference has no gradient memory — use larger batches than training.
    # Allow override via cfg.eval.prospective.batch_size; default 4× training.
    train_batch_size = int(cfg.train.batch_size)
    batch_size = int(p_cfg.get("batch_size", train_batch_size * 4))
    num_workers = int(p_cfg.get("num_workers", 4))
    edge_feat_cols = list(cfg.model.get("edge_feat_cols", [0, 1]))

    disease_map: dict[str, int] = mappings["node_mapping"]["disease"]
    target_map: dict[str, int] = mappings["node_mapping"]["target"]
    inv_target = {v: k for k, v in target_map.items()}
    num_targets = data["target"].num_nodes

    print(
        f"Prospective eval: cutoff={cutoff_year}, ks={ks}, "
        f"diseases={len(diseases)}, num_targets={num_targets}, "
        f"batch_size={batch_size}, num_workers={num_workers}"
    )

    print("Building prior precedence set...")
    prior_set = build_prior_precedence_set(data, cutoff_year)
    print(f"  prior precedence pairs: {len(prior_set)}")

    print("Building future positive set...")
    future_set = build_future_positive_set(data, cutoff_year, prior_set)
    print(f"  future positive pairs: {len(future_set)}")

    out_dir = Path(output_dir) / "prospective"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Precompute future-positive count per disease so we can skip zero-positive
    # diseases without doing the expensive scoring pass (precision/recall
    # macro-average is meaningless when there are no ground-truth positives).
    from collections import Counter
    fut_count_by_d: Counter = Counter(d for (_t, d) in future_set)

    per_disease_rows = []
    macro_acc = {k: {"p": [], "r": []} for k in ks}
    pred_rows: list[pd.DataFrame] = []
    candidate_pool_sizes: list[int] = []  # for end-of-eval summary
    n_skipped_no_positives = 0

    # ---- Pass 1: build a single concatenated edge_label_index across all eligible
    # diseases, plus offsets so we can split scores back per-disease afterwards.
    eligible: list[tuple[str, int, torch.Tensor]] = []  # (efo, d_idx, candidate_pool)
    for efo in diseases:
        if efo not in disease_map:
            print(f"  [skip] {efo}: not in disease node mapping")
            continue
        d_idx = disease_map[efo]
        if fut_count_by_d.get(d_idx, 0) == 0:
            n_skipped_no_positives += 1
            continue
        cp = build_candidate_pool(d_idx, num_targets, prior_set)
        n_cand = cp.shape[1]
        candidate_pool_sizes.append(n_cand)
        if n_cand == 0:
            print(f"  [skip] {efo}: empty candidate pool")
            continue
        eligible.append((efo, d_idx, cp))

    if not eligible:
        print("Prospective eval: no eligible diseases (after filters).")
        return

    print(
        f"Scoring {len(eligible)} diseases in one concatenated pass "
        f"(total candidate pairs: {sum(cp.shape[1] for _,_,cp in eligible)})..."
    )

    cat_edge_index = torch.cat([cp for _, _, cp in eligible], dim=1)
    n_total = cat_edge_index.shape[1]
    cat_edge_times = torch.full((n_total,), cutoff_year + 1, dtype=torch.long)
    cat_dummy_labels = torch.zeros(n_total, dtype=torch.float)

    cat_scores, _ = predict_test(
        model,
        context,
        edge_index=cat_edge_index,
        edge_labels=cat_dummy_labels,
        edge_times=cat_edge_times,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        device=device,
        edge_feat_cols=edge_feat_cols,
        num_workers=num_workers,
    )

    # ---- Pass 2: split scores back per disease and compute metrics.
    cursor = 0
    for efo, d_idx, cp in eligible:
        n_cand = cp.shape[1]
        scores = cat_scores[cursor:cursor + n_cand]
        target_indices = cp[0].tolist()
        labels = np.array(
            [1 if (t, d_idx) in future_set else 0 for t in target_indices],
            dtype=np.int64,
        )
        n_pos = int(labels.sum())
        cursor += n_cand

        prk = _precision_recall_at_k(scores, labels, ks)
        for k, (p, r) in prk.items():
            per_disease_rows.append(
                {
                    "disease_id": efo,
                    "K": k,
                    "n_candidates": n_cand,
                    "n_future_positives": n_pos,
                    "precision_at_k": p,
                    "recall_at_k": r,
                }
            )
            macro_acc[k]["p"].append(p)
            macro_acc[k]["r"].append(r)
            print(
                f"  {efo} K={k}: P@K={p:.4f} R@K={r:.4f} "
                f"(n_cand={n_cand}, n_pos={n_pos})"
            )

        pred_rows.append(
            pd.DataFrame(
                {
                    "disease_id": efo,
                    "target_id": [inv_target[t] for t in target_indices],
                    "score": scores,
                    "future_positive": labels,
                }
            )
        )

    if candidate_pool_sizes:
        arr = np.array(candidate_pool_sizes)
        print(
            f"\nScored {len(arr)} diseases (skipped {n_skipped_no_positives} "
            f"with zero future positives after {cutoff_year}). "
            f"Candidate-pool size: mean={arr.mean():.1f}, "
            f"median={int(np.median(arr))}, "
            f"min={int(arr.min())}, max={int(arr.max())}, "
            f"total_targets={num_targets}"
        )

    if not per_disease_rows:
        print("Prospective eval: no diseases produced metrics.")
        return

    per_disease_df = pd.DataFrame(per_disease_rows)
    per_disease_path = out_dir / "prospective_per_disease.csv"
    per_disease_df.to_csv(per_disease_path, index=False)
    print(f"Per-disease metrics saved to {per_disease_path}")

    macro_rows = []
    for k in ks:
        ps, rs = macro_acc[k]["p"], macro_acc[k]["r"]
        if not ps:
            continue
        macro_rows.append(
            {
                "K": k,
                "precision_at_k_macro": float(np.mean(ps)),
                "recall_at_k_macro": float(np.mean(rs)),
                "n_diseases": len(ps),
            }
        )
    macro_df = pd.DataFrame(macro_rows)
    macro_path = out_dir / "prospective_macro.csv"
    macro_df.to_csv(macro_path, index=False)
    print(f"Macro metrics saved to {macro_path}")

    preds_df = pd.concat(pred_rows, ignore_index=True)
    preds_path = out_dir / "prospective_predictions.parquet"
    preds_df.to_parquet(preds_path, index=False)
    print(f"Prospective predictions saved to {preds_path}")

    try:
        import wandb

        if wandb.run is not None:
            log_dict: dict[str, float] = {}
            for row in per_disease_rows:
                key = f"prospective/{row['disease_id']}/p@{row['K']}"
                log_dict[key] = row["precision_at_k"]
                log_dict[f"prospective/{row['disease_id']}/r@{row['K']}"] = row[
                    "recall_at_k"
                ]
            for row in macro_rows:
                log_dict[f"prospective/macro/p@{row['K']}"] = row[
                    "precision_at_k_macro"
                ]
                log_dict[f"prospective/macro/r@{row['K']}"] = row["recall_at_k_macro"]
            wandb.log(log_dict)
    except Exception as e:
        print(f"wandb logging skipped: {e}")

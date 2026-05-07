#!/usr/bin/env python3
"""Run prospective target discovery evaluation against a saved checkpoint.

Reuses the run's saved config (`<output_dir>/config.yaml`) and best model
(`<output_dir>/best_model.pt`) to rebuild the exact model/context used at
training time and then calls `run_prospective_eval`.

Example:
  python evaluate_prospective_standalone.py \
      --output_dir runs/advancement_lambdarank \
      --diseases EFO_0000676 EFO_0003767
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train_advancement_hgt import build_context_graph, split_advancement_edges, ADV_ETYPE
from src.models.utils import build_model
from src.eval.prospective import run_prospective_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True,
                    help="Run directory containing config.yaml and best_model.pt")
    ap.add_argument("--diseases", nargs="*", default=None,
                    help="EFO IDs to evaluate (overrides config)")
    ap.add_argument("--cutoff_year", type=int, default=None)
    ap.add_argument("--ks", nargs="*", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None,
                    help="Override eval.prospective.batch_size (default: 4× training batch)")
    ap.add_argument("--num_workers", type=int, default=None,
                    help="Override eval.prospective.num_workers for sampler (default: 4)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    cfg_path = output_dir / "config.yaml"
    ckpt_path = output_dir / "best_model.pt"
    if not cfg_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing config.yaml or best_model.pt in {output_dir}")

    cfg = OmegaConf.load(cfg_path)

    if "eval" not in cfg or cfg.eval is None:
        cfg.eval = OmegaConf.create({})
    if "prospective" not in cfg.eval or cfg.eval.prospective is None:
        cfg.eval.prospective = OmegaConf.create(
            {"diseases": [], "cutoff_year": 2015, "ks": [100, 200, 500]}
        )
    if args.diseases is not None:
        cfg.eval.prospective.diseases = list(args.diseases)
    if args.cutoff_year is not None:
        cfg.eval.prospective.cutoff_year = int(args.cutoff_year)
    if args.ks is not None:
        cfg.eval.prospective.ks = [int(k) for k in args.ks]
    if args.batch_size is not None:
        cfg.eval.prospective.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        cfg.eval.prospective.num_workers = int(args.num_workers)

    if not list(cfg.eval.prospective.get("diseases", []) or []):
        print("No diseases specified (use --diseases or set eval.prospective.diseases).")
        return

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    from src.data.temporal_loader import load_event_graph
    to_undirected = bool(cfg.data.get("undirected", False))
    print(f"Loading graph from {cfg.data.graph_file} (undirected={to_undirected})")
    data = load_event_graph(cfg.data.graph_file, to_undirected=to_undirected)

    train_mask, _, _, _ = split_advancement_edges(data)
    edge_time = data[ADV_ETYPE].edge_time

    print("Building context graph...")
    context = build_context_graph(data)

    use_recency = bool(cfg.model.get("use_recency", False))
    time_dim = int(cfg.model.get("time_dim", 0))
    if use_recency:
        train_times = edge_time[train_mask].float()
        t_min_val = float(train_times.min().item())
        t_max_val = float(train_times.max().item())
    else:
        t_min_val, t_max_val = 0.0, 1.0

    model = build_model(
        model_name=cfg.model.name,
        data=context,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        use_rte=cfg.model.get("use_rte", False),
        use_edge_features=cfg.model.get("use_edge_features", False),
        edge_feat_dim=cfg.model.get("edge_feat_dim", 2),
        use_recency=use_recency,
        time_dim=time_dim,
        t_min=t_min_val,
        t_max=t_max_val,
    ).to(device)
    print(f"Loading checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    mappings = torch.load(cfg.data.mappings_file, weights_only=False)

    run_prospective_eval(model, data, context, mappings, cfg, output_dir)


if __name__ == "__main__":
    main()

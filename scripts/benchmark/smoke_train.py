#!/usr/bin/env python
"""Smoke-train a small heterogeneous GNN on a release and score T1 or T2 on val, end to end.

    python scripts/benchmark/smoke_train.py --release-dir $RELEASE_DIR --task discovery \
        --recipe single --out OUT.json

Shows the release works as a user would use it: snapshots read through the
release's own load_thbkg.py, its label tables, and the evaluation harness
(src.benchmark.tasks.evaluate_task). Not a baseline: one seed, a few minutes
of training, no tuning, and no test (it is scored once per final model).

Protocol, the val split of src.benchmark.tasks:
  * train on evidence and labels dated before 2017 (``fit_before``), with one of
    the ``RECIPES``: "single" trains on the snapshot before 2017, every step on
    a batch of the positives dated before it with their own label edges hidden
    (``hide_pairs``); "stacked" trains on the windows 1996-1998 ... 2014-2016
    (``fit_windows``), every step on one window's positives and the snapshot
    before its first year, which holds none of their edges;
  * score the val window (cutoff 2017, events 2017-2019) on the snapshot of
    its cutoff, with the trained weights, over every candidate of every slate,
    keeping the raw predictions (a PredictionLog) next to ``--out`` and the
    run's record in results.json there (src.benchmark.results).

Model: a projection per node type (ModalityEncoder for the molecule
modalities), two HeteroConv layers with one GraphSAGE convolution per relation
(to_hetero's torch.fx tracing fails on torch 2.7 with PyG 2.6.1), and the dot
product of target and disease embeddings; binary cross-entropy against random
targets for the same disease, dropping those already linked to it before the
step's snapshot (they are not candidates).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.benchmark.results import save_result  # noqa: E402
from src.benchmark.slates import PredictionLog, expected_random  # noqa: E402
from src.benchmark.tasks import (  # noqa: E402
    RECIPES, SPLITS, evaluate_task, fit_before, fit_windows, hide_pairs, linked_pairs, training_pairs,
)
from src.models.node_input import input_spec, make_input_projection, node_inputs  # noqa: E402

LABEL_FILE = {"discovery": "discovery", "hypothesis": "hypothesis"}


def release_loader(release_dir: Path):
    """The release's own load_thbkg module, imported without leaving __pycache__/ in the release."""
    spec = importlib.util.spec_from_file_location("release_load_thbkg", release_dir / "load_thbkg.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {release_dir / 'load_thbkg.py'}")
    mod = importlib.util.module_from_spec(spec)
    write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = write_bytecode
    return mod


class Model(nn.Module):
    """Input projection per node type, then two layers of one GraphSAGE convolution per relation, summed."""

    def __init__(self, data, hidden: int, layers: int = 2):
        super().__init__()
        self.proj = nn.ModuleDict({nt: make_input_projection(input_spec(data[nt]), hidden) for nt in data.node_types})
        self.convs = nn.ModuleList(
            HeteroConv({et: SAGEConv((-1, -1), hidden) for et in data.edge_types}, aggr="sum") for _ in range(layers))

    def forward(self, data):
        x = {nt: self.proj[nt](v) for nt, v in node_inputs(data).items()}
        for i, conv in enumerate(self.convs):
            x = conv(x, data.edge_index_dict)
            if i < len(self.convs) - 1:
                x = {nt: F.relu(h) for nt, h in x.items()}
        return x


def pairs(df: pd.DataFrame, dev) -> tuple[torch.Tensor, torch.Tensor]:
    """(target_index, disease_index) of a label table's rows, on ``dev``."""
    return (torch.from_numpy(df["target_index"].to_numpy(np.int64)).to(dev),
            torch.from_numpy(df["disease_index"].to_numpy(np.int64)).to(dev))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-dir", required=True, type=Path)
    ap.add_argument("--task", required=True, choices=sorted(LABEL_FILE))
    ap.add_argument("--recipe", default="single", choices=sorted(RECIPES))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=50_000, help="positives per step")
    ap.add_argument("--negatives", type=int, default=5, help="random targets per positive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rd = args.release_dir
    lt = release_loader(rd)
    meta = json.loads((rd / "metadata.json").read_text())
    n_targets = meta["node_types"]["target"]["num_nodes"]
    n_diseases = meta["node_types"]["disease"]["num_nodes"]
    labels = lt.load_labels(rd, LABEL_FILE[args.task])

    # ---- train on what is dated before the val cutoff: one snapshot ("single") or one per window
    before = fit_before("val")
    if args.recipe == "single":
        fits = [(before, training_pairs(args.task, labels, before))]
    else:
        fits = [(lo, training_pairs(args.task, labels, hi + 1, since=lo)) for lo, hi in fit_windows(before)]
    snaps, pos_by_snap, linked_by_snap = [], [], []
    for cutoff, df in fits:
        snaps.append(lt.load_snapshot(rd, cutoff, normalise="max", recency=False).to(dev))
        pos_by_snap.append(pairs(df, dev))
        link_t, link_d = pairs(linked_pairs(args.task, labels, cutoff), dev)
        linked_by_snap.append(link_t * n_diseases + link_d)
    sizes = np.array([p[0].numel() for p in pos_by_snap], dtype=np.float64)
    n_fit = int(sizes.sum())
    print(f"{args.task} {args.recipe}: {n_fit:,} fit positives from {len(snaps)} snapshot(s), "
          f"cutoffs {fits[0][0]}-{fits[-1][0]} ({time.time() - t0:.0f}s)")
    model = Model(snaps[-1], args.hidden).to(dev)
    with torch.no_grad():
        model(snaps[-1])  # materialise the lazy layers
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for epoch in range(args.epochs):
        model.train()
        k = int(rng.choice(len(snaps), p=sizes / sizes.sum()))  # a snapshot, in proportion to its positives
        pos_t, pos_d = pos_by_snap[k]
        i = torch.from_numpy(rng.choice(pos_t.numel(), min(args.batch, pos_t.numel()), replace=False)).to(dev)
        t, d = pos_t[i], pos_d[i]
        neg_t = torch.randint(0, n_targets, (t.numel() * args.negatives,), device=dev)
        neg_d = d.repeat(args.negatives)
        keep = ~torch.isin(neg_t * n_diseases + neg_d, linked_by_snap[k])
        neg_t, neg_d = neg_t[keep], neg_d[keep]
        z = model(hide_pairs(snaps[k], args.task, t, d) if args.recipe == "single" else snaps[k])
        zt, zd = z["target"], z["disease"]
        pos = (zt[t] * zd[d]).sum(-1)
        neg = (zt[neg_t] * zd[neg_d]).sum(-1)
        loss = F.binary_cross_entropy_with_logits(torch.cat([pos, neg]),
                                                  torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  cutoff {fits[k][0]}  loss {losses[-1]:.4f}  ({time.time() - t0:.0f}s)")
    del snaps

    # ---- score the val window on the snapshot of its cutoff
    model.eval()

    def score_fn_for_year(year, slates):
        snap = lt.load_snapshot(rd, year, normalise="max", recency=False).to(dev)
        with torch.no_grad():
            z = model(snap)
            diseases = sorted({s.disease_index for s in slates})
            col = {d: j for j, d in enumerate(diseases)}
            scores = (z["target"] @ z["disease"][torch.tensor(diseases, device=dev)].T).cpu().numpy()
        print(f"  val cutoff {year}: {len(slates):,} slates ({time.time() - t0:.0f}s)")
        return lambda s: scores[s.candidates, col[s.disease_index]]

    v0, v1 = SPLITS["val"][0]
    log = PredictionLog()
    summary, rows = evaluate_task(args.task, "val", labels, num_targets=n_targets,
                                  score_fn_for_year=score_fn_for_year, log=log)
    log.save(args.out.parent / "predictions", args.out.stem)  # raw predictions, for rescoring later
    floor = expected_random(pd.DataFrame(rows))
    out = {"task": args.task, "split": "val", "window": [v0, v1], "recipe": args.recipe, "fit_before": before,
           "fit_cutoffs": [c for c, _ in fits], "train_positives": n_fit,
           "epochs": args.epochs, "hidden": args.hidden, "seed": args.seed,
           "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "loss_first": losses[0], "loss_last": losses[-1], "seconds": round(time.time() - t0),
           "metrics": summary, "random_expected": floor}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    save_result(args.out.parent / "results.json", task=args.task, model="smoke_sage", recipe=args.recipe,
                seed=args.seed, val=summary, train_positives=n_fit, epochs=args.epochs, seconds=out["seconds"])
    keys = ("precision@R", "recall@100", "ndcg@100")
    print(f"✅ {args.task} {args.recipe}, val {v0}-{v1}: " + ", ".join(f"{k} {summary[k]:.4f} (random {floor.get(k, float('nan')):.4f})"
                                          for k in keys) + f" -> {args.out}")


if __name__ == "__main__":
    main()

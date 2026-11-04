#!/usr/bin/env python
"""Smoke-train a small heterogeneous GNN on a release and score a T1 or T2 model-selection fold, end to end.

    python scripts/benchmark/smoke_train.py --release-dir $RELEASE_DIR --task discovery --out OUT.json

Shows the release works as a user would use it: snapshots read through the
release's own load_thbkg.py, its label tables, and the evaluation harness
(src.benchmark.tasks.evaluate_task). Not a baseline: one seed, a few minutes
of training, no tuning.

Protocol, with no label inside the graph it is trained or scored on:
  * train on the snapshot of ``--train-snapshot`` (evidence dated before it),
    with the task's positives whose label year lies between that year and the
    end of the last ``cv_folds`` fold's train years: none of their own edges is
    in the snapshot;
  * score each decision year Y of that fold's validation block (inside the
    train split; test is never touched) on its own snapshot (evidence before Y),
    with the trained weights, over every candidate of every slate, keeping the
    raw predictions (a PredictionLog) next to ``--out``.

Model: a projection per node type (ModalityEncoder for the molecule
modalities), two HeteroConv layers with one GraphSAGE convolution per relation
(to_hetero's torch.fx tracing fails on torch 2.7 with PyG 2.6.1), and the dot
product of target and disease embeddings; binary cross-entropy against random
targets for the same disease.
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
from src.benchmark.slates import PredictionLog, expected_random  # noqa: E402
from src.benchmark.tasks import LABEL_YEAR, cv_folds, evaluate_task  # noqa: E402
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


def positives(labels: pd.DataFrame, task: str, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """(target_index, disease_index) of the task's positives with label year in [lo, hi]."""
    y = labels[LABEL_YEAR[task]]
    keep = y.between(lo, hi) & (labels["novel"] if task == "hypothesis" else True)
    df = labels.loc[keep]
    return df["target_index"].to_numpy(np.int64), df["disease_index"].to_numpy(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-dir", required=True, type=Path)
    ap.add_argument("--task", required=True, choices=sorted(LABEL_FILE))
    ap.add_argument("--train-snapshot", type=int, default=2011)
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
    labels = lt.load_labels(rd, LABEL_FILE[args.task])

    # ---- train on one snapshot, with positives whose own edges come later
    fold = cv_folds(args.task)[-1]
    g = lt.load_snapshot(rd, args.train_snapshot, normalise="max", recency=False).to(dev)
    tr_t, tr_d = positives(labels, args.task, args.train_snapshot, fold["train"][1])
    print(f"{args.task}: {tr_t.size:,} train positives {args.train_snapshot}-{fold['train'][1]}, "
          f"snapshot {args.train_snapshot}: {sum(g[et].num_edges for et in g.edge_types):,} edges ({time.time() - t0:.0f}s)")
    model = Model(g, args.hidden).to(dev)
    with torch.no_grad():
        model(g)  # materialise the lazy layers
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_t, pos_d = torch.from_numpy(tr_t).to(dev), torch.from_numpy(tr_d).to(dev)
    losses = []
    for epoch in range(args.epochs):
        model.train()
        i = torch.from_numpy(rng.choice(tr_t.size, min(args.batch, tr_t.size), replace=False)).to(dev)
        t, d = pos_t[i], pos_d[i]
        neg_t = torch.randint(0, n_targets, (t.numel() * args.negatives,), device=dev)
        z = model(g)
        zt, zd = z["target"], z["disease"]
        pos = (zt[t] * zd[d]).sum(-1)
        neg = (zt[neg_t] * zd[d.repeat(args.negatives)]).sum(-1)
        loss = F.binary_cross_entropy_with_logits(torch.cat([pos, neg]),
                                                  torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  loss {losses[-1]:.4f}  ({time.time() - t0:.0f}s)")
    del g

    # ---- score the fold's validation block, one decision year at a time on that year's snapshot
    model.eval()

    def score_fn_for_year(year, slates):
        snap = lt.load_snapshot(rd, year, normalise="max", recency=False).to(dev)
        with torch.no_grad():
            z = model(snap)
            diseases = sorted({s.disease_index for s in slates})
            col = {d: j for j, d in enumerate(diseases)}
            scores = (z["target"] @ z["disease"][torch.tensor(diseases, device=dev)].T).cpu().numpy()
        print(f"  val {year}: {len(slates):,} slates ({time.time() - t0:.0f}s)")
        return lambda s: scores[s.candidates, col[s.disease_index]]

    v0, v1 = fold["val"]
    log = PredictionLog()
    summary, rows = evaluate_task(args.task, "train", labels, num_targets=n_targets,
                                  score_fn_for_year=score_fn_for_year, years=range(v0, v1 + 1), log=log)
    log.save(args.out.parent / "predictions", args.out.stem)  # raw predictions, for rescoring later
    floor = expected_random(pd.DataFrame(rows))
    out = {"task": args.task, "split": f"cv fold, val {v0}-{v1}", "train_snapshot": args.train_snapshot,
           "train_positives": int(tr_t.size), "epochs": args.epochs, "hidden": args.hidden, "seed": args.seed,
           "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "loss_first": losses[0], "loss_last": losses[-1], "seconds": round(time.time() - t0),
           "metrics": summary, "random_expected": floor}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    keys = ("precision@R_per_disease", "precision@R", "recall@100", "ndcg@100")
    print(f"✅ {args.task} fold val {v0}-{v1}: " + ", ".join(f"{k} {summary[k]:.4f} (random {floor.get(k, float('nan')):.4f})"
                                          for k in keys) + f" -> {args.out}")


if __name__ == "__main__":
    main()

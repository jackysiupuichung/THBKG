"""Load one decision-year snapshot of the discontinuous THBKG as a PyG HeteroData.

Snapshots are built by scripts/benchmark/build_snapshots.py: snapshot ``Y`` holds
each pair's latest state before ``Y`` with its raw ``score`` and ``recency``.
Normalisation happens here, once per load, so every model and heuristic sees
the same scaling:

``normalise="none"``  raw values.
``normalise="max"``   divide score and recency by the edge type's ``score_max``
                      (metadata.json): a constant of the scoring formula, not
                      fitted to the data, so it leaks nothing across years.
                      Every edge type then lies in [0, 1].
``normalise="log"``   ``log1p(K x) / log1p(K)`` of the max-normalised value
                      (K = 100): spreads the low end of skewed relations such
                      as literature. Also deterministic.

Anything fitted to the data (quantiles, z-scores) belongs in the training code,
fitted on training years only.

    from src.benchmark.snapshots import load_snapshot
    data = load_snapshot(release_dir, 2016, normalise="max")            # edge_attr = [score, recency]
    data = load_snapshot(release_dir, 2016, normalise="max", recency=False)  # edge_attr = [score]
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

LOG_K = 100.0
NORMALISATIONS = ("none", "max", "log")


@lru_cache(maxsize=4)
def _node_tensors(release_dir: str) -> dict:
    """Node features of the release, read once per process (shared by every year)."""
    meta = json.loads((Path(release_dir) / "metadata.json").read_text())
    out = {}
    with safe_open(str(Path(release_dir) / "graph.safetensors"), framework="pt") as f:
        for nt, nm in meta["node_types"].items():
            out[nt] = {"num_nodes": nm["num_nodes"],
                       **{a: f.get_tensor(spec["key"]) for a, spec in nm["tensors"].items()}}
    return out


def _scale(v: np.ndarray, vmax: float | None, normalise: str, et: tuple) -> np.ndarray:
    if normalise == "none":
        return v
    if not vmax:
        raise ValueError(f"no score_max for {et} in metadata.json (run scripts/benchmark/write_edge_scales.py)")
    x = v / vmax
    return x if normalise == "max" else np.log1p(LOG_K * x) / np.log1p(LOG_K)


def load_snapshot(release_dir, year: int, *, normalise: str = "max", recency: bool = True,
                  to_undirected: bool = True, snapshot_dir=None):
    """Snapshot ``year`` as HeteroData with ``edge_attr`` = [score(, recency)] and ``edge_time``.

    Args:
        release_dir: release directory (graph.safetensors + metadata.json).
        year: decision year; the snapshot contains only evidence dated before it.
        normalise: "none", "max" or "log" (see module docstring).
        recency: include the recency column (the score-only variant drops it).
        to_undirected: add a reverse edge type for every relation. Uses
            ``merge=False``: PyG's default merges a same-node-type relation
            (target interacts_with target) with its reverse and *adds* duplicate
            edges' attributes, edge_time included.
        snapshot_dir: defaults to <release_dir>/snapshots.
    """
    from torch_geometric.data import HeteroData
    import torch_geometric.transforms as T

    if normalise not in NORMALISATIONS:
        raise ValueError(f"normalise must be one of {NORMALISATIONS}")
    release_dir = Path(release_dir)
    snap_dir = Path(snapshot_dir) if snapshot_dir else release_dir / "snapshots"
    path = snap_dir / f"{year}.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"no snapshot for {year} in {snap_dir}")
    meta = json.loads((release_dir / "metadata.json").read_text())

    data = HeteroData()
    for nt, nd in _node_tensors(str(release_dir)).items():
        data[nt].num_nodes = nd["num_nodes"]
        for a, v in nd.items():
            if a != "num_nodes":
                data[nt][a] = v

    with safe_open(str(path), framework="np") as f:
        for i, e in enumerate(meta["edge_types"]):
            et = (e["src"], e["rel"], e["dst"])
            vmax = e.get("score_max")
            cols = [_scale(f.get_tensor(f"edge.{i}.score"), vmax, normalise, et)]
            if recency:
                cols.append(_scale(f.get_tensor(f"edge.{i}.recency"), vmax, normalise, et))
            data[et].edge_index = torch.from_numpy(f.get_tensor(f"edge.{i}.edge_index").astype(np.int64))
            data[et].edge_attr = torch.from_numpy(np.stack(cols, axis=1).astype(np.float32))
            data[et].edge_time = torch.from_numpy(f.get_tensor(f"edge.{i}.edge_time").astype(np.int64))

    data.year = int(year)
    data.edge_attr_columns = ["score", "recency"] if recency else ["score"]
    data.normalise = normalise
    if to_undirected:
        data = T.ToUndirected(merge=False)(data)
    return data

"""Standalone loader for the THBKG HuggingFace release.

Two views of the same graph:

* continuous (``load_graph``): the event stream in ``graph.safetensors``, one
  edge per (pair, relation) for every year its score changed, with
  ``edge_attr`` = [score] and ``edge_time`` = that year;
* discontinuous (``load_snapshot``): one snapshot per decision year in
  ``snapshots/``, holding each pair's latest state before that year, with
  ``edge_attr`` = [score, recency].

It also reads the advancement label splits. No pickle, and no THBKG source
package required.

    pip install torch torch_geometric safetensors pandas pyarrow

    from load_thbkg import load_graph, load_snapshot, snapshot_years, load_labels, node_index

    data  = load_graph("path/to/release")                         # continuous, PyG HeteroData, raw scores
    snap  = load_snapshot("path/to/release", 2016)                # discontinuous, decision year 2016, raw
    snap  = load_snapshot("path/to/release", 2016, normalise="max")   # scores scaled to [0, 1] per relation
    train = load_labels("path/to/release", "train")               # pandas DataFrame
    eval_ = load_labels("path/to/release", "eval")

    # map a label endpoint to its graph node index
    t2i = node_index("path/to/release", "target")
    d2i = node_index("path/to/release", "disease")

Downloading the release from the Hub first:

    from huggingface_hub import snapshot_download
    d = snapshot_download("jspc00012567/THBKG", repo_type="dataset")
    data = load_graph(d)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

# Symmetric relations (same node type at both ends, no direction): made
# undirected within their own edge type. Every other relation gets a reverse
# type rev_<rel>, including the ontologies (child->parent differs from
# parent->child).
SYMMETRIC_RELATIONS = frozenset({"interacts_with"})
LOG_K = 100.0
NORMALISATIONS = ("none", "max", "log")


def _scale(v, vmax, normalise, et):
    """Score / recency scaling: "none" (raw), "max" (/ score_max, a constant of
    the scoring formula, so every relation lies in [0, 1]) or "log"."""
    if normalise not in NORMALISATIONS:
        raise ValueError(f"normalise must be one of {NORMALISATIONS}")
    if normalise == "none":
        return v
    if not vmax:
        raise ValueError(f"no score_max for {et} in metadata.json")
    x = v / vmax
    return x if normalise == "max" else np.log1p(LOG_K * x) / np.log1p(LOG_K)


def _node_tensors(release_dir: Path, meta: dict) -> dict:
    """Node tensors only, without reading the edge tensors of graph.safetensors."""
    from safetensors import safe_open
    keys = {spec["key"] for nm in meta["node_types"].values() for spec in nm["tensors"].values()}
    with safe_open(str(release_dir / "graph.safetensors"), framework="pt") as f:
        return {k: f.get_tensor(k) for k in keys}


def _check_snapshot_index(release_dir: Path, meta: dict, year: int) -> None:
    """The snapshots must match this release's edge-type order and graph file."""
    index = json.loads((release_dir / "snapshots" / "index.json").read_text())
    if int(year) not in index["years"]:
        raise FileNotFoundError(f"no snapshot for {year}; available: {index['years'][0]}-{index['years'][-1]}")
    order = ["|".join((e["src"], e["rel"], e["dst"])) for e in meta["edge_types"]]
    if "edge_types" in index and index["edge_types"] != order:
        raise ValueError("snapshot edge-type order differs from metadata.json")
    want = index.get("source_graph_sha256")
    if want is not None:
        key = str(release_dir.resolve())
        if key not in _VERIFIED_GRAPHS:
            h = hashlib.sha256()
            with open(release_dir / "graph.safetensors", "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            _VERIFIED_GRAPHS[key] = h.hexdigest()
        if _VERIFIED_GRAPHS[key] != want:
            raise ValueError("snapshots were built from a different graph.safetensors")


_VERIFIED_GRAPHS: dict = {}


def _to_undirected_typed(data, dedupe_on_time: bool):
    """Add reverse edges without summing attributes.

    PyG's ToUndirected() merges a same-node-type relation with its reverse and
    adds duplicate edges' attributes, edge_time included; this does not. Mirrors
    src.data.temporal_loader.to_undirected_typed in the THBKG repository.
    """
    for et in list(data.edge_types):
        src, rel, dst = et
        if rel.startswith("rev_"):
            continue
        store = data[et]
        nnz = store.edge_index.size(1)
        values = {k: v for k, v in store.items()
                  if k != "edge_index" and torch.is_tensor(v) and v.size(0) == nnz}
        if rel in SYMMETRIC_RELATIONS and src == dst:
            ei = torch.cat([store.edge_index, store.edge_index.flip(0)], dim=1)
            vals = {k: torch.cat([v, v], dim=0) for k, v in values.items()}
            cols = [ei[0], ei[1]]
            if dedupe_on_time and "edge_time" in vals:
                cols.append(vals["edge_time"])
            uniq, inv = torch.unique(torch.stack(cols, dim=1), dim=0, return_inverse=True)
            store.edge_index = uniq[:, :2].t().contiguous()
            for k, v in vals.items():
                idx = inv.view(-1, *([1] * (v.dim() - 1))).expand_as(v)
                out = torch.empty((uniq.size(0),) + tuple(v.shape[1:]), dtype=v.dtype)
                store[k] = out.scatter_reduce_(0, idx, v, reduce="amax", include_self=False)
        else:
            rev = data[dst, f"rev_{rel}", src]
            rev.edge_index = store.edge_index.flip(0)
            for k, v in values.items():
                rev[k] = v
    return data


def _node_store(data, release_dir: Path, meta: dict, tensors, with_node_ids: bool):
    for nt, nmeta in meta["node_types"].items():
        data[nt].num_nodes = nmeta["num_nodes"]
        for attr, spec in nmeta["tensors"].items():
            data[nt][attr] = tensors[spec["key"]]
        if with_node_ids:
            p = release_dir / "mappings" / f"{nt}.parquet"
            if p.exists():
                import pandas as pd
                data[nt].node_id = pd.read_parquet(p).sort_values("index")["node_id"].astype(str).tolist()


def _undirected_keep_ids(data, dedupe_on_time: bool):
    # node_id is a python list; keep it out of the edge transform
    stash = {nt: data[nt].node_id for nt in data.node_types if hasattr(data[nt], "node_id")}
    for nt in stash:
        del data[nt].node_id
    data = _to_undirected_typed(data, dedupe_on_time)
    for nt, ids in stash.items():
        data[nt].node_id = ids
    return data


def load_graph(release_dir, to_undirected: bool = True, with_node_ids: bool = False,
               normalise: str = "none"):
    """Rebuild the continuous HeteroData graph from a release directory.

    Args:
        release_dir: directory holding graph.safetensors + metadata.json (+ mappings/)
        to_undirected: add reverse edges as the THBKG training loader does
            (src.data.temporal_loader.load_event_graph): interacts_with becomes
            undirected within its type, every other relation gets a rev_ type.
            Set False for the raw directed graph as stored.
        with_node_ids: attach the original string IDs as ``data[ntype].node_id``
            (read from mappings/<ntype>.parquet). Off by default: a Python list
            on a node store makes PyG's neighbour samplers fail; use
            ``node_index`` to map IDs instead.
        normalise: "none" (raw scores, the default), "max" or "log", applied to
            the score column of edge_attr as in ``load_snapshot``.
    """
    from torch_geometric.data import HeteroData

    release_dir = Path(release_dir)
    meta = json.loads((release_dir / "metadata.json").read_text())
    tensors = load_file(str(release_dir / "graph.safetensors"))

    data = HeteroData()
    _node_store(data, release_dir, meta, tensors, with_node_ids)
    for emeta in meta["edge_types"]:
        et = (emeta["src"], emeta["rel"], emeta["dst"])
        for attr, spec in emeta["tensors"].items():
            data[et][attr] = tensors[spec["key"]]
        if normalise != "none" and "edge_attr" in emeta["tensors"]:
            a = data[et].edge_attr.clone()
            a[:, 0] = torch.from_numpy(_scale(a[:, 0].numpy(), emeta.get("score_max"), normalise, et))
            data[et].edge_attr = a

    return _undirected_keep_ids(data, dedupe_on_time=True) if to_undirected else data


def snapshot_years(release_dir) -> list[int]:
    """Decision years available in the discontinuous graph."""
    return json.loads((Path(release_dir) / "snapshots" / "index.json").read_text())["years"]


def load_snapshot(release_dir, year: int, normalise: str = "none", recency: bool = True,
                  to_undirected: bool = True, with_node_ids: bool = False):
    """Load the discontinuous graph at decision year ``year``.

    Holds each pair's latest state before ``year``: ``edge_attr`` = [score,
    recency] (or [score] with ``recency=False``) and ``edge_time`` = year of that
    state (-1 for static ontology edges).

    Args:
        normalise: "none" (raw, the default, as stored), "max" (divide by the
            edge type's ``score_max`` from metadata.json, a constant of the
            scoring formula, so every edge type lies in [0, 1]) or "log"
            (log1p(100 x) / log1p(100) of the max-normalised value).
        to_undirected: as in ``load_graph``, with one row per pair.
    """
    from safetensors.numpy import load_file as load_np
    from torch_geometric.data import HeteroData

    if normalise not in NORMALISATIONS:
        raise ValueError(f"normalise must be one of {NORMALISATIONS}")
    release_dir = Path(release_dir)
    meta = json.loads((release_dir / "metadata.json").read_text())
    _check_snapshot_index(release_dir, meta, year)
    snap = load_np(str(release_dir / "snapshots" / f"{int(year)}.safetensors"))

    data = HeteroData()
    _node_store(data, release_dir, meta, _node_tensors(release_dir, meta), with_node_ids)
    for i, e in enumerate(meta["edge_types"]):
        et = (e["src"], e["rel"], e["dst"])
        cols = [_scale(snap[f"edge.{i}.score"], e.get("score_max"), normalise, et)]
        if recency:
            cols.append(_scale(snap[f"edge.{i}.recency"], e.get("score_max"), normalise, et))
        data[et].edge_index = torch.from_numpy(snap[f"edge.{i}.edge_index"].astype(np.int64))
        data[et].edge_attr = torch.from_numpy(np.stack(cols, axis=1).astype(np.float32))
        data[et].edge_time = torch.from_numpy(snap[f"edge.{i}.edge_time"].astype(np.int64))
    data.year = int(year)
    data.edge_attr_columns = ["score", "recency"] if recency else ["score"]

    return _undirected_keep_ids(data, dedupe_on_time=False) if to_undirected else data


def load_labels(release_dir, split: str = "train"):
    """Read an advancement label split as a pandas DataFrame.

    Columns: target_id, disease_id, transition_year, outcome. ``split`` accepts
    "train" or "eval" (aliases: "test" -> "eval").
    """
    import pandas as pd
    name = {"train": "train", "eval": "eval", "test": "eval"}.get(split, split)
    return pd.read_parquet(Path(release_dir) / "labels" / f"{name}.parquet")


def node_index(release_dir, node_type: str) -> dict:
    """Return {node_id: index} for a node type, e.g. to resolve label endpoints."""
    import pandas as pd
    df = pd.read_parquet(Path(release_dir) / "mappings" / f"{node_type}.parquet")
    return dict(zip(df["node_id"].astype(str), df["index"].astype(int)))


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    g = load_graph(d)
    print(g)
    if (Path(d) / "snapshots" / "index.json").exists():
        ys = snapshot_years(d)
        print(f"snapshots: {ys[0]}-{ys[-1]}")
        print(load_snapshot(d, ys[len(ys) // 2]))
    for split in ("train", "eval"):
        try:
            df = load_labels(d, split)
            print(f"{split}: {len(df):,} labels, {int(df['outcome'].astype(bool).sum()):,} positive")
        except FileNotFoundError:
            pass

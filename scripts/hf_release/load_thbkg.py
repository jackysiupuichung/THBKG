"""Standalone loader for the THBKG HuggingFace release.

Reconstructs the PyG ``HeteroData`` graph from ``graph.safetensors`` +
``metadata.json`` and reads the advancement label splits. No pickle, and no
THBKG source package required.

    pip install torch torch_geometric safetensors pandas pyarrow

    from load_thbkg import load_graph, load_labels, node_index

    data  = load_graph("path/to/release")            # PyG HeteroData (training view)
    train = load_labels("path/to/release", "train")  # pandas DataFrame
    eval_ = load_labels("path/to/release", "eval")

    # map a label endpoint to its graph node index
    t2i = node_index("path/to/release", "target")
    d2i = node_index("path/to/release", "disease")

Downloading the release from the Hub first:

    from huggingface_hub import snapshot_download
    d = snapshot_download("jackysiupuichung/THBKG", repo_type="dataset")
    data = load_graph(d)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_graph(release_dir, to_undirected: bool = True, with_node_ids: bool = True):
    """Rebuild the HeteroData graph from a release directory.

    Args:
        release_dir: directory holding graph.safetensors + metadata.json (+ mappings/)
        to_undirected: add reverse edges to match the THBKG training loader
            (src.data.temporal_loader.load_event_graph). Set False for the raw
            directed graph as stored.
        with_node_ids: attach the original string IDs as ``data[ntype].node_id``
            (read from mappings/<ntype>.parquet if present).
    """
    from torch_geometric.data import HeteroData

    release_dir = Path(release_dir)
    meta = json.loads((release_dir / "metadata.json").read_text())
    tensors = load_file(str(release_dir / "graph.safetensors"))

    data = HeteroData()

    for nt, nmeta in meta["node_types"].items():
        data[nt].num_nodes = nmeta["num_nodes"]
        for attr, spec in nmeta["tensors"].items():
            data[nt][attr] = tensors[spec["key"]]
        if with_node_ids:
            p = release_dir / "mappings" / f"{nt}.parquet"
            if p.exists():
                import pandas as pd
                ids = pd.read_parquet(p).sort_values("index")["node_id"].astype(str).tolist()
                data[nt].node_id = ids

    for emeta in meta["edge_types"]:
        et = (emeta["src"], emeta["rel"], emeta["dst"])
        for attr, spec in emeta["tensors"].items():
            data[et][attr] = tensors[spec["key"]]

    if to_undirected:
        import torch_geometric.transforms as T
        # ToUndirected slices per-edge/-node tensors; node_id is a python list,
        # so stash it, transform, then restore (mirrors load_event_graph).
        stash = {}
        for nt in list(data.node_types):
            if hasattr(data[nt], "node_id"):
                stash[nt] = data[nt].node_id
                del data[nt].node_id
        data = T.ToUndirected()(data)
        for nt, ids in stash.items():
            data[nt].node_id = ids

    return data


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
    for split in ("train", "eval"):
        try:
            df = load_labels(d, split)
            print(f"{split}: {len(df):,} labels, {int(df['outcome'].astype(bool).sum()):,} positive")
        except FileNotFoundError:
            pass

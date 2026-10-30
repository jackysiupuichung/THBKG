"""Read a THBKG release directory one tensor at a time, without PyG.

A release holds metadata.json (node and edge schema), graph.safetensors (the
tensors it lists) and mappings/<node type>.parquet (node id <-> index). Used by
the release checks (scripts/benchmark/check_release.py, check_inputs.py). Only
the first edge_attr column (score) is read, so releases exported with or
without novelty read alike.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open

EdgeType = tuple[str, str, str]


def edge_key(et: EdgeType) -> str:
    return "|".join(et)


class Release:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "metadata.json").read_text())
        self._edge_meta = {(e["src"], e["rel"], e["dst"]): e for e in self.meta["edge_types"]}

    @property
    def node_types(self) -> list[str]:
        return sorted(self.meta["node_types"])

    @property
    def edge_types(self) -> list[EdgeType]:
        """In metadata.json order (the order snapshots are keyed by)."""
        return list(self._edge_meta)

    def num_nodes(self, nt: str) -> int:
        return int(self.meta["node_types"][nt]["num_nodes"])

    def edge_meta(self, et: EdgeType) -> dict:
        return self._edge_meta[et]

    def tensor(self, key: str) -> np.ndarray:
        with safe_open(str(self.path / "graph.safetensors"), framework="np") as f:
            return f.get_tensor(key)

    def node_tensor(self, nt: str, attr: str) -> np.ndarray:
        return self.tensor(self.meta["node_types"][nt]["tensors"][attr]["key"])

    def edges(self, et: EdgeType) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """edge_index [2, E] int64, edge_time [E] int64, score [E] float64."""
        spec = self._edge_meta[et]["tensors"]
        ei = self.tensor(spec["edge_index"]["key"]).astype(np.int64)
        t = self.tensor(spec["edge_time"]["key"]).astype(np.int64)
        s = self.tensor(spec["edge_attr"]["key"])[:, 0].astype(np.float64)
        return ei, t, s

    def mapping(self, nt: str) -> pd.DataFrame:
        """mappings/<nt>.parquet as stored: index, node_id."""
        return pd.read_parquet(self.path / "mappings" / f"{nt}.parquet")

    def node_ids(self, nt: str) -> np.ndarray:
        """Node id strings in index order."""
        return self.mapping(nt).sort_values("index")["node_id"].astype(str).to_numpy()

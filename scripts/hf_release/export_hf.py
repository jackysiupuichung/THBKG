#!/usr/bin/env python
"""Export the THBKG HeteroData graph to a HuggingFace-ready release directory.

Converts the pickle ``.pt`` artifacts into pickle-free, framework-neutral
formats so the dataset can be published without the "unsafe pickle" banner and
without locking downstream users to a specific PyG version:

  - graph tensors      -> graph.safetensors     (node features, edge_index/attr/time)
  - graph structure    -> metadata.json         (node/edge schema, dtypes, shapes)
  - node id <-> index  -> mappings/<ntype>.parquet
  - advancement labels -> labels/train.parquet, labels/eval.parquet

It also stages ``load_thbkg.py`` (the standalone reassembly helper), the dataset
card and ``croissant.json`` into the output dir, so the whole folder can be
``huggingface-cli upload``-ed as-is.

Heavy: loads the full ~500 MB graph into memory. Run via sbatch (see
export_hf.sh), not in the foreground.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import save_file

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]  # scripts/hf_release/ -> repo root

_DTYPE_STR = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.bool: "bool",
}


def _dtype_name(t: torch.Tensor) -> str:
    return _DTYPE_STR.get(t.dtype, str(t.dtype).replace("torch.", ""))


def export(graph_file, mappings_file, train_csv, test_csv, out_dir,
           card_file, croissant_file):
    out = Path(out_dir)
    (out / "mappings").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    print(f"Loading graph:    {graph_file}")
    data = torch.load(graph_file, weights_only=False)
    print(f"Loading mappings: {mappings_file}")
    mappings = torch.load(mappings_file, weights_only=False)
    node_mapping = mappings.get("node_mapping", {}) if isinstance(mappings, dict) else {}

    tensors: dict[str, torch.Tensor] = {}
    meta = {
        "format": "thbkg-hetero-safetensors",
        "format_version": "1.0",
        "reassemble_with": "load_thbkg.py:load_graph",
        # The training loader (src.data.temporal_loader.load_event_graph) adds
        # reverse edges before message passing. The tensors stored here are the
        # canonical DIRECTED graph; load_thbkg.load_graph(to_undirected=True)
        # reproduces the training-time view.
        "training_view_is_undirected": True,
        "node_types": {},
        "edge_types": [],
        "node_type_mapping": mappings.get("node_type_mapping", {}) if isinstance(mappings, dict) else {},
    }

    # --- nodes ---
    for nt in sorted(data.node_types):
        store = data[nt]
        node_meta = {"num_nodes": int(store.num_nodes), "tensors": {}}
        for attr, val in store.items():
            if not torch.is_tensor(val):
                continue  # skip node_id (list[str]) and scalars like num_nodes
            key = f"node.{nt}.{attr}"
            tensors[key] = val.detach().contiguous().clone()
            node_meta["tensors"][attr] = {
                "key": key, "shape": list(val.shape), "dtype": _dtype_name(val),
            }
        meta["node_types"][nt] = node_meta

        # node_id <-> index parquet (from the stored node_id list, or invert the
        # build-time node_mapping if the graph was loaded without node_id).
        node_ids = getattr(store, "node_id", None)
        if node_ids is None and nt in node_mapping:
            node_ids = [k for k, _ in sorted(node_mapping[nt].items(), key=lambda kv: kv[1])]
        if node_ids is not None:
            pd.DataFrame(
                {"index": range(len(node_ids)), "node_id": [str(x) for x in node_ids]}
            ).to_parquet(out / "mappings" / f"{nt}.parquet", index=False)

    # --- edges ---
    for i, et in enumerate(data.edge_types):
        store = data[et]
        src, rel, dst = et
        edge_meta = {
            "src": src, "rel": rel, "dst": dst,
            "num_edges": int(store.edge_index.size(1)),
            "tensors": {},
        }
        for attr in ("edge_index", "edge_attr", "edge_time"):
            if attr not in store:
                continue
            val = store[attr]
            if not torch.is_tensor(val):
                continue
            key = f"edge.{i}.{attr}"
            tensors[key] = val.detach().contiguous().clone()
            edge_meta["tensors"][attr] = {
                "key": key, "shape": list(val.shape), "dtype": _dtype_name(val),
            }
        meta["edge_types"].append(edge_meta)

    # --- write safetensors + metadata ---
    save_file(tensors, str(out / "graph.safetensors"))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    n_edges = sum(e["num_edges"] for e in meta["edge_types"])
    print(f"graph.safetensors: {len(tensors)} tensors | "
          f"{len(meta['node_types'])} node types, "
          f"{len(meta['edge_types'])} edge types, {n_edges:,} edges")

    # --- labels -> parquet ---
    for csv, name in [(train_csv, "train"), (test_csv, "eval")]:
        df = pd.read_csv(csv)
        df.to_parquet(out / "labels" / f"{name}.parquet", index=False)
        pos = int(df["outcome"].astype(bool).sum()) if "outcome" in df.columns else -1
        print(f"labels/{name}.parquet: {len(df):,} rows (pos={pos:,})")

    # --- stage helper, card, croissant ---
    shutil.copy(_HERE / "load_thbkg.py", out / "load_thbkg.py")
    if Path(card_file).exists():
        shutil.copy(card_file, out / "README.md")
    if Path(croissant_file).exists():
        shutil.copy(croissant_file, out / "croissant.json")

    print(f"\nStaged HF release dir: {out}")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out)}  ({p.stat().st_size:,} B)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph-file", required=True,
                    help="hetero_graph_with_features_*.pt (PyG HeteroData)")
    ap.add_argument("--mappings-file", required=True,
                    help="temporal_graph_*_mappings.pt (build-time node/edge mappings)")
    ap.add_argument("--train-csv",
                    default=str(_REPO_ROOT / "data/clinical_trial_advancement/26.09_w3/train_dataset.csv"))
    ap.add_argument("--test-csv",
                    default=str(_REPO_ROOT / "data/clinical_trial_advancement/26.09_w3/test_dataset.csv"))
    ap.add_argument("--out", required=True, help="output release directory")
    ap.add_argument("--card-file", default=str(_HERE / "README_dataset_card.md"))
    ap.add_argument("--croissant-file", default=str(_REPO_ROOT / "croissant.json"))
    args = ap.parse_args()
    export(args.graph_file, args.mappings_file, args.train_csv, args.test_csv,
           args.out, args.card_file, args.croissant_file)


if __name__ == "__main__":
    main()

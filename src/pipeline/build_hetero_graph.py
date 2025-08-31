#!/usr/bin/env python3
import os
import argparse
import glob
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData

from src import data


def load_nodes(node_dir):
    """
    Load node parquet files into a dict and build id→type lookup.
    Returns:
      - nodes: dict of {node_type: DataFrame}
      - id_to_type: dict mapping node_id → node_type
    """
    nodes = {}
    id_to_type = {}

    for fname in os.listdir(node_dir):
        if fname.endswith(".parquet"):
            node_type = os.path.splitext(fname)[0]
            node_df = pd.read_parquet(os.path.join(node_dir, fname))
            nodes[node_type] = node_df
            print(f"  - {node_type}: {len(node_df)} nodes")

            for nid in node_df["id"].astype(str).tolist():
                id_to_type[nid] = node_type

    return nodes, id_to_type

def load_edges(edge_dir):
    """Load all edge parquet files (including subdirectories) into a single DataFrame."""
    files = glob.glob(os.path.join(edge_dir, "**", "*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {edge_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def get_most_evidented_edges(edges, relation_mode="datatype"):
    """
    Get the most evidenced edges based on a specific relation mode.
    """
    if relation_mode == "datatype":
        edges["rel_key"] = edges["relation"]
    elif relation_mode == "source":
        edges["rel_key"] = edges["sourceId"]
    else:
        raise ValueError("relation_mode must be 'datatype' or 'source'")

    # Sort so highest score is first
    edges = edges.sort_values("score", ascending=False)

    # Deduplicate while keeping source_type & target_type
    keep_cols = ["source", "target", "rel_key", "score", "source_type", "target_type"]
    if "year" in edges.columns:
        keep_cols.append("year")
    if "datasourceId" in edges.columns:
        keep_cols.append("datasourceId")

    edges = edges.drop_duplicates(subset=["source", "target", "rel_key"], keep="first")[keep_cols]

    return edges



def temporal_split(edges, cutoff, test_horizon=5):
    """
    Split into:
      - train: edges with publicationYear <= cutoff
      - test:  edges in (cutoff, cutoff+test_horizon]
    """
    train_edges = edges[edges["year"] <= cutoff]
    test_edges = edges[
        (edges["year"] > cutoff)
        & (edges["year"] <= cutoff + test_horizon)
    ]
    return train_edges, test_edges



def build_heterodata(nodes, train_edges, test_edges, user_map, item_map):
    """
    Build a PyG HeteroData object from nodes and train/test edges.
    Expects edge DataFrames to include: source, target, relation, source_type, target_type.
    """
    data = HeteroData()

    # === Nodes ===
    id_maps = {}
    for node_type, node_df in nodes.items():
        if node_type == "diseases" and user_map:
            id_map = user_map
        elif node_type == "targets" and item_map:
            id_map = item_map
        else:
            ids = node_df["id"].astype(str).tolist()
            id_map = {nid: i for i, nid in enumerate(ids)}
        id_maps[node_type] = id_map
        node_df["mapped_id"] = node_df["id"].map(id_map)
        nodes[node_type] = node_df
        data[node_type].num_nodes = len(node_df)

    # === Edges ===
    for split_name, edge_df in [("train", train_edges), ("test", test_edges)]:
        for (src_type, rel_name, dst_type), group in edge_df.groupby(
            ["source_type", "rel_key", "target_type"]
        ):
            # map IDs to integer indices
            src_ids = [id_maps[src_type][s] for s in group["source"].astype(str)]
            dst_ids = [id_maps[dst_type][t] for t in group["target"].astype(str)]
            edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)

            # add to heterodata
            data[(src_type, rel_name, dst_type)].edge_index = edge_index

            # add score if available
            if "score" in group.columns:
                data[(src_type, rel_name, dst_type)].edge_score = torch.tensor(
                    group["score"].values, dtype=torch.float
                )

            # add split mask
            if split_name == "train":
                data[(src_type, rel_name, dst_type)].train_mask = torch.ones(edge_index.size(1), dtype=torch.bool)
                data[(src_type, rel_name, dst_type)].test_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
            else:
                data[(src_type, rel_name, dst_type)].train_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
                data[(src_type, rel_name, dst_type)].test_mask = torch.ones(edge_index.size(1), dtype=torch.bool)

    return data

from torch_geometric.data import HeteroData
import torch

def build_heterodata_with_cold_split(
    nodes, all_edges, train_df, valid_df, test_df, cutoff: int, horizon: int = 5,
    supervision_source="diseases", supervision_target="targets", supervision_relation="clinical_trial"
):
    """
    Build HeteroData with train/valid/test masks for (supervision_source, supervision_relation, supervision_target).
    Other edge types are kept fully, but filtered to nodes we actually loaded.
    """
    # === Sanity checks for leakage ===
    if "year" in all_edges.columns:
        max_year = all_edges["year"].max()
        if max_year > cutoff:
            raise ValueError(
                f"❌ Leakage in context edges: found year={max_year} > cutoff={cutoff}"
            )

    for name, df, low, high in [
        ("train", train_df, None, cutoff),
        ("valid", valid_df, cutoff, cutoff + horizon),
        ("test", test_df, cutoff, cutoff + horizon),
    ]:
        if "year" not in df.columns:
            continue
        min_y, max_y = df["year"].min(), df["year"].max()

        if name == "train" and max_y > cutoff:
            raise ValueError(
                f"❌ Leakage in train_df: contains edges after cutoff "
                f"(max year={max_y}, cutoff={cutoff})"
            )
        if name == "valid" and (min_y <= cutoff or max_y > cutoff + horizon):
            raise ValueError(
                f"❌ Leakage in valid_df: expected ({cutoff}, {cutoff+horizon}], "
                f"but found years {min_y}–{max_y}"
            )
        if name == "test" and (min_y <= cutoff or max_y > cutoff + horizon):
            raise ValueError(
                f"❌ Leakage in test_df: expected ({cutoff}, {cutoff+horizon}], "
                f"but found years {min_y}–{max_y}"
            )

    data = HeteroData()

    # === Nodes ===
    id_maps = {}
    for node_type, node_df in nodes.items():
        ids = node_df["id"].astype(str).tolist()
        id_map = {nid: i for i, nid in enumerate(ids)}
        id_maps[node_type] = id_map
        node_df["mapped_id"] = node_df["id"].map(id_map)
        data[node_type].num_nodes = len(node_df)

    # === Helper for adding masked supervision edges ===
    def add_supervision_edges(df, mask_name):
        if len(df) == 0:
            return None

        src_ids = [id_maps[supervision_source][s] for s in df["user_id"].astype(str) if s in id_maps[supervision_source]]
        dst_ids = [id_maps[supervision_target][t] for t in df["item_id"].astype(str) if t in id_maps[supervision_target]]

        if not src_ids or not dst_ids:
            return None

        edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
        key = (supervision_source, supervision_relation, supervision_target)

        if key not in data.edge_types:
            data[key].edge_index = edge_index
            data[key].edge_score = torch.tensor(df["label"].values, dtype=torch.float)

            # initialise masks
            data[key].train_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
            data[key].valid_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
            data[key].test_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        else:
            # append new edges
            old_ei = data[key].edge_index
            old_score = data[key].edge_score
            data[key].edge_index = torch.cat([old_ei, edge_index], dim=1)
            data[key].edge_score = torch.cat(
                [old_score, torch.tensor(df["label"].values, dtype=torch.float)]
            )

            # extend masks
            for m in ["train_mask", "valid_mask", "test_mask"]:
                old_mask = data[key][m]
                data[key][m] = torch.cat([old_mask, torch.zeros(edge_index.size(1), dtype=torch.bool)])

        # set the relevant mask to True for these new edges
        data[key][mask_name][-edge_index.size(1):] = True

    # === Supervised edge splits ===
    add_supervision_edges(train_df, "train_mask")
    add_supervision_edges(valid_df, "valid_mask")
    add_supervision_edges(test_df, "test_mask")

    # === Other edges (keep fully, no split) ===
    others = all_edges[all_edges["rel_key"] != supervision_relation]
    for (src_type, rel_name, dst_type), group in others.groupby(
        ["source_type", "rel_key", "target_type"]
    ):
        src_ids = [id_maps[src_type][s] for s in group["source"].astype(str) if s in id_maps[src_type]]
        dst_ids = [id_maps[dst_type][t] for t in group["target"].astype(str) if t in id_maps[dst_type]]
        if not src_ids or not dst_ids:
            continue
        edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
        data[(src_type, rel_name, dst_type)].edge_index = edge_index
        if "score" in group.columns:
            data[(src_type, rel_name, dst_type)].edge_score = torch.tensor(
                group["score"].values, dtype=torch.float
            )

    return data




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-dir", required=True, help="Directory with edge parquet files")
    parser.add_argument("--node-dir", required=True, help="Directory with node parquet files")
    parser.add_argument("--cutoff", type=int, default=2015, help="Training cutoff year")
    parser.add_argument("--test-horizon", type=int, default=5, help="Number of years after cutoff for test set")
    parser.add_argument("--relation-mode", choices=["datatype", "source"], default="datatype")
    parser.add_argument("--out", required=True, help="Output torch file path (.pt)")
    args = parser.parse_args()

    print("📂 Loading nodes...")
    nodes, id_to_type = load_nodes(args.node_dir)

    print("📂 Loading edges...")
    edges = load_edges(args.edge_dir)
    print(f"✅ Loaded {len(edges)} edges")

    print("🔗 Annotating edge types from node lookup...")
    edges["source_type"] = edges["source"].astype(str).map(id_to_type)
    edges["target_type"] = edges["target"].astype(str).map(id_to_type)

    missing_src = edges["source_type"].isna().sum()
    missing_tgt = edges["target_type"].isna().sum()
    if missing_src or missing_tgt:
        print(f"⚠️ Missing type for {missing_src} sources, {missing_tgt} targets")

    print("⏳ Splitting into train/test...")
    train_edges, test_edges = temporal_split(edges, cutoff=args.cutoff, test_horizon=args.test_horizon)
    print(f"✅ Train edges (raw): {len(train_edges)}, Test edges (raw): {len(test_edges)}")

    print("🧹 Deduplicating train/test separately...")
    train_edges = deduplicate_edges(train_edges, relation_mode=args.relation_mode)
    test_edges = deduplicate_edges(test_edges, relation_mode=args.relation_mode)
    print(f"✅ Train edges after dedup: {len(train_edges)}, Test edges after dedup: {len(test_edges)}")

    graph_object = build_heterodata(nodes, train_edges, test_edges)
    print(graph_object)
    print("Metadata:", graph_object.metadata())

    torch.save(graph_object, args.out)
    print(f"✅ Saved hetero graph to {args.out}")

if __name__ == "__main__":
    main()

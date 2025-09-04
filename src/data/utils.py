import pandas as pd
import os
import argparse

import torch
import numpy as np
from torch_geometric.data import HeteroData


def evidence_edge_temporal_split(
    evidence_edges,
    cutoff: int = 2015,
    out_dir="data/"
):
    # Filter for relevant evidence edges before cutoff
    evidence_edges = evidence_edges[evidence_edges["year"] <= cutoff]
    return evidence_edges

def supervision_edge_temporal_and_cold_split(
    parquet_path: str,
    cutoff: int = 2015,
    horizon: int = 5,
    cold_start_diseases=None,
    out_dir="data/"
):
    """
    Split chembl edges into train/valid/test with temporal cutoff and cold-start targets.

    Args:
      parquet_path: path to chembl parquet file
      cutoff: training cutoff year
      horizon: years after cutoff for test set
      cold_start_diseases: list of disease IDs reserved for cold-start (excluded from train)
      out_dir: directory to save train/valid/test CSVs
    """

    out_dir = os.path.join(out_dir, "dataframe")    
    os.makedirs(out_dir, exist_ok=True)

    edges = pd.read_parquet(parquet_path)
    print(f"✅ Loaded {len(edges)} edges")

    # filter for only clinical_trial relations between target and disease for positive sampling
    edges = edges[edges["relation"] == "clinical_trial"].copy()

    if "year" not in edges.columns:
        raise ValueError("Edges parquet must contain 'year' column for temporal split")
    if "score" not in edges.columns:
        raise ValueError("Edges parquet must contain 'score' column to use as label")

    # --------------------------
    # Define cold-start target set
    # --------------------------
    cold_start_diseases = set(str(x) for x in (cold_start_diseases or []))
    print(f"✅ Using {len(cold_start_diseases)} cold-start diseases")

    # --------------------------
    # TRAIN = edges before cutoff, excluding cold-start users
    # TODO: do you exclude cold_start here too
    # --------------------------
    train_df = edges[
        (edges["year"] <= cutoff)
        & (~edges["source"].astype(str).isin(cold_start_diseases))
    ].copy()

    # --------------------------
    # VALID = edges after cutoff (<= cutoff+horizon) for non–cold-start users
    # --------------------------
    valid_df = edges[
        (edges["year"] > cutoff)
        & (edges["year"] <= cutoff + horizon)
        & (~edges["source"].astype(str).isin(cold_start_diseases))
    ].copy()

    # --------------------------
    # TEST = edges after cutoff (<= cutoff+horizon) for cold-start users only
    # --------------------------
    test_df = edges[
        (edges["year"] > cutoff)
        & (edges["year"] <= cutoff + horizon)
        & (edges["source"].astype(str).isin(cold_start_diseases))
    ].copy()

    # Cold-start users must not leak into train/valid
    assert not train_df["source"].astype(str).isin(cold_start_diseases).any(), \
        "❌ Cold-start users found in train!"
    assert not valid_df["source"].astype(str).isin(cold_start_diseases).any(), \
        "❌ Cold-start users found in valid!"

    # Test must contain only cold-start users
    if len(test_df) > 0:
        assert test_df["source"].astype(str).isin(cold_start_diseases).all(), \
            "❌ Non cold-start users found in test!"
    else:
        print("⚠️ WARNING: No cold-start interactions found in test set.")

    # --------------------------
    # Convert to RecBole format
    # --------------------------
    def to_recbole(df):
        # force numeric and check for NaN
        labels = pd.to_numeric(df["score"], errors="coerce")
        if labels.isna().any():
            bad_rows = df[labels.isna()]
            raise ValueError(
                f"❌ Found NaN or non-numeric scores in {len(bad_rows)} rows:\n{bad_rows.head()}"
            )
        return pd.DataFrame({
            "user_id": df["source"].astype(str),
            "item_id": df["target"].astype(str),
            "label": labels.astype(float)
        })


    train_out = to_recbole(train_df)
    valid_out = to_recbole(valid_df)
    test_out = to_recbole(test_df)

    os.makedirs(out_dir, exist_ok=True)
    train_out.to_csv(f"{out_dir}/train.csv", index=False)
    valid_out.to_csv(f"{out_dir}/valid.csv", index=False)
    test_out.to_csv(f"{out_dir}/test.csv", index=False)

    print(f"💾 Saved splits → {out_dir}")
    print(f"   Train: {len(train_out)}, Valid: {len(valid_out)}, Test (cold-start): {len(test_out)}")
    return train_out, valid_out, test_out

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
            data[key].edge_attr = torch.tensor(df["label"].values, dtype=torch.float)

            # initialise masks
            data[key].train_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
            data[key].valid_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
            data[key].test_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        else:
            # append new edges
            old_ei = data[key].edge_index
            old_attr = data[key].edge_attr
            data[key].edge_index = torch.cat([old_ei, edge_index], dim=1)
            data[key].edge_attr = torch.cat(
                [old_attr, torch.tensor(df["label"].values, dtype=torch.float)]
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
            data[(src_type, rel_name, dst_type)].edge_attr = torch.tensor(
                group["score"].values, dtype=torch.float
            )

    return data, id_maps



def attach_node_features(hetero_graph, id_maps, embeddings=None, emb_dim=64, seed=42):
    rng = np.random.default_rng(seed)
    for ntype, id_map in id_maps.items():
        num_nodes = len(id_map)
        if embeddings and ntype in embeddings:
            emb = torch.tensor(embeddings[ntype], dtype=torch.float)
            assert emb.shape[0] == num_nodes, (
                f"❌ Embedding rows for {ntype} ({emb.shape[0]}) "
                f"!= num_nodes in graph ({num_nodes})"
            )
            hetero_graph[ntype].x = emb
        else:
            hetero_graph[ntype].x = torch.tensor(
                rng.normal(size=(num_nodes, emb_dim)), dtype=torch.float
            )
    return hetero_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="Path to chembl edges parquet")
    parser.add_argument("--cutoff", type=int, default=2015, help="Training cutoff year")
    parser.add_argument("--horizon", type=int, default=5, help="Years after cutoff for test set")
    parser.add_argument("--cold-start-targets", nargs='*', default=None, help="List of target IDs for cold-start")
    parser.add_argument("--out-dir", default="data/", help="Output directory for train/valid/test CSVs")
    args = parser.parse_args()

    supervision_edge_temporal_and_cold_split(
        parquet_path=args.parquet,
        cutoff=args.cutoff,
        horizon=args.horizon,
        cold_start_diseases=args.cold_start_diseases,
        out_dir=args.out_dir
    )

import pandas as pd
import os
import argparse


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

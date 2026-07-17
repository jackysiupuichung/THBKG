"""Supplementary table: train & eval pairs and positives, stratified by therapeutic area.

Counting matches RS-by-TA in evaluate_advancement.py: a pair is counted in every TA
its disease maps to (multi-membership), so per-TA rows do not sum to the split total.
The 'all' row gives the true de-duplicated totals.
"""

import json
from pathlib import Path

import pandas as pd
import xarray as xr

REPO = Path("/data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph")
DATA = Path("/gpfs/scratch/bty414/clinical_advancement_paper/data/datasets_26.03_w3")
SPLITS = {
    "train": DATA / "training_dataset.zarr",
    "eval": DATA / "evaluation_dataset.zarr",
}
TA_PARQUET = REPO / "advancement_data/features/therapeutic_areas.parquet"
PRIMARY_TAS = REPO / "advancement_data/results/primary_therapeutic_areas.json"
OUT_DIR = REPO / "headline_results"
OUT_DIR.mkdir(exist_ok=True)

ta_df = pd.read_parquet(TA_PARQUET)
ta_base = ta_df[["disease_id", "therapeutic_area_name"]].drop_duplicates()
with open(PRIMARY_TAS) as f:
    primary = set(json.load(f))


def split_summary(name: str, zarr: Path) -> tuple[pd.DataFrame, int, int, int]:
    ds = xr.open_zarr(zarr).load()
    outcomes = pd.DataFrame(
        {
            "target_id": ds.target_id.values,
            "disease_id": ds.disease_id.values,
            "outcome": ds.outcome.squeeze("outcomes").values.astype(int),
        }
    )
    n_total = len(outcomes)
    n_pos = int(outcomes["outcome"].sum())

    # synthetic 'all' TA -> every pair once (true totals)
    all_ta = pd.DataFrame(
        {"disease_id": outcomes["disease_id"].unique(), "therapeutic_area_name": "all"}
    )
    ta_map = pd.concat([ta_base, all_ta], ignore_index=True)
    merged = outcomes.merge(ta_map, on="disease_id", how="inner")

    summ = (
        merged.groupby("therapeutic_area_name")
        .agg(n_pairs=("outcome", "size"), n_positives=("outcome", "sum"))
        .reset_index()
    )
    summ.columns = ["therapeutic_area_name", f"{name}_pairs", f"{name}_positives"]

    pairs_with_primary = (
        merged[merged["therapeutic_area_name"].isin(primary - {"all"})][
            ["target_id", "disease_id"]
        ]
        .drop_duplicates()
        .shape[0]
    )
    return summ, n_total, n_pos, pairs_with_primary


parts, totals = {}, {}
for name, zarr in SPLITS.items():
    summ, n_total, n_pos, n_prim = split_summary(name, zarr)
    parts[name] = summ
    totals[name] = (n_total, n_pos, n_prim)

# merge splits on TA
table = parts["train"].merge(parts["eval"], on="therapeutic_area_name", how="outer").fillna(0)
for c in table.columns[1:]:
    table[c] = table[c].astype(int)
table["train_pct_pos"] = (100 * table["train_positives"] / table["train_pairs"]).round(2)
table["eval_pct_pos"] = (100 * table["eval_positives"] / table["eval_pairs"]).round(2)

# restrict to primary TA set (+ 'all'); flag dropped
in_primary = table[table["therapeutic_area_name"].isin(primary)].copy()
dropped = table[~table["therapeutic_area_name"].isin(primary)]

in_primary["_ord"] = in_primary["therapeutic_area_name"].eq("all").map({True: 0, False: 1})
in_primary = in_primary.sort_values(["_ord", "eval_pairs"], ascending=[True, False]).drop(
    columns="_ord"
)

cols = [
    "therapeutic_area_name",
    "train_pairs",
    "train_positives",
    "train_pct_pos",
    "eval_pairs",
    "eval_positives",
    "eval_pct_pos",
]
in_primary = in_primary[cols]

print("=== 26.03 w3 train/eval TA-stratified summary ===")
for name in ("train", "eval"):
    n_total, n_pos, n_prim = totals[name]
    print(
        f"{name:>5}: {n_total} pairs, {n_pos} positives ({100*n_pos/n_total:.2f}%); "
        f"{n_prim} ({100*n_prim/n_total:.1f}%) map to >=1 primary TA"
    )
print()
print(in_primary.to_string(index=False))
if len(dropped):
    print("\n--- non-primary TAs (excluded) ---")
    print(dropped.sort_values("eval_pairs", ascending=False).to_string(index=False))

caption = [
    "# Supplementary table: 26.03 w3 clinical-advancement dataset, pairs & positives by therapeutic area.",
    "# Counting is multi-membership: a target-disease pair is counted in EVERY therapeutic area its",
    "# disease maps to, matching the RS-by-TA computation in evaluate_advancement.py. Per-TA rows therefore",
    "# OVERLAP and do NOT sum to the split totals. De-duplicated totals: train = 21602 pairs / 4917 pos",
    "# (22.76%); eval = 7193 pairs / 668 pos (9.29%). The 'all' row gives overlapping (not de-duplicated)",
    "# totals across primary TAs. The train/eval positive-rate gap reflects decision-aligned temporal",
    "# masking (later, evidence-sparse eval pairs), not a split artifact. 13 primary TAs shown; ~15%",
    "# of pairs map only to non-primary TAs and are omitted here.",
]
csv_path = OUT_DIR / "traineval_ta_summary_w3.csv"
with open(csv_path, "w") as fh:
    fh.write("\n".join(caption) + "\n")
    in_primary.to_csv(fh, index=False)
print(f"\nwrote {csv_path}")

tex = in_primary.rename(
    columns={
        "therapeutic_area_name": "Therapeutic area",
        "train_pairs": "Train pairs",
        "train_positives": "Train pos.",
        "train_pct_pos": "Train \\%",
        "eval_pairs": "Eval pairs",
        "eval_positives": "Eval pos.",
        "eval_pct_pos": "Eval \\%",
    }
)
tex_path = OUT_DIR / "traineval_ta_summary_w3.tex"
tex.to_latex(tex_path, index=False, float_format="%.2f", escape=False)
print(f"wrote {tex_path}")

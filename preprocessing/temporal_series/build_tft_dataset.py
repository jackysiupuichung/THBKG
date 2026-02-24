#!/usr/bin/env python3
"""
Step 05: Build TFT longitudinal dataset from pre-parsed edge parquets.

Inherits output from Step 01 (collecting_edges_01.sh):
  - raw edge parquets:  output/evidences/edges/
  - ChEMBL edges:       output/evidences/edges/target_clinical_trial_disease_chembl*.parquet

For each TD pair:
  - Anchor:  first year pair reaches Phase 2 (ChEMBL)
  - Mask:    all features use only evidence_year < anchor_year
  - Window:  T-{lookback} to T-1 (relative to anchor)
  - Outcome: 1 if pair reaches Phase 3+ in (anchor_year, outcome_max_year]

Output: longitudinal parquet keyed by (targetId, diseaseId, relative_year)
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
from typing import Tuple

# --- Import shared utilities from Step 01 pipeline ---
_PIPELINE_DIR = str(Path(__file__).resolve().parents[1] / "temporal_graph" / "pipeline")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, _PIPELINE_DIR)

from build_event_list import harmonic_sum, aggregate_scores, add_datatype_info


# ──────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────

def load_edges(edges_dir: str, sample_ratio: float = None) -> pd.DataFrame:
    """Load all pre-parsed edge parquets from the Step 01 output directory."""
    parquet_files = glob(os.path.join(edges_dir, "*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {edges_dir}")

    print(f"📂 Loading {len(parquet_files)} edge parquet files from {edges_dir}")
    dfs = []
    for pf in tqdm(parquet_files, desc="Loading edges"):
        try:
            df = pd.read_parquet(pf)
            if sample_ratio:
                df = df.sample(frac=sample_ratio, random_state=42)
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Error reading {Path(pf).name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"   Loaded {len(combined):,} total edge records")
    return combined


# ──────────────────────────────────────────────
# Anchor Table
# ──────────────────────────────────────────────

def build_anchor_table(
    edges: pd.DataFrame,
    train_max: int,
    val_max: int,
    test_max: int,
    outcome_max: int,
) -> pd.DataFrame:
    """
    Identify T=0 (first Phase 2 entry) for each TD pair from ChEMBL.
    Assign partition tags and binary outcome labels.

    Returns:
        DataFrame with columns: [sourceId, targetId, anchor_year, partition, outcome]
    """
    # Use edges that have a year and a ChEMBL clinical phase signal
    phase_cols = [c for c in edges.columns if 'phase' in c.lower() or 'relation' in c.lower()]
    year_col = 'year' if 'year' in edges.columns else None

    if not year_col:
        raise ValueError("Edge parquets must have a 'year' column. Run Step 01 with build_event_list.py first.")

    # Filter to ChEMBL clinical trial edges
    chembl_mask = edges['datasource'].str.contains('chembl', case=False, na=False) if 'datasource' in edges.columns \
        else edges['sourceId'].isin(['chembl']) if 'sourceId' in edges.columns \
        else pd.Series([True] * len(edges))

    # Detect relation column for Phase 2+ filter
    relation_col = next((c for c in edges.columns if 'relation' in c.lower()), None)

    chembl = edges[chembl_mask].copy()
    if relation_col:
        phase2_mask = chembl[relation_col].str.contains('phase_2|phase_3|phase_4|phase2|phase3|phase4',
                                                         case=False, na=False)
        chembl = chembl[phase2_mask]

    chembl = chembl[chembl[year_col].notna()].copy()
    chembl[year_col] = chembl[year_col].astype(int)

    # T=0 = first year in Phase 2+
    anchors = chembl.groupby(['sourceId', 'targetId'])[year_col].min().reset_index()
    anchors = anchors.rename(columns={year_col: 'anchor_year', 'sourceId': 'diseaseId_raw', 'targetId': 'targetId_raw'})

    # Detect actual column names
    id_cols = [c for c in ['sourceId', 'targetId'] if c in chembl.columns]
    anchors = chembl.groupby(id_cols)[year_col].min().reset_index().rename(columns={year_col: 'anchor_year'})

    print(f"   Found {len(anchors):,} TD pairs with a Phase 2+ entry")

    # Partition tagging
    print(f"🗓  Partitions: Train ≤ {train_max}, Val [{train_max+1}–{val_max}], Test [{val_max+1}–{test_max}]")
    anchors['partition'] = 'excluded'
    anchors.loc[(anchors['anchor_year'] >= 1990) & (anchors['anchor_year'] <= train_max), 'partition'] = 'train'
    anchors.loc[(anchors['anchor_year'] > train_max) & (anchors['anchor_year'] <= val_max),  'partition'] = 'val'
    anchors.loc[(anchors['anchor_year'] > val_max)   & (anchors['anchor_year'] <= test_max), 'partition'] = 'test'

    anchors = anchors[anchors['partition'] != 'excluded'].copy()
    print(f"   Partition counts:\n{anchors['partition'].value_counts()}")

    # Outcome: did the pair reach Phase 3/4 AFTER anchor year?
    if relation_col:
        phase3_mask = (
            edges[relation_col].str.contains('phase_3|phase_4|phase3|phase4', case=False, na=False)
            & edges[year_col].notna()
            & (edges[year_col].astype(float) > anchors['anchor_year'].max())  # rough pre-filter
        )
        future = edges[phase3_mask & (edges[year_col].astype(float) <= outcome_max)].copy()
    else:
        future = pd.DataFrame()

    if not future.empty:
        future_id_cols = [c for c in id_cols if c in future.columns]
        positive_pairs = set(map(tuple, future[future_id_cols].drop_duplicates().values))

        def is_positive(row):
            return int(tuple(row[c] for c in id_cols) in positive_pairs)

        anchors['outcome'] = anchors.apply(is_positive, axis=1)
    else:
        print("   ⚠️ Could not detect Phase 3/4 transitions — setting all outcomes to 0")
        anchors['outcome'] = 0

    return anchors


# ──────────────────────────────────────────────
# Dynamic Feature Extraction
# ──────────────────────────────────────────────

def build_dynamic_features(
    edges: pd.DataFrame,
    anchors: pd.DataFrame,
    id_cols: list,
    lookback: int,
) -> pd.DataFrame:
    """
    Build source-level harmonic association scores and novelty scores per TD pair per relative year.

    All features use ONLY evidence from evidence_year < anchor_year (pair-specific mask).

    Returns long-format DataFrame with columns:
      [*id_cols, relative_year, {source}_S, {source}_N, ...]
    """
    print("📈 Building dynamic source-level sequences...")

    year_col = 'year'
    source_col = 'datasource' if 'datasource' in edges.columns else \
                 'datasourceId' if 'datasourceId' in edges.columns else None
    score_col  = 'score' if 'score' in edges.columns else \
                 'edge_weight' if 'edge_weight' in edges.columns else None

    if not source_col or not score_col:
        print(f"   ⚠️ Could not find source ({source_col}) or score ({score_col}) columns; skipping dynamic features.")
        return pd.DataFrame()

    edges = edges[edges[year_col].notna()].copy()
    edges[year_col] = edges[year_col].astype(int)

    # Join anchor_year onto evidence (vectorized merge)
    merged = edges.merge(anchors[id_cols + ['anchor_year']], on=id_cols, how='inner')

    # Pair-specific mask: use only pre-anchor evidence
    merged = merged[merged[year_col] < merged['anchor_year']].copy()
    merged['relative_year'] = merged[year_col] - merged['anchor_year']
    merged = merged[merged['relative_year'] >= -lookback]

    if merged.empty:
        print("   ⚠️ No historical evidence found within lookback window.")
        return pd.DataFrame()

    print(f"   {len(merged):,} historical evidence records within lookback window")

    # Vectorized harmonic sum per (TD pair, source, relative_year)
    group_cols = id_cols + [source_col, 'relative_year']
    print("   Computing harmonic association scores (S)...")
    agg = merged.groupby(group_cols, as_index=False).agg(
        S=(score_col, lambda x: harmonic_sum(x.values))
    )

    # Novelty score N = S_t - S_{t-1} (shifted within each TD pair × source group)
    agg = agg.sort_values(id_cols + [source_col, 'relative_year'])
    agg['S_prev'] = agg.groupby(id_cols + [source_col])['S'].shift(1).fillna(0)
    agg['N'] = (agg['S'] - agg['S_prev']).clip(lower=0)  # positive novelty only

    # Pivot: one row per (TD pair, relative_year), columns per source
    print("   Pivoting to wide format (source columns)...")
    s_wide = agg.pivot_table(index=id_cols + ['relative_year'], columns=source_col, values='S', fill_value=0)
    n_wide = agg.pivot_table(index=id_cols + ['relative_year'], columns=source_col, values='N', fill_value=0)

    s_wide.columns = [f"{c}_S" for c in s_wide.columns]
    n_wide.columns = [f"{c}_N" for c in n_wide.columns]

    wide = s_wide.join(n_wide).reset_index()
    return wide


# ──────────────────────────────────────────────
# Main Orchestration
# ──────────────────────────────────────────────

def build_tft_dataset(
    raw_edges_dir: str,
    output_dir: str,
    train_max: int = 2014,
    val_max:   int = 2015,
    test_max:  int = 2022,
    outcome_max: int = 2024,
    lookback:  int = 10,
    sample_ratio: float = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load pre-parsed edge parquets from Step 01
    edges = load_edges(raw_edges_dir, sample_ratio=sample_ratio)

    # Detect TD identifier columns
    id_cols = []
    for cand in [('sourceId', 'targetId'), ('diseaseId', 'targetId')]:
        if all(c in edges.columns for c in cand):
            id_cols = list(cand)
            break
    if not id_cols:
        raise ValueError(f"Could not detect TD identifier columns. Available: {edges.columns.tolist()}")

    print(f"   TD identifier columns: {id_cols}")

    # 2. Build anchor table
    print("\n⚓ Building anchor table (T=0 = first Phase 2 entry)...")
    anchors = build_anchor_table(edges, train_max, val_max, test_max, outcome_max)

    # 3. Build dynamic source-level sequences
    dynamic = build_dynamic_features(edges, anchors, id_cols, lookback)

    # 4. Fill time-series grid (ensure all relative years are present for each TD pair)
    print("🧩 Filling time-series grid...")
    grid = anchors[id_cols + ['anchor_year', 'partition', 'outcome']].copy()
    year_range = pd.DataFrame({'relative_year': range(-lookback, 0)})
    grid = grid.assign(_key=1).merge(year_range.assign(_key=1), on='_key').drop('_key', axis=1)

    if not dynamic.empty:
        final = grid.merge(dynamic, on=id_cols + ['relative_year'], how='left')
    else:
        final = grid.copy()

    # Fill NAs
    feature_cols = [c for c in final.columns if c.endswith('_S') or c.endswith('_N')]
    final[feature_cols] = final[feature_cols].fillna(0)

    # 5. Summary
    print(f"\n📊 DATASET SUMMARY")
    print("="*60)
    pairs = final.drop_duplicates(id_cols)
    summary = pairs.groupby(['partition', 'outcome']).size().unstack(fill_value=0)
    if 1 in summary.columns and 0 in summary.columns:
        summary['pos_rate (%)'] = (summary[1] / (summary[0] + summary[1]) * 100).round(2)
    print("\nPair Outcomes per Partition:")
    print(summary)
    print(f"\nTime steps per pair: {lookback}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:6]}{'...' if len(feature_cols) > 6 else ''}")
    print("="*60)

    # 6. Save
    out_file = output_path / "tft_longitudinal.parquet"
    print(f"\n💾 Saving to {out_file}...")
    final.to_parquet(out_file, index=False)

    anchors_file = output_path / "tft_anchors.parquet"
    anchors.to_parquet(anchors_file, index=False)
    print(f"💾 Anchor table saved to {anchors_file}")
    print("✅ Done!")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 05: Build TFT longitudinal dataset from Step 01 edge parquets"
    )
    parser.add_argument("--raw-edges-dir", default="output/evidences/edges",
                        help="Directory with pre-parsed edge parquets (output of kg_pipeline / build_event_list)")
    parser.add_argument("--output-dir",    default="output/tft_dataset",
                        help="Output directory for TFT dataset parquets")
    parser.add_argument("--train-max",  type=int, default=2014, help="Max anchor year for train set")
    parser.add_argument("--val-max",    type=int, default=2015, help="Max anchor year for val set")
    parser.add_argument("--test-max",   type=int, default=2022, help="Max anchor year for test set")
    parser.add_argument("--outcome-max", type=int, default=2024, help="Max year for outcome window")
    parser.add_argument("--lookback",   type=int, default=10,   help="Lookback window in years (T-N to T-1)")
    parser.add_argument("--sample-ratio", type=float, default=None,
                        help="Sample fraction of edges (e.g., 0.01 for 1%% for testing)")
    args = parser.parse_args()

    build_tft_dataset(
        raw_edges_dir = args.raw_edges_dir,
        output_dir    = args.output_dir,
        train_max     = args.train_max,
        val_max       = args.val_max,
        test_max      = args.test_max,
        outcome_max   = args.outcome_max,
        lookback      = args.lookback,
        sample_ratio  = args.sample_ratio,
    )


if __name__ == "__main__":
    main()

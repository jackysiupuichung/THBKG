#!/usr/bin/env python3
import os
import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from datetime import datetime

# ----------------------------------------------------
# 1. UTILITIES & CONFIG
# ----------------------------------------------------

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def harmonic_sum(scores, max_harmonic=1.644):
    """
    Compute harmonic sum of top-50 scores (Open Targets standard).
    """
    if len(scores) == 0:
        return 0.0
    s = np.sort(scores)[::-1][:50]
    idx = np.arange(1, len(s) + 1)
    return np.sum(s / (idx ** 2)) / max_harmonic

def load_evidence(directory, is_static=False):
    """
    Load all parquet files from a directory.
    """
    dfs = []
    # Search for all parquets in the directory
    parquet_files = glob(os.path.join(directory, "*.parquet"))
    
    for pq in parquet_files:
        try:
            df = pd.read_parquet(pq)
            if df.empty:
                continue
            
            # For dynamic files, ensure 'year' is numeric and handled
            if not is_static and "year" in df.columns:
                if df["year"].isnull().any():
                    raise ValueError(f"⚠️ {pq} has null values in 'year' column. Cannot build progression graph.")
                df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
            
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {pq}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

# ----------------------------------------------------
# 2. INSPECTION LOGIC
# ----------------------------------------------------

def inspect_evidence_graph(dynamic_evd, static_evd):
    """
    Summarize raw input evidence.
    """
    print("\n🔍 ================ EVIDENCE GRAPH SUMMARY ================")
    
    all_evd = pd.concat([dynamic_evd, static_evd], ignore_index=True)
    if all_evd.empty:
        print("Empty evidence graph.")
        return

    # Node summary
    unique_sources = set(all_evd["sourceId"].unique())
    unique_targets = set(all_evd["targetId"].unique())
    all_nodes = unique_sources | unique_targets
    
    print(f"🌐 Total uniquely identified nodes: {len(all_nodes)}")
    
    # Node types
    source_types = all_evd.groupby("source_type")["sourceId"].nunique()
    target_types = all_evd.groupby("target_type")["targetId"].nunique()
    
    print("\n🟦 Node counts by type:")
    for t, count in source_types.items():
        print(f"  - {t} (as source): {count}")
    for t, count in target_types.items():
        print(f"  - {t} (as target): {count}")

    # Edge summary
    print(f"\n🔗 Total edges (raw evidence records): {len(all_evd)}")
    print(f"📦 Dynamic edges: {len(dynamic_evd)}")
    print(f"📦 Static edges: {len(static_evd)}")
    
    # DataSource stats
    print("\n📚 Edge counts per datasource:")
    print(all_evd["datasourceId"].value_counts().head(10))

    if not dynamic_evd.empty:
        print(f"\n📆 Evidence Year range: {dynamic_evd['year'].min()} - {dynamic_evd['year'].max()}")
    print("============================================================\n")

def inspect_progression_graph(df_dynamic, df_static):
    """
    Summarize aggregated progression data.
    """
    print("\n📈 ================ PROGRESSION GRAPH SUMMARY ================")
    
    if not df_dynamic.empty:
        print(f"🚀 Dynamic progression records: {len(df_dynamic)}")
        unique_dynamic_edges = df_dynamic.groupby(["sourceId", "targetId", "relation", "datasourceId"]).ngroups
        print(f"🔗 Unique dynamic relationships tracked: {unique_dynamic_edges}")
        
        # Temporal growth
        year_growth = df_dynamic.groupby("year").size()
        print("\n⏳ Progression growth over time (top 5 years by count):")
        print(year_growth.sort_index(ascending=False).head(5))
        
    if not df_static.empty:
        print(f"\n✅ Static progression records: {len(df_static)}")
        print(f"🔗 Unique static relationships: {len(df_static)}")
    
    print("==============================================================\n")

# ----------------------------------------------------
# 3. PROGRESSION BUILD
# ----------------------------------------------------

def build_progression(dynamic_evd, static_evd, config, output_dir, use_weights=True):
    """
    Build progression graph by aggregating scores to (source, target, relation, datasourceId) level.
    """
    ds_config = config.get("data_sources", {})
    years_range = config.get("time_range", {"first_year": 2000, "last_year": 2025})
    years = np.arange(years_range["first_year"], years_range["last_year"] + 1)

    # --- Process Dynamic Edges ---
    dynamic_progression = []
    if not dynamic_evd.empty:
        print(f"📊 Aggregating dynamic progression (Use weights: {use_weights})...")
        
        # 1. Map weights from config
        if use_weights:
            dynamic_evd["weight"] = dynamic_evd["datasourceId"].map(lambda x: ds_config.get(x, {}).get("weight", 1.0))
        else:
            dynamic_evd["weight"] = 1.0
            
        dynamic_evd["weighted_score"] = dynamic_evd["score"] * dynamic_evd["weight"]
        
        # 2. Group by edge identity
        # Aggregate on: sourceId, targetId, source_type, target_type, relation, datasourceId
        group_keys = ["sourceId", "targetId", "source_type", "target_type", "relation", "datasourceId"]
        grouped = dynamic_evd.groupby(group_keys)
        
        for keys, group in tqdm(grouped, desc="Processing Dynamic Edges"):
            src, tgt, src_t, tgt_t, rel, ds = keys
            
            # Get specific cutoff for this datasource
            cutoff = ds_config.get(ds, {}).get("cutoff", 0.0)
            
            # Group scores by year for this specific edge
            year_dict = group.groupby("year")["weighted_score"].apply(list).to_dict()
            collected_scores = []
            
            # Compute cumulative harmonic sum year-by-year
            prev_hs = -1.0
            for y in years:
                if y in year_dict:
                    collected_scores.extend(year_dict[y])
                
                if not collected_scores:
                    continue
                    
                hs = harmonic_sum(collected_scores)
                
                # Filter by datasource-specific cutoff
                if hs < cutoff:
                    continue

                # Monotonic filter: only store if the score increases (or first appearance)
                if hs > prev_hs:
                    dynamic_progression.append({
                        "sourceId": src,
                        "targetId": tgt,
                        "source_type": src_t,
                        "target_type": tgt_t,
                        "relation": rel,
                        "datasourceId": ds,
                        "year": y,
                        "score": hs
                    })
                    prev_hs = hs

    df_dynamic = pd.DataFrame(dynamic_progression)
    
    # --- Process Static Edges ---
    # Static edges are simply unique relationships without temporal progression
    df_static = pd.DataFrame()
    if not static_evd.empty:
        print("📌 Processing static edges...")
        df_static = static_evd[["sourceId", "targetId", "source_type", "target_type", "relation", "datasourceId", "score"]].drop_duplicates()
        
        # Apply datasource-specific cutoffs to static edges
        def _filter_static(row):
            ds = row["datasourceId"]
            cutoff = ds_config.get(ds, {}).get("cutoff", 0.0)
            return row["score"] >= cutoff
            
        if not df_static.empty:
            df_static = df_static[df_static.apply(_filter_static, axis=1)]

    # --- Save Outputs ---
    os.makedirs(output_dir, exist_ok=True)
    dynamic_path = os.path.join(output_dir, "source_level_progression_dynamic.parquet")
    static_path = os.path.join(output_dir, "source_level_progression_static.parquet")
    
    df_dynamic.to_parquet(dynamic_path, index=False)
    df_static.to_parquet(static_path, index=False)
    
    print(f"💾 Saved Dynamic: {dynamic_path}")
    print(f"💾 Saved Static:  {static_path}")

    # --- Metadata Tracking ---
    metadata = {
        "metadata": {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "config_source": "progression_config.yaml",
            "use_weights": use_weights,
            "data_source_configs": ds_config
        },
        "tracking": {
            "dynamic_file": "source_level_progression_dynamic.parquet",
            "static_file": "source_level_progression_static.parquet",
            "dynamic_edge_count": len(df_dynamic),
            "static_edge_count": len(df_static)
        }
    }
    
    metadata_path = os.path.join(output_dir, "progression_metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    print(f"📄 Metadata Tracking: {metadata_path}")

    return df_dynamic, df_static

# ----------------------------------------------------
# 4. EXECUTION
# ----------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Progression Graph Pipeline")
    parser.add_argument("--dynamic-dir", required=True, help="Directory with dynamic edges")
    parser.add_argument("--static-dir", required=True, help="Directory with static edges")
    parser.add_argument("--config-file", required=True, help="Path to progression_config.yaml")
    parser.add_argument("--output-dir", required=True, help="Directory for output parquets and metadata")
    parser.add_argument("--use-weights", action="store_true", default=False, help="Whether to apply data source weights from config")

    args = parser.parse_args()

    print("\n🚀 ================ STARTING PROGRESSION PIPELINE ================")
    
    # Load config
    config = load_config(args.config_file)
    
    # Load data
    print("📂 Loading input evidence...")
    dynamic_evd = load_evidence(args.dynamic_dir, is_static=False)
    static_evd = load_evidence(args.static_dir, is_static=True)
    
    # Inspect evidence
    inspect_evidence_graph(dynamic_evd, static_evd)
    
    # Build progression
    df_dyn, df_stat = build_progression(dynamic_evd, static_evd, config, args.output_dir, use_weights=args.use_weights)
    
    # Inspect progression
    inspect_progression_graph(df_dyn, df_stat)
    
    print("✅ ================ PIPELINE COMPLETED SUCCESSFULLY ================\n")

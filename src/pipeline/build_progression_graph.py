#!/usr/bin/env python

__author__ = "Cote Falaguera (mjfalagueramata@gmail.com)"
__date__ = "02 Jul 2025"

"""
timeseries.py: Assess the evolution over time of evidence supporting target-disease associations in the Open Targets Platform.

Useful GitHub links:
- https://github.com/opentargets/timeseries
- https://github.com/opentargets/issues/issues/2739
"""


import datetime
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

from src.parsers.chembl_trial_expander import expand_chembl_clinical_trials



# ----------------------------------------
# CONFIGURATION
# ----------------------------------------
# EDGE_DIR = "/data/scratch/bty414/opentarget_evidences/23.06/kg_output/edges"
# OUT_DIR = "/data/scratch/bty414/opentarget_evidences/23.06/progression_graph"
EDGE_DIR = "/Users/pchungsiu/Documents/opentarget_het_graph/data/evidenceDated_subset/23.06/kg_output/edges"
OUT_DIR = "/Users/pchungsiu/Documents/opentarget_het_graph/data/evidenceDated_subset/23.06/kg_output/progression_graph"
DATASOURCE_HARMONIC_NOVELTY_FILE = f"{OUT_DIR}/datasource_harmonic_novelty.parquet"
DATATYPE_HARMONIC_NOVELTY_FILE = f"{OUT_DIR}/datatype_harmonic_novelty.parquet"
os.makedirs(OUT_DIR, exist_ok=True)

FIRST_YEAR = 2010
LAST_YEAR = 2025
YEARS = np.arange(FIRST_YEAR, LAST_YEAR + 1)
MAX_HARMONIC = 1.644  # theoretical max sum of 1/i^2

# novelty settings
NOVELTY_SCALE = 2     # logistic steepness
NOVELTY_SHIFT = 2     # midpoint
NOVELTY_WINDOW = 10   # years after peak to decay

data_sources = [
    {
        "id": "gwas_credible_sets",
        "sectionId": "gwasCredibleSets",
        "label": "GWAS associations",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,  # needs to be a float
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#gwas-associations",
    },
    {
        "id": "eva",
        "sectionId": "eva",
        "label": "ClinVar",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#clinvar",
    },
    {
        "id": "gene_burden",
        "sectionId": "geneBurden",
        "label": "Gene Burden",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#gene-burden",
    },
    {
        "id": "genomics_england",
        "sectionId": "genomicsEngland",
        "label": "GEL PanelApp",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#genomics-england-panelapp",
    },
    {
        "id": "gene2phenotype",
        "sectionId": "gene2Phenotype",
        "label": "Gene2phenotype",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#gene2phenotype",
    },
    {
        "id": "uniprot_literature",
        "sectionId": "uniprotLiterature",
        "label": "UniProt literature",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#uniprot-literature",
    },
    {
        "id": "uniprot_variants",
        "sectionId": "uniprotVariants",
        "label": "UniProt curated variants",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#uniprot-variants",
    },
    {
        "id": "orphanet",
        "sectionId": "orphanet",
        "label": "Orphanet",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#orphanet",
    },
    {
        "id": "clingen",
        "sectionId": "clinGen",
        "label": "Clingen",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#clingen",
    },
    {
        "id": "cancer_gene_census",
        "sectionId": "cancerGeneCensus",
        "label": "Cancer Gene Census",
        "aggregation": "Somatic mutations",
        "aggregationId": "somatic_mutation",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#cancer-gene-census",
    },
    {
        "id": "intogen",
        "sectionId": "intOgen",
        "label": "IntOGen",
        "aggregation": "Somatic mutations",
        "aggregationId": "somatic_mutation",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#intogen",
    },
    {
        "id": "eva_somatic",
        "sectionId": "evaSomatic",
        "label": "ClinVar (somatic)",
        "aggregation": "Somatic mutations",
        "aggregationId": "somatic_mutation",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#clinvar-somatic",
    },
    {
        "id": "cancer_biomarkers",
        "sectionId": "cancerBiomarkers",
        "label": "Cancer Biomarkers",
        "aggregation": "Somatic mutations",
        "aggregationId": "somatic_mutation",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#cancer-biomarkers",
    },
    {
        "id": "chembl",
        "sectionId": "chembl",
        "label": "ChEMBL",
        "aggregation": "Known drug",
        "aggregationId": "known_drug",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#chembl",
    },
    {
        "id": "crispr_screen",
        "sectionId": "crispr_screen",
        "label": "CRISPR Screens",
        "aggregation": "Affected pathway",
        "aggregationId": "affected_pathway",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#project-score",
    },
    {
        "id": "crispr",
        "sectionId": "crispr",
        "label": "Project Score",
        "aggregation": "Affected pathway",
        "aggregationId": "affected_pathway",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#project-score",
    },
    {
        "id": "slapenrich",
        "sectionId": "slapEnrich",
        "label": "SLAPenrich",
        "aggregation": "Affected pathway",
        "aggregationId": "affected_pathway",
        "weight": 0.5,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#slapenrich",
    },
    {
        "id": "progeny",
        "sectionId": "progeny",
        "label": "PROGENy",
        "aggregation": "Affected pathway",
        "aggregationId": "affected_pathway",
        "weight": 0.5,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#slapenrich",
    },
    {
        "id": "reactome",
        "sectionId": "reactome",
        "label": "Reactome",
        "aggregation": "Affected pathway",
        "aggregationId": "affected_pathway",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#reactome",
    },
    {
        "id": "sysbio",
        "sectionId": "sysBio",
        "label": "Gene signatures",
        "aggregation": "Affected pathway",
        "aggregationId": "affected_pathway",
        "weight": 0.5,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#gene-signatures",
    },
    {
        "id": "europepmc",
        "sectionId": "europePmc",
        "label": "Europe PMC",
        "aggregation": "Literature",
        "aggregationId": "literature",
        "weight": 0.2,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#europe-pmc",
    },
    {
        "id": "expression_atlas",
        "sectionId": "expression",
        "label": "Expression Atlas",
        "aggregation": "RNA expression",
        "aggregationId": "rna_expression",
        "weight": 0.2,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#expression-atlas",
    },
    {
        "id": "impc",
        "sectionId": "impc",
        "label": "IMPC",
        "aggregation": "Animal model",
        "aggregationId": "animal_model",
        "weight": 0.2,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#impc",
    },
    # {
    #     "id": "ot_crispr",
    #     "sectionId": "otCrispr",
    #     "label": "OT CRISPR",
    #     "aggregation": "Partner-only",
    #     "aggregationId": "partner_only",
    #     "weight": 0.5,
    #     "isPrivate": True,
    #     "docsLink": "https://partner-platform.opentargets.org/projects",
    # },
    # {
    #     "id": "encore",
    #     "sectionId": "encore",
    #     "label": "ENCORE",
    #     "aggregation": "Partner-only",
    #     "aggregationId": "partner_only",
    #     "weight": 0.5,
    #     "isPrivate": True,
    #     "docsLink": "https://partner-platform.opentargets.org/projects",
    # },
    # {
    #     "id": "ot_crispr_validation",
    #     "sectionId": "validationlab",
    #     "label": "OT Validation",
    #     "aggregation": "Partner-only",
    #     "aggregationId": "partner_only",
    #     "weight": 0.5,
    #     "isPrivate": True,
    #     "docsLink": "https://partner-platform.opentargets.org/projects",
    # },
]

DATA_SOURCES = {
    ds["id"]: {
        "datatype": ds["aggregationId"],
        "weight": float(ds["weight"]),
    }
    for ds in data_sources
}


# ----------------------------------------------------
# 1. LOAD ALL DYNAMIC EVIDENCE
# ----------------------------------------------------
def load_dynamic_evidence():
    dfs = []
    for fname in os.listdir(EDGE_DIR):
        if fname.startswith("sourceId=") and fname.endswith(".parquet"):
            datasource = fname.replace("sourceId=", "").replace(".parquet", "")
            df = pd.read_parquet(f"{EDGE_DIR}/{fname}")

            df["datasourceId"] = datasource
            df["year"] = df["year"].astype(int)
            
            # only expand ChEMBL clinical trials
            if datasource == "chembl":
                df = expand_chembl_clinical_trials(df)

            dfs.append(df[["sourceId", "targetId", "source_type", "target_type", "relation", "datasourceId", "score", "year"]])

    print(f"Loaded {len(dfs)} data sources")
    return pd.concat(dfs, ignore_index=True)

# ----------------------------------------------------
# 1.5. SANITY CHECK: UNIQUE NODES + UNIQUE EDGES
# ----------------------------------------------------
def inspect_graph(evd):
    print("\n================ GRAPH SUMMARY ================\n")

    # Unique nodes
    diseases = set(evd[evd["source_type"] == "disease"]["sourceId"]) | \
               set(evd[evd["target_type"] == "disease"]["targetId"])

    targets = set(evd[evd["source_type"] == "target"]["sourceId"]) | \
              set(evd[evd["target_type"] == "target"]["targetId"])

    other_nodes = set(evd["sourceId"]) | set(evd["targetId"])
    other_nodes = other_nodes - diseases - targets

    print(f"🧬 Unique disease nodes    : {len(diseases)}")
    print(f"🎯 Unique target nodes     : {len(targets)}")
    print(f"📦 Other nodes (pathway, drug, GO, etc.) : {len(other_nodes)}")
    print(f"🌐 Total unique nodes      : {len(set(evd['sourceId']) | set(evd['targetId']))}")
    print()

    # Unique edges (as tuples)
    unique_edges = set(
        tuple(row)
        for row in evd[["sourceId", "relation", "targetId"]].itertuples(index=False, name=None)
    )
    print(f"🔗 Total unique edges      : {len(unique_edges)}")

    # Show top few edges
    print("\n🔎 Sample edges:")
    print(evd[["sourceId", "relation", "targetId"]].head(10))

    # Relation statistics
    print("\n📚 Edge counts per relation:")
    print(evd["relation"].value_counts())

    # Datasource stats
    print("\n📦 Edge counts per datasource:")
    print(evd["datasourceId"].value_counts())

    # Year coverage
    print("\n📆 Year range:")
    print(f"Min year = {evd['year'].min()}, Max year = {evd['year'].max()}")


# ----------------------------------------------------
# UTILITY: harmonic sum of top-50 scores
# these are based on the implementation in https://github.com/opentargets/timeseries/blob/main/timeseries.py#L449
# ----------------------------------------------------
def harmonic_sum(scores):
    if len(scores) == 0:
        return 0.0

    s = np.sort(scores)[::-1][:50]  # top 50 descending
    idx = np.arange(1, len(s) + 1)
    return np.sum(s / (idx ** 2)) / MAX_HARMONIC

def _compute_novelty(group, score_col):
    years = group["year"].values
    scores = group[score_col].values

    diffs = np.diff(scores, prepend=0)
    peak_years = years[diffs > 0]
    peaks = diffs[diffs > 0]

    novelty_map = {}

    for py, pv in zip(peak_years, peaks):
        for t in range(py, py + NOVELTY_WINDOW + 1):
            nv = pv / (1 + np.exp(NOVELTY_SCALE * (t - py - NOVELTY_SHIFT)))
            novelty_map[t] = max(nv, novelty_map.get(t, 0))

    result = []
    for _, row in group.iterrows():
        y = row["year"]
        result.append(list(row.values) + [novelty_map.get(y, 0.0)])

    return result


# ----------------------------------------------------
# 2. DATASOURCE-LEVEL HARMONIC SCORE
# ----------------------------------------------------
def harmonic_by_datasource(evd):
    rows = []

    grouped = evd.groupby(["sourceId", "targetId", "source_type", "target_type", "relation", "datasourceId"])

    for (src, tgt, src_type, tgt_type, rel, ds), group in tqdm(grouped, desc="Datasource harmonic"):
        # group scores by year
        year_dict = group.groupby("year")["score"].apply(list).to_dict()

        collected = []
        for y in YEARS:
            if y in year_dict:
                collected.extend(year_dict[y])
            hs = harmonic_sum(collected)
            rows.append([src, tgt, src_type, tgt_type, rel, ds, y, hs])

    return pd.DataFrame(rows, columns=[
        "sourceId", "targetId", "source_type", "target_type", "relation", "datasourceId", "year", "datasource_score"
    ])


# ----------------------------------------------------
# 3. DATASOURCE-LEVEL NOVELTY
# ----------------------------------------------------
def novelty_by_datasource(df):
    rows = []
    grouped = df.groupby(["sourceId", "targetId", "source_type",
                          "target_type", "relation", "datasourceId"])

    for _, group in tqdm(grouped, desc="Datasource novelty"):
        group = group.sort_values("year")
        rows.extend(_compute_novelty(group, "datasource_score"))

    cols = df.columns.tolist() + ["novelty"]
    return pd.DataFrame(rows, columns=cols)


# ----------------------------------------------------
# 4. DATATYPE-LEVEL HARMONIC SCORE
# ----------------------------------------------------
def harmonic_by_datatype(ds_df):
    rows = []

    # attach datatype & weight
    ds_df["datatypeId"] = ds_df["datasourceId"].map(lambda x: DATA_SOURCES[x]["datatype"])
    ds_df["weight"] = ds_df["datasourceId"].map(lambda x: DATA_SOURCES[x]["weight"])
    ds_df["weighted"] = ds_df["datasource_score"] * ds_df["weight"]

    grouped = ds_df.groupby(["sourceId", "targetId", "source_type", "target_type", "relation", "datatypeId"])

    for (src, tgt, src_type, tgt_type, rel, dt), group in tqdm(grouped, desc="Datatype harmonic"):
        year_dict = group.groupby("year")["weighted"].apply(list).to_dict()
        collected = []

        for y in YEARS:
            if y in year_dict:
                collected.extend(year_dict[y])
            hs = harmonic_sum(collected)
            rows.append([src, tgt, src_type, tgt_type, rel, dt, y, hs])

    return pd.DataFrame(rows, columns=[
        "sourceId", "targetId", "source_type", "target_type", "relation", "datatypeId", "year", "datatype_score"
    ])


# ----------------------------------------------------
# 5. DATATYPE-LEVEL NOVELTY
# ----------------------------------------------------
def novelty_by_datatype(df):
    rows = []
    grouped = df.groupby(["sourceId", "targetId", "datatypeId"])

    for _, group in tqdm(grouped, desc="Datatype novelty"):
        group = group.sort_values("year")
        rows.extend(_compute_novelty(group, "datatype_score"))

    cols = df.columns.tolist() + ["novelty"]
    return pd.DataFrame(rows, columns=cols)

# ----------------------------------------------------
# 6. TEMPORAL-DEDUPLICATION
# ----------------------------------------------------
def filter_temporal_edges(df):
    """
    Keep ONLY rows where datasource_score CHANGED from the previous year.
    Removes:
      - earliest-year rows unless they start with non-zero score
      - repeated zeros
      - repeated constant values
      - any year with no change in score

    Fully vectorized.
    """

    df = df.copy()

    # Identify the evidence stream
    df["datasource_relation"] = df["datasourceId"] + "::" + df["relation"]

    # Sort so diff works correctly
    df = df.sort_values(["sourceId", "targetId", "datasource_relation", "year"])

    # Group by edge
    g = df.groupby(["sourceId", "targetId", "datasource_relation"])

    # Previous year's score
    df["score_prev"] = g["datasource_score"].shift(1)

    # Change occurs ONLY if score differs AND it's not the first row
    cond_change = (df["datasource_score"] != df["score_prev"]) & (~df["score_prev"].isna())

    # Also keep the FIRST YEAR *if* it starts with a non-zero score
    # (first appearance of evidence)
    cond_first_nonzero = df["score_prev"].isna() & (df["datasource_score"] > 0)

    # Final mask
    keep = cond_change | cond_first_nonzero

    # Filter
    filtered = df[keep].copy()

    # Clean up
    filtered = filtered.drop(columns=["score_prev"])

    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(filtered)}")
    print(f"Removed rows: {len(df) - len(filtered)}")

    return filtered





# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    print("Loading evidence...")
    evd = load_dynamic_evidence()
    
    inspect_graph(evd)

    print("Computing datasource harmonic...")
    ds_h = harmonic_by_datasource(evd)
    
    print("Computing datasource novelty...")
    ds_hn = novelty_by_datasource(ds_h)
    # collapse static groups, keep earliest year
    print("Filtering static edges (keep earliest year)...")
    ds_hn["datasource_relation"] = ds_hn["datasourceId"] + "::" + ds_hn["relation"]
    ds_hn_filtered = filter_temporal_edges(ds_hn)

    print("Saving datasource harmonic novelty (filtered)...")
    ds_hn_filtered.to_parquet(DATASOURCE_HARMONIC_NOVELTY_FILE, index=False)

    # print("Computing datatype harmonic...")
    # dt_h = harmonic_by_datatype(ds_hn)

    # print("Computing datatype novelty...")
    # dt_hn = novelty_by_datatype(dt_h)
    
    # # collapse static groups, keep earliest year
    # print("Filtering static edges (keep earliest year)...")
    # dt_hn["datasource_relation"] = dt_hn["datasourceId"] + "::" + dt_hn["relation"]
    # dt_hn_filtered = filter_temporal_edges(dt_hn)

    # print("Saving datatype harmonic novelty (filtered)...")
    # dt_hn_filtered.to_parquet(DATATYPE_HARMONIC_NOVELTY_FILE, index=False)

    # print("🎉 Completed OT temporal metrics pipeline")
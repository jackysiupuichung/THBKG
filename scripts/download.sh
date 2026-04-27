#!/bin/bash

BASE="https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.03/output"
OUTDIR="/gpfs/scratch/bty414/opentarget_evidences/26.03/evidenceDated"

# 26.03 node dir names -> local dir names expected by node_schema.yaml / static_edge_schema.yaml
declare -A NODES=(
    ["target"]="targets"
    ["disease"]="diseases"
    ["drug_molecule"]="molecule"
    ["reactome"]="reactome"
    ["go"]="go"
    ["interaction_evidence"]="interactionEvidence"
)

mkdir -p "$OUTDIR"

# === OT 26.03 node parquet downloads ===
echo "Starting OpenTargets 26.03 node download..."

for remote_name in "${!NODES[@]}"; do
    local_name="${NODES[$remote_name]}"
    NODE_URL="$BASE/$remote_name"
    NODE_OUT="$OUTDIR/$local_name"
    mkdir -p "$NODE_OUT"

    echo ""
    echo "============================"
    echo "Downloading: $remote_name -> $local_name"
    echo "URL: $NODE_URL"
    echo "============================"

    parquet_files=$(wget -qO- "$NODE_URL/" | grep -o '[^"]*\.parquet')

    if [ -z "$parquet_files" ]; then
        echo "No parquet files found for $remote_name"
        continue
    fi

    for file in $parquet_files; do
        echo "$file"
        wget -nc -P "$NODE_OUT" "$NODE_URL/$file"
    done

    echo "Done $remote_name -> $local_name"
done

echo ""
echo "Completed OT node downloads -> $OUTDIR"

# === Gene Ontology OBO ===
echo ""
echo "============================"
echo "Downloading Gene Ontology go-basic.obo..."
echo "============================"

GO_OBO_URL="http://purl.obolibrary.org/obo/go/go-basic.obo"
GO_OUT="$OUTDIR/go_ontology"
mkdir -p "$GO_OUT"
wget -nc -O "$GO_OUT/go-basic.obo" "$GO_OBO_URL"
echo "Done: $GO_OUT/go-basic.obo"

# === IntAct human PPI ===
echo ""
echo "============================"
echo "Downloading IntAct human PPI..."
echo "============================"

INTACT_URL="https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/species/human.zip"
INTACT_OUT="$OUTDIR/intact"
mkdir -p "$INTACT_OUT"
wget -nc -O "$INTACT_OUT/human.zip" "$INTACT_URL"
echo "Unzipping..."
unzip -o "$INTACT_OUT/human.zip" -d "$INTACT_OUT"
rm "$INTACT_OUT/human.zip"
echo "Done: $INTACT_OUT/"

# === OT 26.03 evidence downloads ===
# FTP uses evidence_{source} naming; downloaded into sourceId={source} dirs to match edge_schema.yaml
#
# Removed upstream (not on 26.03 FTP, commented out of edge_schema.yaml):
#   - slapenrich   -> superseded by other sources
#   - sysbio       -> superseded by other sources (Gene Signatures)
#   - chembl       -> replaced by evidence_clinical_precedence
#
# All evidence_ sources on 26.03 FTP are accounted for below.
echo ""
echo "Starting OpenTargets 26.03 evidence download..."

declare -A EVIDENCES=(
    ["evidence_cancer_biomarkers"]="sourceId=cancer_biomarkers"
    ["evidence_cancer_gene_census"]="sourceId=cancer_gene_census"
    ["evidence_clinical_precedence"]="sourceId=clinical_precedence"
    ["evidence_clingen"]="sourceId=clingen"
    ["evidence_crispr"]="sourceId=crispr"
    ["evidence_crispr_screen"]="sourceId=crispr_screen"
    ["evidence_europepmc"]="sourceId=europepmc"
    ["evidence_eva"]="sourceId=eva"
    ["evidence_eva_somatic"]="sourceId=eva_somatic"
    ["evidence_expression_atlas"]="sourceId=expression_atlas"
    ["evidence_gene2phenotype"]="sourceId=gene2phenotype"
    ["evidence_gene_burden"]="sourceId=gene_burden"
    ["evidence_genomics_england"]="sourceId=genomics_england"
    ["evidence_impc"]="sourceId=impc"
    ["evidence_orphanet"]="sourceId=orphanet"
    ["evidence_reactome"]="sourceId=reactome"
    ["evidence_uniprot_literature"]="sourceId=uniprot_literature"
    ["evidence_uniprot_variants"]="sourceId=uniprot_variants"
    ["evidence_gwas_credible_sets"]="sourceId=gwas_credible_sets"
    ["evidence_intogen"]="sourceId=intogen"
)

for remote_name in "${!EVIDENCES[@]}"; do
    local_name="${EVIDENCES[$remote_name]}"
    EV_URL="$BASE/$remote_name"
    EV_OUT="$OUTDIR/$local_name"
    mkdir -p "$EV_OUT"

    echo ""
    echo "============================"
    echo "Downloading: $remote_name -> $local_name"
    echo "URL: $EV_URL"
    echo "============================"

    parquet_files=$(wget -qO- "$EV_URL/" | grep -o '[^"]*\.parquet')

    if [ -z "$parquet_files" ]; then
        echo "No parquet files found for $remote_name"
        continue
    fi

    for file in $parquet_files; do
        echo "$file"
        wget -nc -P "$EV_OUT" "$EV_URL/$file"
    done

    echo "Done $remote_name -> $local_name"
done

echo ""
echo "All downloads complete."

#!/bin/bash

OT_VERSION="${OT_VERSION:-26.09}"
OT_ROOT="https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/${OT_VERSION}"
BASE="${OT_ROOT}/output"
OUTDIR="/gpfs/scratch/bty414/opentarget_evidences/${OT_VERSION}/evidenceDated"

FEATURE_RAW_DIR="/gpfs/scratch/bty414/opentarget_evidences/${OT_VERSION}/features/raw"

# Pinned external releases. GO comes from OT's own input copy so the GO
# hierarchy matches the OT `go` nodes; IntAct and HPA are pinned to a release
# instead of `current`. All are recorded in $OUTDIR/source_versions.json.
INTACT_RELEASE="2026-01-09"
HPA_VERSION="25"

# Download $1 to $2 unless $2 already exists and is non-empty. Writes to a
# .tmp file first so a failed transfer never leaves a file that a rerun skips.
fetch() {
    local url="$1" dest="$2"
    if [ -s "$dest" ]; then
        echo "Already present: $dest"
        return 0
    fi
    wget -O "$dest.tmp" "$url" && mv "$dest.tmp" "$dest" || {
        rm -f "$dest.tmp"
        echo "ERROR: download failed: $url" >&2
        exit 1
    }
}

# OT node dir names -> local dir names expected by node_schema.yaml / static_edge_schema.yaml
declare -A NODES=(
    ["target"]="targets"
    ["disease"]="diseases"
    ["drug_molecule"]="molecule"
    ["reactome"]="reactome"
    ["go"]="go"
    ["interaction_evidence"]="interactionEvidence"
)

mkdir -p "$OUTDIR"

# === OT node parquet downloads ===
echo "Starting OpenTargets ${OT_VERSION} node download..."

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

    parquet_files=$(wget -qO- "$NODE_URL/" | grep -o 'href="[^"]*\.parquet"' | sed 's/^href="//; s/"$//')

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
# OT's input copy of go.obo (full, not go-basic). The GO parser keeps only
# is_a edges and drops obsolete terms, which are identical in both files.
echo ""
echo "============================"
echo "Downloading Gene Ontology go.obo (OT ${OT_VERSION} input copy)..."
echo "============================"

GO_OBO_URL="${OT_ROOT}/input/go/go.obo"
GO_OUT="$OUTDIR/go_ontology"
mkdir -p "$GO_OUT"
fetch "$GO_OBO_URL" "$GO_OUT/go.obo"
GO_DATA_VERSION=$(grep -m1 '^data-version:' "$GO_OUT/go.obo" | sed 's/^data-version: *//')
if [ -z "$GO_DATA_VERSION" ]; then
    echo "ERROR: no data-version header in $GO_OUT/go.obo" >&2
    exit 1
fi
echo "Done: $GO_OUT/go.obo ($GO_DATA_VERSION)"

# === IntAct human PPI ===
echo ""
echo "============================"
echo "Downloading IntAct human PPI (release ${INTACT_RELEASE})..."
echo "============================"

INTACT_URL="https://ftp.ebi.ac.uk/pub/databases/intact/${INTACT_RELEASE}/psimitab/species/human.zip"
INTACT_OUT="$OUTDIR/intact"
mkdir -p "$INTACT_OUT"
# The zip also holds human_negative.txt and viral subsets; IntActParser reads
# whichever .txt it lists first, so extract only the human PPI file.
if [ -s "$INTACT_OUT/human.txt" ]; then
    echo "Already present: $INTACT_OUT/human.txt"
else
    fetch "$INTACT_URL" "$INTACT_OUT/human.zip"
    echo "Unzipping human.txt..."
    unzip -o "$INTACT_OUT/human.zip" human.txt -d "$INTACT_OUT" || {
        echo "ERROR: unzip failed: $INTACT_OUT/human.zip" >&2
        exit 1
    }
    rm "$INTACT_OUT/human.zip"
fi
echo "Done: $INTACT_OUT/human.txt"

# === Target node feature inputs ===
# Layout expected by preprocessing/node_features/target_features.py under
# --feature-data-dir: target_prioritisation/part-*.parquet and
# rna_expression/rna_single_cell_type_group.tsv.zip.
echo ""
echo "============================"
echo "Downloading target feature inputs -> $FEATURE_RAW_DIR"
echo "============================"

PRIO_URL="$BASE/target_prioritisation"
PRIO_OUT="$FEATURE_RAW_DIR/target_prioritisation"
mkdir -p "$PRIO_OUT"
for file in $(wget -qO- "$PRIO_URL/" | grep -o 'href="[^"]*\.parquet"' | sed 's/^href="//; s/"$//' | sort -u); do
    wget -nc -P "$PRIO_OUT" "$PRIO_URL/$file"
done
echo "Done: $PRIO_OUT/"

# HPA vNN hosts serve that release's files; the current one redirects to www.
HPA_URL="https://v${HPA_VERSION}.proteinatlas.org/download/tsv/rna_single_cell_type_group.tsv.zip"
HPA_OUT="$FEATURE_RAW_DIR/rna_expression"
mkdir -p "$HPA_OUT"
fetch "$HPA_URL" "$HPA_OUT/rna_single_cell_type_group.tsv.zip"
HPA_LAST_MODIFIED=$(wget -S --spider "$HPA_URL" 2>&1 | grep -i 'last-modified' | tail -1 | sed 's/^ *[Ll]ast-[Mm]odified: *//')
echo "Done: $HPA_OUT/rna_single_cell_type_group.tsv.zip"

# === OT evidence downloads ===
# FTP uses evidence_{source} naming; downloaded into sourceId={source} dirs to match edge_schema.yaml
#
# Removed upstream (absent from the 26.03 and 26.09 FTP, commented out of edge_schema.yaml):
#   - slapenrich   -> superseded by other sources
#   - sysbio       -> superseded by other sources (Gene Signatures)
#   - chembl       -> replaced by evidence_clinical_precedence
#
# All evidence_ sources on the 26.03 and 26.09 FTP are accounted for below.
echo ""
echo "Starting OpenTargets ${OT_VERSION} evidence download..."

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

    parquet_files=$(wget -qO- "$EV_URL/" | grep -o 'href="[^"]*\.parquet"' | sed 's/^href="//; s/"$//')

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

# === Source versions ===
OT_INTEGRITY_SHA1=$(wget -qO- "${OT_ROOT}/release_data_integrity.sha1" | awk '{print $1}')
cat > "$OUTDIR/source_versions.json" <<EOF
{
  "downloaded_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "open_targets": {
    "release": "${OT_VERSION}",
    "url": "${BASE}",
    "release_data_integrity_sha1": "${OT_INTEGRITY_SHA1}"
  },
  "gene_ontology": {
    "data_version": "${GO_DATA_VERSION}",
    "url": "${GO_OBO_URL}"
  },
  "intact": {
    "release": "${INTACT_RELEASE}",
    "url": "${INTACT_URL}"
  },
  "target_prioritisation": {
    "release": "${OT_VERSION}",
    "url": "${PRIO_URL}"
  },
  "human_protein_atlas": {
    "version": "${HPA_VERSION}",
    "url": "${HPA_URL}",
    "last_modified": "${HPA_LAST_MODIFIED}"
  }
}
EOF
echo "Wrote $OUTDIR/source_versions.json"

echo ""
echo "All downloads complete."

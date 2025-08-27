#!/bin/bash
set -euo pipefail

source .venv/bin/activate

CHEMBL_PARQUET="./data/kg_output/edges/sourceId=chembl.parquet"
CUTOFF=2010
HORIZON=5
# TODO: this should go to run dir
OUT_DIR="./data/"
COLD_START_FILE="./data/cold_start_targets.txt"


echo "🚀 Splitting data with temporal + cold-start..."
# python -m src.data.split \
#   --parquet "$CHEMBL_PARQUET" \
#   --cutoff "$CUTOFF" \
#   --horizon "$HORIZON" \
#   --cold-start-targets "$COLD_START_FILE" \
#   --out-dir "$OUT_DIR"

python -m src.data.split \
  --parquet "$CHEMBL_PARQUET" \
  --cutoff "$CUTOFF" \
  --horizon "$HORIZON" \
  --out-dir "$OUT_DIR"


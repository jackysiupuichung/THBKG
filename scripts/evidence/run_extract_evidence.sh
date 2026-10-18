#!/bin/bash
#SBATCH -J extract_evidence
#SBATCH -o %x.o%j
#SBATCH -p computeshort
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 1:0:0
#
# Extract the Open Targets evidence behind THBKG paths or subgraphs.
# Single process: one core. Arguments are passed to src.evidence.extract.
#
#   sbatch scripts/evidence/run_extract_evidence.sh \
#       --path "ENSG00000120217 -- MONDO_0002169" --as-of 2018 --raw key \
#       --out-dir /path/to/out
#
#   # explanations from an older graph: give the mappings that run used
#   sbatch scripts/evidence/run_extract_evidence.sh --release 26.03 \
#       --pagelink-paths <run>/per_pair_paths.parquet \
#       --mappings <root>/26.03/progression/temporal_graph_datatype_mappings.PRE2026.pt \
#       --out-dir /path/to/out
#
# A 26.09 --raw key run over europepmc-backed edges peaked at ~4.5 GB.

set -euo pipefail

VENV="${VENV:-.venv}"
source "$VENV/bin/activate"
export THBKG_DATA_ROOT="${THBKG_DATA_ROOT:-/gpfs/scratch/bty414/opentarget_evidences}"

python -m src.evidence.extract "$@"

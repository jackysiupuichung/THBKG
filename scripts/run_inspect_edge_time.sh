#!/bin/bash
#SBATCH --job-name=inspect_etime
#SBATCH --partition=computeshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:20:0
#SBATCH --output=%x.o%j

set -euo pipefail
cd /data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph
uv run python scripts/inspect_edge_time_provenance.py

[![Dataset: Zenodo](https://img.shields.io/badge/dataset-Zenodo%2010.5281%2Fzenodo.20795231-1682D4)](https://doi.org/10.5281/zenodo.20795231)
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/jackysiupuichung/THBKG)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-CUDA%2011.8-EE4C2C)](https://pyg.org/)

# THBKG — Temporal Heterogeneous Biomedical Knowledge Graph

A dated biomedical knowledge graph built from
[Open Targets](https://www.opentargets.org/) 26.09 (with Gene Ontology, Reactome,
IntAct, the Human Protein Atlas and ClinicalTrials.gov), plus the code for its
**clinical-advancement benchmark**: ranking target–disease pairs that entered
Phase II by their likelihood of advancing to Phase III, scored only from
evidence datable *strictly before* each pair's decision year.

Every temporal edge carries the year its evidence first appeared, so the graph
can be queried as of any historical decision point without leakage. Full dataset
description and statistics are in [croissant.json](croissant.json).

## Load the dataset

The 26.09 graph and advancement labels are on
[Hugging Face](https://huggingface.co/datasets/jackysiupuichung/THBKG)
(CC-BY-4.0) in a pickle-free format, with a standalone loader:

```python
from huggingface_hub import snapshot_download
from load_thbkg import load_graph, load_labels   # shipped with the dataset

path  = snapshot_download("jackysiupuichung/THBKG", repo_type="dataset")
graph = load_graph(path)            # PyG HeteroData (training view)
train = load_labels(path, "train")  # target_id, disease_id, transition_year, outcome
```

The previous 26.03 build is the Hugging Face revision `v26.03`; its PyG graph
tensors are also attached to Zenodo version
[10.5281/zenodo.21529524](https://doi.org/10.5281/zenodo.21529524). Cite the
concept DOI [10.5281/zenodo.20795231](https://doi.org/10.5281/zenodo.20795231)
(all versions).

To run the CLI against an unpacked copy, point it at the data root:

```bash
export THBKG_DATA_ROOT=/path/to/opentarget_evidences
```

Paths resolve under `$THBKG_DATA_ROOT/<release>/...` (e.g. `26.09/graph/`);
individual files can be overridden per call (`--graph_file`, `--mappings_file`).

## Setup

Dependencies are managed with [uv](https://github.com/astral-sh/uv); Python ≥
3.11, PyTorch + PyG on CUDA 11.8. Training and evaluation are heavy GPU/CPU jobs,
run via SLURM (`sbatch`), not in the foreground.

```bash
uv sync
```

## Evaluate

```bash
python -m src.evaluate_advancement                              # all registered runs
python -m src.evaluate_advancement --only p3_eahgt_both,b1_hgt  # a subset
```

The evaluation CLI's registered runs are the published 26.03 results; 26.09
training configs are in `config/experiments/w3_2609/`.

The primary metric is **Relative Success @ K** (an importance-weighted hit rate),
reported per therapeutic area (TA-mean over 13 areas) and Wilcoxon-tested against
a randomized-decisions baseline.

| Reference | What it is | Source |
|---|---|---|
| **EA-HGT** | Grouped five-seed, validation-selected, percentile-rank-fused ensemble (official result) | this repo |
| **RDG** | Ridge regression on decision-time features (Czech et al.) | `evaluation_dataset.zarr` |
| **OTS** | Open Targets global association score | `evaluation_dataset.zarr` |

## Explainability

Post-hoc explanation code lives under `explain/`: the library in `src/explain/`
(integrated gradients, attention, PaGE-Link), command-line entrypoints in
`explain/cli/`, evidence-assembly helpers in `explain/evidence/`, case-study
tooling in `explain/casestudy/`, and SLURM drivers in `explain/drivers/`.

## Rebuild from source

Ordered SLURM stages under `scripts/`. `OT_VERSION` selects the Open Targets
release (default `26.09`); every upstream source is pinned, and the versions
used are written to `evidenceDated/source_versions.json`.

```bash
bash   scripts/download.sh                      # OT release + pinned GO, IntAct, HPA
sbatch scripts/build_nct_terminal_dates.sh      # trial completion dates (ClinicalTrials.gov)
sbatch scripts/collecting_edges_01.sh           # trial outcomes, KG edges, dated event lists
sbatch scripts/building_event_graph_02.sh       # HeteroData temporal edges (+ advancement nodes)
sbatch scripts/collecting_node_features_03.sh   # node features (GPU)
sbatch scripts/assembling_graph_04.sh           # attach features → final graph
```

The advancement labels for each release are committed under
`data/clinical_trial_advancement/<release>_w3/`.

Then train the ensemble (per-seed configs in `config/experiments/headline/`) and
fuse:

```bash
sbatch scripts/advancement_prediction/run_grouped_ensemble_strictmask.sh
```

## License

Source code: [MIT](LICENSE). Dataset artifacts: CC-BY-4.0 (see Zenodo above).

> The current build is Open Targets 26.09. The 26.03 build behind the
> published results is archived as the Hugging Face revision `v26.03` and on
> Zenodo; rerunning the pipeline with `OT_VERSION=26.03` does not reproduce it
> exactly, because 26.09 dates trial outcomes by actual completion and
> re-queries ClinicalTrials.gov. Results are not directly comparable across
> builds, because disease IDs and the clinical data also changed upstream. The
> earlier 23.06 build had a same-year clinical-trial-edge leak and is
> **deprecated**.

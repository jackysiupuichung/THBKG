---
license: cc-by-4.0
pretty_name: THBKG — Temporal Heterogeneous Biomedical Knowledge Graph
task_categories:
- graph-ml
- tabular-classification
- other
tags:
- biomedical
- knowledge-graph
- temporal
- link-prediction
- drug-discovery
- target-identification
- hypothesis-generation
- clinical-trials
- leak-free-evaluation
size_categories:
- 1M<n<10M
configs:
- config_name: discovery
  data_files:
  - split: pairs
    path: labels/discovery.parquet
- config_name: hypothesis
  data_files:
  - split: pairs
    path: labels/hypothesis.parquet
- config_name: advancement
  data_files:
  - split: train
    path: labels/train.parquet
  - split: eval
    path: labels/eval.parquet
---

# THBKG — Temporal Heterogeneous Biomedical Knowledge Graph

**THBKG v2**: a dated biomedical knowledge graph built from
[Open Targets](https://www.opentargets.org/) **26.09**, with Gene Ontology,
Reactome, IntAct, the Human Protein Atlas, ChEMBL and ClinicalTrials.gov. Every
evidence edge carries the year it was first observed, so the graph can be read
as of any past decision year. Three benchmark tasks follow a target–disease pair
from its first published link, to its first clinical trial, to Phase III:

| Task | A model ranks… | Positive |
|---|---|---|
| **T1 discovery** | every target with no trial for the disease yet | the pair's first clinical trial starts in year *Y* |
| **T2 hypothesis** | every target with no direct link of any kind to the disease yet | the pair's first direct link is a literature co-mention in year *Y* |
| **T3 advancement** | target–disease pairs entering Phase II in year *Y* | the pair reaches Phase III by a fixed horizon |

In every task a model scores year *Y* using **only evidence dated strictly before
*Y***. All labels are read off the graph itself, so the year a label becomes
true is exactly the year the time rule hides.

## v2: what changed since v26.09

v2 (release format 2.0) replaces the v26.09 build of the same Open Targets
release (format 1.0). Code written for v26.09 does not run on it unchanged.

- **Edge attributes are `[score]`.** The `novelty` column is gone. `metadata.json`
  gives each edge type a `score_max`, a constant of the scoring formula, for
  normalising scores without fitting anything to the data.
- **Molecules cover every drug type** (3,717: small molecules, antibodies,
  proteins, oligonucleotides, …) and their features are stored per modality,
  never concatenated: `x_morgan` (1024-bit Morgan fingerprint), `x_esm` (1280-d
  ESM-2 650M over the ChEMBL protein chains), `has_morgan`, `has_esm` and
  `drug_type`.
- **Disease features are 768-d** PubMedBERT embeddings of each disease's name and
  description (256-d before).
- **A snapshot view** (`snapshots/`): each pair's latest state before each
  decision year 1995–2027, with a recency feature.
- **New tasks T1 and T2**, and T3 labels rebuilt from the graph's own trial edges:
  train entries 1995–2019 labelled by 2022, test entries 2020–2023 labelled by
  2026 (v26.09: train to 2015 by 2018, eval 2016–2021 by 2024). There is no
  official validation split; models are selected on fixed time-blocked folds
  inside the train years.
- **Build fixes.** The start-dated trial edge carries the trial's phase weight
  rather than Open Targets' score, which is halved for trials that later stopped
  for safety or efficacy (an outcome leaking into the start year). Gene Ontology
  and IntAct edges are no longer lost to a failed PubMed date lookup. Literature
  is dated by publication date. Zero-score evidence and trials with no start date
  are dropped. Trials of unknown status get no outcome edge.

## Files

| Path | What it is |
|---|---|
| `graph.safetensors` | The continuous graph: node features, `edge_index`, `edge_attr`, `edge_time` (585 MB, pickle-free). |
| `metadata.json` | Schema to reassemble it: node and edge types, tensor keys, shapes, dtypes, `score_max` per edge type. |
| `mappings/<type>.parquet` | `index ↔ node_id` per node type. |
| `snapshots/<Y>.safetensors`, `snapshots/index.json` | The snapshot view: one file per decision year, keyed by `metadata.json` edge-type order (about 2 GB in all). |
| `labels/discovery.parquet` | T1: one row per target–disease pair with a trial. |
| `labels/hypothesis.parquet` | T2: one row per directly linked target–disease pair. |
| `labels/train.parquet`, `labels/eval.parquet` | T3: pairs entering Phase II, with their outcome. |
| `labels/therapeutic_areas.parquet`, `labels/primary_therapeutic_areas.json` | Disease → Open Targets therapeutic areas, and the areas T3 is reported on. |
| `scores/target_disease_by_source.parquet` | Per-source target–disease scores by year, for baselines that need the sources apart. |
| `features/` | Raw 1280-d ESM-2 embeddings of the biologics, target-feature imputation flags, how disease embeddings were made. |
| `benchmark_tasks.json` | Splits, slate filters, label and slate counts per task and split, closed-form random floors. |
| `check_release.json`, `check_inputs.json` | The release checks this build passed (see *Checks*). |
| `molecule_features.yaml`, `source_versions.json` | Molecule feature recipe; pinned upstream versions. |
| `load_thbkg.py` | Standalone loader: PyG `HeteroData` for either view, label tables. |
| `croissant.json` | MLCommons Croissant metadata, with the sha256 of every file. |

## The graph

| Node type | Nodes | Features |
|---|---|---|
| `target` | 19,632 | 56-d: Open Targets target-prioritisation features (tissue distribution, genetic constraint, mouse orthologue identity) and Human Protein Atlas single-cell-type RNA expression |
| `disease` | 45,896 | 768-d PubMedBERT embedding of name and description |
| `molecule` | 3,717 | per modality, see above |
| `go` | 38,092 | 64-d graph-autoencoder embedding of the GO hierarchy |
| `reactome` | 2,883 | 64-d graph-autoencoder embedding of the Reactome hierarchy |

**Edges:** 14,104,155 rows over 19 relations. Sixteen are dated evidence:
literature, genetic association, somatic mutation, animal model, RNA expression,
affected pathway, disease–pathway association, protein interaction, GO
annotation, pathway membership, drug target (`modulated_by`) and five
clinical-trial relations. Three are static
ontology edges (disease, GO and Reactome hierarchies). Protein–protein
interaction (`interacts_with`) is symmetric.

**Two views of the same evidence.**

- *Continuous* (`load_graph`): one row per pair, relation and year in which the
  pair's score changed. The score is Open Targets' own aggregate of the evidence
  dated up to that year (a weighted harmonic sum; trials take their highest phase
  weight). Evidence dated 1985 or earlier is collapsed onto 1985.
- *Snapshot* (`load_snapshot(Y)`): each pair's latest row dated before *Y*, with
  `edge_attr = [score, recency]`. Recency is a decaying trace of the pair's
  recent score increases, 0 for static edges.

**Trial edges are dated by event.** A registered trial adds a
`clinical_trial_ongoing` edge at its start date. When it finishes it adds an
outcome edge (`positive`, `unmet_efficacy`, `adverse_effects` or
`Unknown/Operational`) at its completion date from ClinicalTrials.gov.

## Benchmark tasks

Decision years are split the same way for T1 and T2: **train 1996–2019, test
2020–2025**. T3 splits its pairs by Phase II entry year instead (train 1995–2019,
test 2020–2023), and its test entries end at 2023 so every pair has at least
three years to advance. Training on a split means using its labels and the graph
before each decision year.

There is no official validation split. Select models on the time-blocked folds
inside the train years (`cv_folds` in `benchmark_tasks.json`, the same for every
method): validation blocks 2008–2011, 2012–2015 and 2016–2019, each fold
training on all earlier train years. T3 folds are labelled like the official
split, train pairs by the year before the block plus three, validation pairs by
the block's last year plus three. Never train or select on a test year.

### T1 discovery: which targets will enter the clinic for this disease?

One slate per (disease, year *Y*) with at least one positive. It ranks every
target with no trial for that disease before *Y* (about 19,600 candidates).
Positives are the targets whose first trial for the disease, at any phase,
starts in *Y*. Test positives need a first trial at Phase 1 or later; pairs whose
first trials are early Phase 1 or of unknown phase are left out of the test
slates. Test slates are kept for the 728 diseases that gain more than three new
targets over the test period.

| Split | Slates | Diseases | Positives |
|---|---|---|---|
| train | 11,517 | 2,561 | 77,375 |
| test | 1,908 | 728 | 13,534 |

For NDCG, positives are graded by the highest phase their finished trials
reached by the end of the graph (`grade`). The grade looks past *Y*, so use it
only to score rankings.

### T2 hypothesis: which targets will be linked to this disease next?

One slate per (disease, year *Y*). It ranks every target with **no direct edge of
any relation to the disease before *Y***, trials included. Positives are the
targets whose first direct link to the disease is a Europe PMC literature
co-mention in *Y*. Pairs first linked by other evidence are left out of that
year's slate, and a linked pair leaves the pool, so a repeat co-mention never
counts. There is no quality filter on the co-mention.

| Split | Slates | Diseases | Positives |
|---|---|---|---|
| train | 93,214 | 8,997 | 1,177,962 |
| test | 35,628 | 9,108 | 924,845 |

Results are also reported per stratum of the positives:
- co-mentioned again in a later year;
- later linked by non-literature evidence;
- later trialled;
- by how many diseases the target was linked to before *Y* (0–9, 10–99, 100–999, 1000+).

The first three look past *Y* and have less time to happen in later test years.
Compare them within a split.

### T3 advancement: which Phase II programmes will reach Phase III?

A pair enters Phase II in the first year a Phase II trial starts while it has no
Phase III+ trial. It is positive if a Phase III+ trial starts by the horizon of
its label file: 2022 for train, 2026 for test. Negative means "not at Phase III
by the horizon", not failed. Each pair is scored at its entry year.

| Split | Phase II entry | Outcome by | Pairs | Positive |
|---|---|---|---|---|
| train | 1995–2019 | 2022 | 43,468 | 18.0% |
| test | 2020–2023 | 2026 | 7,786 | 4.8% |

**Train outcomes reach past the first test decisions.** Train and test pairs are
disjoint, but 1,077 train positives started Phase III in 2020–2022. 50.6% of test
pairs share a target with such a pair whose Phase III came at or after the test
pair's decision year (17.9% share a disease), and those test pairs advance more
often (6.4% vs 3.2%). A model that learns which targets advance can use outcomes
a decision in that year could not have known.

## Metrics and baselines

**T1 and T2.** The primary metric is **precision@R**: the precision at top *K* of
[TargetBench](https://www.nature.com/articles/s41598-026-47765-3), with *K* set
to the number of positives, i.e. the share of the top *R* candidates that are
positives. A disease's slates are pooled across the split's
years, with one vote per disease (`precision@R_per_disease`). Precision, recall
and NDCG at *K* = 100, 200 and 500, macro-averaged over slates, are secondary.
Ties are broken against the positives, so a constant score cannot beat random.

**T3.** Relative Success @ N (the advancement rate of a model's top *N* pairs over
the rate of the rest) per primary therapeutic area (13 areas, plus "all") and
their mean, plus AUROC and AP, under two rankings:
- **raw**: scores ranked across all test entry years;
- **year-stratified**: each score replaced by its mid-rank percentile among the
  pairs entering Phase II in the same year, then the years pooled, as in the
  THBKG rebuttal (mid-rank, so a constant score stays at RS 1).

The label is Phase III by a fixed horizon, so the positive rate falls with entry
year as follow-up shortens (5.8% of 2020 entries, 3.7% of 2023 entries). A raw
ranking mixes that with real signal: on Open Targets 26.03 the year-stratified
ranking took the Open Targets score to chance (AUROC 0.556 to 0.500), and on this
test it moves the score's TA-mean RS@10 from 2.22 to 3.17 and its pooled RS@50
from 7.31 to 4.65. We recommend comparing models on the year-stratified ranking:
it compares each pair only with the programmes that entered Phase II in its
year. The raw ranking is kept for comparison with Czech et al. and earlier
THBKG results.

Baselines that need no training, on test:

| Task | Method | Primary metric |
|---|---|---|
| T1 | random | 0.0006 |
| T1 | Open Targets direct association score | 0.0083 |
| T1 | genetics-only association score | 0.0030 |
| T2 | random | 0.0011 |
| T2 | target popularity (diseases linked before *Y*) | 0.0112 |
| T3 | random | RS@10 = 1.00 (both rankings) |
| T3 | Open Targets direct association score | RS@10 = 3.17 year-stratified (2.22 raw), AUROC 0.65 (0.61) |
| T3 | genetics-only association score | RS@10 = 2.16 year-stratified (1.90 raw), AUROC 0.55 (0.50) |

T1 and T2 use `precision@R_per_disease`; T3 uses the therapeutic-area mean of
RS@10 and of AUROC. Open Targets scores are 0 for every T2 candidate (none has a
direct edge yet), so T2's reference is target popularity. Nearly all of its hits
are targets already linked to 1,000 or more diseases.

The slate builder, the metrics and these baselines are in the
[GitHub repository](https://github.com/jackysiupuichung/THBKG)
(`src/benchmark/`, `scripts/benchmark/run_cpu_baselines.py`).

## Quick start

```python
from huggingface_hub import snapshot_download
from load_thbkg import load_graph, load_snapshot, load_labels, node_index

path = snapshot_download("jspc00012567/THBKG", repo_type="dataset")
data = load_graph(path)                  # continuous view, PyG HeteroData, reverse edges added
snap = load_snapshot(path, 2020, normalise="max")   # evidence dated before 2020, scores in [0, 1]

t1 = load_labels(path, "discovery")      # T1 pairs: first_trial_year, grade, ...
t2 = load_labels(path, "hypothesis")     # T2 pairs: first_direct_year, novel, strata columns
train, eval_ = load_labels(path, "train"), load_labels(path, "eval")   # T3
t2i = node_index(path, "target")         # node_id -> index
```

Scoring a T1 or T2 split with the repository's harness, one decision year at a
time (a model can embed each year's snapshot once):

```python
from src.benchmark.tasks import evaluate_task

def score_fn_for_year(year, slates):
    z = my_model(load_snapshot(path, year))            # evidence before `year` only
    return lambda s: my_scores(z, s.disease_index, s.candidates)

summary, per_slate = evaluate_task("hypothesis", "test", t2, num_targets=19_632,
                                   score_fn_for_year=score_fn_for_year)
print(summary["precision@R_per_disease"])
```

The label tables also load as 🤗 `datasets` configs: `discovery`, `hypothesis`
and `advancement` (splits `train`, `eval`).

## Checks

Every build runs two checks on the assembled release, and a failure stops the
build. The results ship with it.

- `check_release.json`: the release is well formed:
  - files, tensors and dtypes match `metadata.json`;
  - every edge type's times and scores lie in range;
  - the shipped loader and the snapshots agree with the graph;
  - every label file equals the table rebuilt from this graph;
  - `croissant.json` lists every file with its current sha256.
- `check_inputs.json`: the graph holds exactly its own Open Targets input under
  the build's rules. Every target–disease pair the input supports is present, no
  pair the input lacks is present, and each pair's first year matches the input.

## Limitations

- **Dates are evidence dates, not curation dates.** A snapshot for year *Y* shows
  evidence published before *Y* as Open Targets 26.09 curates it today. Evidence
  that was only added to the databases later is still visible at *Y*.
- **Precision is a lower bound.** A pair first trialled or co-mentioned after *Y*
  counts as a negative at *Y*.
- **Literature co-mentions are text-mined.** They cannot tell a passing mention
  from a reported association. About half of the T2 positives of 2016–2019 are
  never co-mentioned again.
- **T3 train outcomes overlap the first test years.** Train labels run to 2022,
  so outcomes of targets shared with test pairs can be learned before those test
  decisions (see T3). Fixed-horizon labels also give older entries more
  follow-up; compare models on the year-stratified ranking.
- **Popularity is a strong prior.** New links favour well-studied targets; report
  results by target-degree stratum.
- **Builds are not comparable.** Open Targets replaced many EFO disease IDs with
  MONDO IDs in 26.06, merged duplicate terms in 26.09, and its clinical records
  change between releases.

## Versions

This is **v2** (revision `v2`, release format 2.0), built on Open Targets 26.09.
Tags now version THBKG itself; the Open Targets release it is built on is in
`source_versions.json`. Earlier builds stay available by revision:

```python
snapshot_download("jspc00012567/THBKG", repo_type="dataset", revision="v2")      # this build
snapshot_download("jspc00012567/THBKG", repo_type="dataset", revision="v26.09")  # format 1.0
snapshot_download("jspc00012567/THBKG", repo_type="dataset", revision="v26.03")
```

An earlier 23.06 build had a same-year clinical-trial-edge leak and is
**deprecated**.

## License & citation

Dataset artifacts: **CC-BY-4.0**. Source code (construction pipeline, benchmark
harness): MIT, at <https://github.com/jackysiupuichung/THBKG>. Archival DOI on
Zenodo: concept [10.5281/zenodo.20795231](https://doi.org/10.5281/zenodo.20795231).

### Sources

THBKG is derived from the resources below. If you use the parts that come from
them, please cite them as well.

| Source | Licence | In THBKG |
|---|---|---|
| [Open Targets Platform](https://platform-docs.opentargets.org/licence) 26.09 | CC0 1.0 | evidence edges and per-source scores, disease index, molecules and SMILES, drug–target and clinical-trial edges, target-prioritisation features (Buniello et al., *Nucleic Acids Res.* 2025) |
| [Gene Ontology](https://geneontology.org/docs/go-citation-policy/), release 2026-07-26 | CC BY 4.0 | GO terms, hierarchy and annotations |
| [Reactome](https://reactome.org/license) | CC0 | pathways, hierarchy and membership |
| [IntAct](https://www.ebi.ac.uk/intact/), release 2026-01-09 | CC BY 4.0 | protein–protein interactions |
| [Human Protein Atlas](https://v25.proteinatlas.org/about/licence) v25 | CC BY 4.0 | single-cell-type RNA expression in the target features (v25.proteinatlas.org; Karlsson et al., *Sci. Adv.* 2021) |
| [ChEMBL](https://chembl.gitbook.io/chembl-interface-documentation/about) 37 | CC BY-SA 3.0 | protein sequences of biologics, embedded with ESM-2 |
| [ClinicalTrials.gov](https://clinicaltrials.gov/about-site/terms-conditions) | U.S. National Library of Medicine terms | trial start and completion dates, fetched 2026-10-13 |
| MONDO, EFO (via Open Targets) | CC BY 4.0, Apache 2.0 | disease identifiers and hierarchy |
| Orphanet (via Open Targets) | CC BY 4.0 | genetic-association evidence: Orphadata Science © INSERM 1999, as in Open Targets 26.09, aggregated into scores |
| UniProt (via Open Targets) | CC BY 4.0 | genetic-association evidence |
| [ESM-2 650M](https://github.com/facebookresearch/esm), [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) | MIT | models used to compute molecule and disease features (Lin et al., *Science* 2023; Gu et al., *ACM Trans. Comput. Healthc.* 2021) |

- **ESM-2 molecule features.** `x_esm` and `features/molecule_esm2_full.*` are
  embeddings computed from ChEMBL 37 protein sequences. We treat them as
  computed features of factual sequences, not an adaptation of the ChEMBL
  database, and distribute them under CC BY 4.0 with this attribution to ChEMBL.
- **Sources Open Targets redistributes under its own terms.** Evidence from the
  COSMIC Cancer Gene Census, Genomics England PanelApp and Project Score is
  included as Open Targets 26.09 distributes it. Open Targets' licence statement
  says these sources agreed for their data to be used without restriction by
  Open Targets users. Using those resources directly is governed by their own
  terms, which restrict redistribution and commercial use. Their contribution is
  listed per source (`datasource`) in `scores/target_disease_by_source.parquet`.
  The graph's edge scores combine it with other sources.
- No endorsement by any source is implied.

```
Siu, Cabrera, Mudaliar, and Zubiaga.
THBKG: A Temporal Biomedical Knowledge Graph for Leak-Free Clinical Advancement Prediction.
```

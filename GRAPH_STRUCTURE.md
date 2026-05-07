# Heterogeneous Graph Structure

Graph file: `/gpfs/scratch/bty414/opentarget_evidences/23.06/graph/hetero_graph_with_features_datatype.pt`  
Data release: Open Targets 23.06  
Time range: 1995–2025

---

## Node Types

| Node type | Count | Feature dim | Description |
|-----------|------:|:-----------:|-------------|
| `target` | 19,316 | 56 | Protein-coding gene targets (ENSEMBL IDs). Features encode biotype and functional annotations. |
| `disease` | 11,403 | 256 | Diseases and phenotypes from EFO ontology. Features encode therapeutic area embeddings. |
| `go` | 14,686 | 64 | Gene Ontology terms (biological process, molecular function, cellular component). |
| `reactome` | 1,218 | 64 | Reactome pathway nodes. |
| `molecule` | 2,703 | 1,024 | Drug/compound nodes (ChEMBL). Features are derived from canonical SMILES (1024-dim). |

---

## Edge Types

All edges carry two attributes:
- **`edge_attr`** — 2-dim float vector: `[score, recency_weighted_score]` (cumulative association score at a given year and a recency-discounted variant)
- **`edge_time`** — integer year of the evidence snapshot (or `INT64_MIN` for static edges)

### Temporal Edges

Temporal edges have a valid `edge_time` year (1990–2025). They represent evidence that accumulated over time and are filtered/snapshotted by year during training.

| Edge type | Source → Target | Count | Time range | Data sources | Description |
|-----------|----------------|------:|:----------:|-------------|-------------|
| `clinical_trial_positive` | disease → target | 56,190 | 1995–2022 | ChEMBL | Clinical trials with a positive (successful) outcome |
| `clinical_trial_Unknown/Operational` | disease → target | 52,738 | 1995–2024 | ChEMBL | Trials with unknown status or still operational |
| `clinical_trial_unmet_efficacy` | disease → target | 4,240 | 1997–2022 | ChEMBL | Trials terminated due to unmet efficacy |
| `clinical_trial_adverse_effects` | disease → target | 2,115 | 1998–2022 | ChEMBL | Trials terminated due to adverse effects |
| `genetic_association` | target → disease | 82,408 | 1995–2023 | GWAS credible sets, EVA, gene burden, Genomics England, gene2phenotype, UniProt literature/variants, Orphanet, ClinGen | Genetic evidence linking a gene to a disease |
| `animal_model` | target → disease | 442,082 | 1995–2023 | IMPC | Phenotypic evidence from animal knockout models |
| `literature` | target → disease | 237,795 | 1995–2023 | EuropePMC | Co-mention / NLP-derived association from literature |
| `rna_expression` | target → disease | 171,555 | 2004–2021 | Expression Atlas | Differential RNA expression linking target to disease context |
| `somatic_mutation` | target → disease | 63,557 | 1995–2023 | Cancer Gene Census, IntOGen, EVA somatic, cancer biomarkers | Somatic mutation evidence (cancer-focused) |
| `affected_pathway` | target → disease | 38,804 | 1995–2021 | Reactome, SLAPenrich, CRISPR, CRISPR screen, SysBio, PROGENy | Pathway-level perturbation evidence |
| `involved_in` | target → reactome | 36,972 | 1995–2021 | Reactome, SLAPenrich | Target membership in a Reactome pathway |
| `associated_with` | disease → reactome | 3,639 | 1995–2021 | Reactome, SLAPenrich | Disease association with a Reactome pathway |
| `has_function_in` | target → go | 256,090 | 1995–2023 | Gene Ontology annotations | Target annotated with a GO term |
| `modulated_by` | target → molecule | 45,416 | 1995–2025 | ChEMBL | Target modulated by a drug/compound |
| `interacts_with` | target → target | 659,444 | 1995–2025 | IntAct (protein–protein interaction) | Protein–protein interaction |
| `advancement` | target → disease | 34,410 | 1990–2022 | ChEMBL (derived) | **Label edge.** Indicates that a target–disease pair has advanced to a higher clinical phase at the given year. Used as the prediction target. |

### Static Edges (Ontology / Hierarchy)

Static edges encode fixed hierarchical relationships with no timestamp (`edge_time = INT64_MIN`). Their `edge_attr` is `[0.0, 1.0]` (constant).

| Edge type | Source → Target | Count | Data source | Description |
|-----------|----------------|------:|------------|-------------|
| `is_subtype_of` | disease → disease | 11,548 | EFO / Disease Ontology | Disease ontology parent–child relationship |
| `is_subtype_of` | go → go | 14,415 | Gene Ontology | GO term hierarchy (`is_a` relationships) |
| `is_subpathway_of` | reactome → reactome | 492 | Reactome | Reactome pathway hierarchy |

---

## Directionality

The graph is **directed**. No reverse edges are stored or added at runtime.

- `load_event_graph()` in [src/data/temporal_loader.py](src/data/temporal_loader.py) has a `to_undirected` flag (which would invoke `T.ToUndirected()`), but both training scripts (`train_advancement_hgt.py`, `train_advancement_lambdarank.py`) call it with the default `to_undirected=False`.
- `interacts_with` (target → target) is partially symmetric in content: ~21% of edges (118k/565k) have their reverse pair explicitly present, but this is a property of the source data (IntAct), not intentional bidirectionality.
- Message passing therefore only flows along the stored edge directions.

---

## Summary

- **19 edge relation types** in total: **16 temporal**, **3 static**
- **Directed graph** — no reverse edges added during training
- Static edges are ontology/hierarchy edges that never change
- Temporal edges each carry a year (snapshot) and a 2-dim score: `[raw_score, recency_score]`
- The `advancement` edge is the **prediction target** for the drug advancement task
- The four `clinical_trial_*` edge types are split by trial outcome (positive, adverse effects, unmet efficacy, unknown/operational) from ChEMBL, and flow **disease → target** (i.e., a disease is the source of the trial evidence)
- All other evidence edges flow **target → disease** (the gene/protein is the source)

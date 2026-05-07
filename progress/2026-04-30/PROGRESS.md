# Progress — 2026-04-15 → 2026-04-30

## What got done

- **Better model performance.** New SOTA on advancement prediction —
  the proposed undirected EAHGT (full model with score + novelty edge
  features) substantially outperforms the prior baseline and both
  information-retrieval references (RDG, OTS).
- **Ablation study completed.** All eight configurations (b1–b5, p1–p3)
  re-run under a single canonical recipe so only the ablation knob
  varies. Headline takeaways: edge attributes help jointly more than
  individually, HGT beats GATv2 across the board, and RTE trades
  top-of-list ranking for AUC. Write-up in [ABLATION.md](ABLATION.md).
- **Graph updated to OpenTargets 26.03.** Refreshed datasources
  (dropped `slapenrich`/`sysbio`, replaced `chembl` with
  `clinical_precedence`, added `gwas_credible_sets` and `intogen`),
  updated parsers for the new unified date columns, and re-ran the
  preprocessing pipeline.

## In progress

- **Explainability.** Working on retrieving the relevant subgraph for a
  prediction and tracing each edge back to its supporting literature
  (PMID) so predictions can be presented with evidence. The literature
  sidecar piece is in — every built graph now has a companion
  `_literature.parquet` mapping `(source, target, datasource, year)` to
  PMIDs / evidence IDs, with a loader API for cumulative temporal
  queries and PubMed/OT link formatting. Next: subgraph extraction
  around a target prediction and end-to-end "prediction → subgraph →
  citations" walkthrough.

## To do

- **Poster for DERI Day (2026-05-07).** Update with the new SOTA
  numbers, ablation headline, and an explainability teaser.
- **Midway presentation deck (.pptx).** Pull together model results,
  ablation findings, 26.03 graph upgrade, and the explainability
  direction.
- Multi-seed reruns of the ablation for mean ± std reporting.
- Restore `ndcg_ta_mean@K` early-stop metric (currently degenerate).
- First downstream eval on the OT 26.03 graph.
- Subgraph + literature retrieval wired into a usable explainability
  view.

"""Full Open Targets records behind parsed evidence rows (opt-in, slower).

Parsed rows carry the Open Targets evidence ``id``; the full record lives in
the downloaded release under ``evidenceDated/sourceId=<datasource>/``. This
module fetches those records by id and attaches them as JSON, either as a
per-datasource set of key fields (``key``) or as the whole record (``full``).

Special cases
-------------
* clinical_precedence: trial resolution wrote one "ongoing" row per trial plus
  one row per resolved outcome, all sharing the trial's evidence id. A
  ``clinical_trial_<bucket>`` edge is matched on (id, bucket). A
  ``modulated_by`` edge takes the resolved row when one exists and lists every
  bucket seen for the trial.
* gene_ontology (``has_function_in``): there is no evidence id. Details come
  from the target's GO annotations, matched on GO term and cited PubMed ID.
* intact (``interacts_with``): PubMed IDs only. The raw MITAB file is not
  joined.
* ontology hierarchy edges: no evidence record exists; details state that.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

COMMON_FIELDS = ["id", "datasourceId", "datatypeId", "targetFromSourceId",
                 "diseaseFromSource", "diseaseFromSourceMappedId", "score",
                 "evidenceDate", "publicationDate", "literature"]

KEY_FIELDS: dict[str, list[str]] = {
    "clinical_precedence": ["drugId", "drugFromSource", "clinicalReportId",
                            "clinicalStage", "studyStartDate", "trialWhyStopped",
                            "trialStopReasonCategories", "trial_outcome_bucket",
                            "resolvedTrialDate"],
    "eva": ["variantId", "variantRsId", "variantHgvsId",
            "variantFunctionalConsequenceId", "clinicalSignificances",
            "alleleOrigins", "allelicRequirements", "confidence", "studyId",
            "cohortPhenotypes"],
    "eva_somatic": ["variantId", "variantRsId", "variantHgvsId",
                    "variantFunctionalConsequenceId", "clinicalSignificances",
                    "alleleOrigins", "confidence", "studyId", "cohortPhenotypes"],
    "gwas_credible_sets": ["studyLocusId", "resourceScore", "curationDate"],
    "gene_burden": ["projectId", "cohortId", "ancestry", "statisticalMethod",
                    "pValueMantissa", "pValueExponent", "beta", "oddsRatio",
                    "studySampleSize", "studyCases", "allelicRequirements"],
    "europepmc": ["textMiningSentences", "pmcIds", "publicationYear",
                  "resourceScore"],
    "impc": ["targetInModel", "targetInModelMgiId", "biologicalModelId",
             "biologicalModelAllelicComposition",
             "biologicalModelGeneticBackground",
             "diseaseModelAssociatedHumanPhenotypes",
             "diseaseModelAssociatedModelPhenotypes", "resourceScore"],
    "expression_atlas": ["studyId", "studyOverview", "contrast",
                         "log2FoldChangeValue", "log2FoldChangePercentileRank",
                         "resourceScore", "confidence"],
    "crispr": ["diseaseCellLines", "resourceScore"],
    "crispr_screen": ["studyId", "projectId", "studyOverview", "cellType",
                      "crisprScreenLibrary", "contrast", "log2FoldChangeValue",
                      "resourceScore"],
    "cancer_biomarkers": ["biomarkerName", "drugFromSource", "drugId",
                          "drugResponse", "confidence", "biomarkers", "urls"],
    "cancer_gene_census": ["mutatedSamples", "resourceScore"],
    "intogen": ["cohortId", "cohortShortName", "cohortDescription",
                "significantDriverMethods", "mutatedSamples", "resourceScore"],
    "clingen": ["studyId", "confidence", "allelicRequirements", "urls"],
    "gene2phenotype": ["studyId", "confidence", "allelicRequirements",
                       "variantFunctionalConsequenceId"],
    "genomics_england": ["studyId", "studyOverview", "confidence",
                         "allelicRequirements", "cohortPhenotypes"],
    "orphanet": ["confidence", "alleleOrigins", "variantFunctionalConsequenceId"],
    "reactome": ["pathways", "reactionId", "reactionName", "targetModulation",
                 "variantAminoacidDescriptions"],
    "uniprot_literature": ["confidence", "targetModulation"],
    "uniprot_variants": ["variantId", "variantRsId", "confidence",
                         "targetModulation"],
}

_CLINICAL = "clinical_precedence"
_ONGOING = "clinical_trial_ongoing"


def to_json(obj) -> str:
    return json.dumps(obj, default=_json_default, ensure_ascii=False)


def _json_default(o):
    if isinstance(o, (_dt.date, _dt.datetime, pd.Timestamp)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, bytes):
        return o.decode("utf-8", "replace")
    return str(o)


def _clean(record: dict) -> dict:
    """Drop null / empty fields so records stay readable."""
    return {k: v for k, v in record.items()
            if v is not None and not (isinstance(v, (list, dict, str)) and len(v) == 0)}


class RawEvidenceStore:
    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)
        self._warned: set[str] = set()

    def _warn(self, datasource: str, msg: str) -> None:
        if datasource not in self._warned:
            self._warned.add(datasource)
            print(f"[evidence] {datasource}: {msg}; rows keep their parsed fields "
                  "only", flush=True)

    def read_by_ids(self, datasource: str, ids: set[str],
                    columns: list[str] | None) -> list[dict] | None:
        """Raw records whose ``id`` is in ``ids``; ``None`` if the source's raw
        files are not available locally.

        Reads the ``id`` column first and loads full rows only for row groups
        that contain a requested id, so large sources (europepmc) stay cheap.
        """
        d = self.raw_dir / f"sourceId={datasource}"
        files = sorted(d.rglob("*.parquet")) if d.is_dir() else []
        if not files:
            self._warn(datasource, f"no raw parquet files under {d}")
            return None
        want = pa.array(sorted(ids), type=pa.string())
        records: list[dict] = []
        for path in files:
            pf = pq.ParquetFile(path)
            names = pf.schema_arrow.names
            if "id" not in names:
                continue
            cols = None if columns is None else [c for c in columns if c in names]
            for rg in range(pf.num_row_groups):
                mask = pc.is_in(pf.read_row_group(rg, columns=["id"]).column("id"),
                                value_set=want)
                if pc.any(mask).as_py():
                    records.extend(pf.read_row_group(rg, columns=cols)
                                   .filter(mask).to_pylist())
        return records

    def attach(self, ev: pd.DataFrame, mode: str) -> pd.DataFrame:
        """Return ``ev`` with a ``details`` JSON column (and ``raw_record`` in
        ``full`` mode). Rows with no recoverable record get ``None``. PubMed
        IDs cited by a raw record are added to the row's ``pmids``."""
        if mode not in ("key", "full"):
            raise ValueError(mode)
        ev = ev.copy()
        ev["details"] = None
        if mode == "full":
            ev["raw_record"] = None
        if ev.empty:
            return ev

        # Ontology hierarchy edges: nothing to fetch.
        static = ev["is_static"].astype(bool)
        ev.loc[static, "details"] = to_json(
            {"provenance": "ontology hierarchy; no evidence record"})

        has_id = ev["evidence_id"].notna() & ~static
        for datasource, idx in ev[has_id].groupby("datasourceId").groups.items():
            self._attach_by_id(ev, idx, datasource, mode)

        go = (ev["datasourceId"] == "gene_ontology") & ~static
        if go.any():
            self._attach_go(ev, ev.index[go])
        return ev

    def _attach_by_id(self, ev, idx, datasource, mode) -> None:
        key_cols = list(dict.fromkeys(["id"] + COMMON_FIELDS
                                      + KEY_FIELDS.get(datasource, [])))
        records = self.read_by_ids(datasource, set(ev.loc[idx, "evidence_id"]),
                                   None if mode == "full" else key_cols)
        if records is None:
            return
        by_id: dict[str, list[dict]] = {}
        for rec in records:
            by_id.setdefault(rec["id"], []).append(rec)

        for i in idx:
            recs = by_id.get(ev.at[i, "evidence_id"], [])
            rec = self._pick(recs, ev.at[i, "relation"]) if datasource == _CLINICAL \
                else (recs[0] if recs else None)
            if rec is None:
                continue
            key = {k: rec[k] for k in key_cols if k in rec}
            if datasource == _CLINICAL:
                key["trial_outcome_buckets"] = sorted(
                    {r.get("trial_outcome_bucket") for r in recs} - {None})
            ev.at[i, "details"] = to_json(_clean(key))
            if mode == "full":
                ev.at[i, "raw_record"] = to_json(rec)
            lit = [str(p) for p in (rec.get("literature") or []) if p]
            if lit:
                ev.at[i, "pmids"] = list(dict.fromkeys(list(ev.at[i, "pmids"]) + lit))

    @staticmethod
    def _pick(recs: list[dict], relation: str) -> dict | None:
        if not recs:
            return None
        if relation.startswith("clinical_trial_"):
            exact = [r for r in recs if r.get("trial_outcome_bucket") == relation]
            return exact[0] if exact else None
        resolved = [r for r in recs if r.get("trial_outcome_bucket") != _ONGOING]
        return (resolved or recs)[0]

    def _attach_go(self, ev, idx) -> None:
        tdir = self.raw_dir / "targets"
        if not tdir.is_dir():
            return
        if not any(tdir.rglob("*.parquet")):
            self._warn("gene_ontology", f"no target annotations under {tdir}")
            return
        targets = sorted(set(ev.loc[idx, "sourceId"]))
        rows = ds.dataset(tdir, format="parquet").to_table(
            filter=pc.field("id").isin(targets), columns=["id", "go"]).to_pylist()
        annots: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            for a in row["go"] or []:
                annots.setdefault((row["id"], a["id"]), []).append(a)
        for i in idx:
            cands = annots.get((ev.at[i, "sourceId"], ev.at[i, "targetId"]), [])
            cited = {f"PMID:{p}" for p in ev.at[i, "pmids"]}
            match = [a for a in cands if a.get("source") in cited] or cands
            if match:
                ev.at[i, "details"] = to_json({"go_annotations": [_clean(a) for a in match]})

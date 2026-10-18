"""Unit tests for src.evidence on a tiny synthetic release.

Run with the standard library (no pytest needed):
    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.evidence import query as Q
from src.evidence.extract import EvidenceExtractor, main
from src.evidence.release import NodeMappings, ReleasePaths

T1, T2 = "ENSG00000000001", "ENSG00000000002"
D1, D2 = "MONDO_0000001", "MONDO_0000002"
M1, GO1 = "CHEMBL1", "GO:0000001"


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _edge(src, dst, st, rel, dt, ds, score, year, **extra):
    return dict(sourceId=src, targetId=dst, source_type=st, target_type=dt,
                relation=rel, datasourceId=ds, score=score, year=year, **extra)


def build_release(root: Path) -> ReleasePaths:
    base = root / "26.09"
    e = base / "evidences" / "edges"
    _write(e / "target_genetic_association_disease_eva.parquet", [
        _edge(T1, D1, "target", "genetic_association", "disease", "eva", 0.9, 2015.0, id="e1"),
        _edge(T1, D1, "target", "genetic_association", "disease", "eva", 0.5, 2019.0, id="e2"),
        _edge(T2, D1, "target", "genetic_association", "disease", "eva", 0.3, 2010.0, id="e3"),
    ])
    _write(e / "target_clinical_trial_positive_disease_clinical_precedence.parquet", [
        _edge(T1, D1, "target", "clinical_trial_positive", "disease",
              "clinical_precedence", 1.0, 2017.0, id="c1"),
    ])
    _write(e / "molecule_modulated_by_target_clinical_precedence.parquet", [
        _edge(M1, T1, "molecule", "modulated_by", "target",
              "clinical_precedence", 1.0, 2012.0, id="c1"),
    ])
    _write(e / "target_interacts_with_target_intact.parquet", [
        _edge(T2, T1, "target", "interacts_with", "target", "intact", 0.5, 2011.0,
              literature=["111"]),
        _edge(T1, T2, "target", "interacts_with", "target", "intact", 0.5, 2013.0,
              literature=["222"]),
    ])
    _write(e / "target_has_function_in_go_gene_ontology.parquet", [
        _edge(T1, GO1, "target", "has_function_in", "go", "gene_ontology", 1.0,
              2014.0, literature=["333"]),
    ])
    _write(e / "target_literature_disease_europepmc.parquet", [
        _edge(T1, D2, "target", "literature", "disease", "europepmc", 0.2, 2020, id="p1"),
    ])
    _write(base / "evidences" / "static_edges" /
           "disease_is_subtype_of_disease_disease_ontology.parquet", [
        _edge(D2, D1, "disease", "is_subtype_of", "disease", "disease_ontology",
              1.0, None),
    ])

    raw = base / "evidenceDated"
    _write(raw / "sourceId=eva" / "part-0.parquet", [
        dict(id="e1", targetId=T1, diseaseId=D1, datasourceId="eva", score=0.9,
             variantRsId="rs1", literature=["444"], studyId="RCV1"),
        dict(id="e2", targetId=T1, diseaseId=D1, datasourceId="eva", score=0.5,
             variantRsId="rs2", literature=None, studyId="RCV2"),
        dict(id="e3", targetId=T2, diseaseId=D1, datasourceId="eva", score=0.3,
             variantRsId="rs3", literature=None, studyId="RCV3"),
    ])
    _write(raw / "sourceId=clinical_precedence" / "part-0.parquet", [
        dict(id="c1", targetId=T1, diseaseId=D1, drugId=M1, datasourceId="clinical_precedence",
             clinicalReportId="nct01234567", clinicalStage="PHASE_2",
             trial_outcome_bucket="clinical_trial_ongoing", resolvedTrialDate="2012-01-01"),
        dict(id="c1", targetId=T1, diseaseId=D1, drugId=M1, datasourceId="clinical_precedence",
             clinicalReportId="nct01234567", clinicalStage="PHASE_2",
             trial_outcome_bucket="clinical_trial_positive", resolvedTrialDate="2017-05-01"),
    ])
    _write(raw / "targets" / "part-0.parquet", [
        dict(id=T1, approvedSymbol="GENE1", go=[
            dict(id=GO1, source="PMID:333", evidence="IDA", aspect="F",
                 geneProduct="P1", ecoId="ECO:1"),
            dict(id=GO1, source="GO_REF:1", evidence="IEA", aspect="F",
                 geneProduct="P1", ecoId="ECO:2"),
        ]),
        dict(id=T2, approvedSymbol="GENE2", go=[]),
    ])
    _write(raw / "diseases" / "part-0.parquet",
           [dict(id=D1, name="disease one"), dict(id=D2, name="disease two")])
    _write(raw / "molecule" / "part-0.parquet", [dict(id=M1, name="drug one")])
    # A purged source: the directory exists but holds no parquet files.
    (raw / "sourceId=europepmc").mkdir(parents=True)

    prog = base / "progression"
    prog.mkdir(parents=True, exist_ok=True)
    torch.save({"node_mapping": {
        "target": {T1: 0, T2: 1}, "disease": {D1: 0, D2: 1},
        "molecule": {M1: 0}, "go": {GO1: 0}, "reactome": {}}},
        prog / "temporal_graph_datatype_mappings.pt")
    return ReleasePaths(root, "26.09")


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.paths = build_release(cls.root)
        cls.ex = EvidenceExtractor(cls.paths)
        cls.mappings = NodeMappings(cls.paths.default_mappings_file)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_paths(self, *paths, **kw):
        return self.ex.extract(Q.from_path_strings(list(paths)), **kw)

    # -- parsing -----------------------------------------------------------
    def test_parse_path_connectors_and_types(self):
        q = Q.parse_path(f"{M1} -[modulated_by]-> {T1} <-[rev_x]- {T2} -- disease:GO_0000050 "
                         f"-[clinical_trial_*]- {D1}", "p")
        self.assertEqual(q["src_type"].tolist(), ["molecule", "target", "target", "disease"])
        self.assertEqual(q["relation"].tolist(),
                         ["modulated_by", "rev_x", None, "clinical_trial_*"])
        self.assertEqual(q["dst_id"].tolist(), [T1, T2, "GO_0000050", D1])
        with self.assertRaises(ValueError):
            Q.parse_path(T1, "p")

    # -- lookup ------------------------------------------------------------
    def test_wildcard_pair_collects_all_relations(self):
        ev = self.run_paths(f"{T1} -- {D1}").evidence
        self.assertEqual(sorted(ev["datasourceId"]), ["clinical_precedence", "eva", "eva"])
        self.assertTrue((ev["sourceId"] == T1).all())

    def test_cross_pairs_are_excluded(self):
        ev = self.run_paths(f"{T1} -- {D1}", f"{T2} -- {D2}").evidence
        self.assertNotIn("europepmc", set(ev["datasourceId"]))
        self.assertEqual(set(ev.loc[ev.query_id == "path2", "datasourceId"]), set())

    def test_reverse_relation_and_auto_flip(self):
        rev = self.ex.extract(pd.DataFrame([dict(
            query_id="q", hop=0, src_type="disease", src_id=D1,
            relation="rev_genetic_association", dst_type="target", dst_id=T1,
            weight=None, input_edge_type=None)])).evidence
        flip = self.run_paths(f"{D1} -[genetic_association]-> {T1}").evidence
        self.assertEqual(len(rev), 2)
        self.assertEqual(sorted(rev["evidence_id"]), sorted(flip["evidence_id"]))

    def test_same_type_relation_both_orientations(self):
        ev = self.run_paths(f"{T1} -[interacts_with]- {T2}").evidence
        self.assertEqual(sorted(p for ps in ev["pmids"] for p in ps), ["111", "222"])

    def test_relation_glob(self):
        ev = self.run_paths(f"{T1} -[clinical_trial_*]- {D1}").evidence
        self.assertEqual(ev["relation"].tolist(), ["clinical_trial_positive"])

    def test_as_of_drop_and_flag(self):
        dropped = self.run_paths(f"{T1} -- {D1}", f"{D2} -- {D1}", as_of=2016).evidence
        self.assertEqual(sorted(dropped["year"].dropna().astype(int)), [2015])
        # Ontology edges are undated by design and survive the cut.
        self.assertIn("disease_ontology", set(dropped["datasourceId"]))
        flagged = self.run_paths(f"{T1} -- {D1}", as_of=2016, year_policy="flag").evidence
        self.assertEqual(len(flagged), 3)
        self.assertEqual(int(flagged["after_cutoff"].sum()), 2)

    def test_unresolved_relation(self):
        res = self.run_paths(f"{T1} -[no_such_relation]- {D1}")
        self.assertTrue(res.evidence.empty)
        self.assertEqual(len(res.unresolved), 1)

    # -- raw records -------------------------------------------------------
    def test_raw_key_details(self):
        ev = self.run_paths(f"{M1} -[modulated_by]- {T1} -- {D1}", f"{T1} -- {GO1}",
                            f"{T1} -[interacts_with]- {T2}", raw="key").evidence
        det = {(r.datasourceId, r.relation, r.evidence_id or r.pmids[0]):
               (json.loads(r.details) if r.details else None)
               for r in ev.itertuples()}
        self.assertEqual(det[("eva", "genetic_association", "e1")]["variantRsId"], "rs1")
        pos = det[("clinical_precedence", "clinical_trial_positive", "c1")]
        self.assertEqual(pos["trial_outcome_bucket"], "clinical_trial_positive")
        mod = det[("clinical_precedence", "modulated_by", "c1")]
        self.assertEqual(mod["trial_outcome_bucket"], "clinical_trial_positive")
        self.assertEqual(mod["trial_outcome_buckets"],
                         ["clinical_trial_ongoing", "clinical_trial_positive"])
        go = det[("gene_ontology", "has_function_in", "333")]["go_annotations"]
        self.assertEqual([a["evidence"] for a in go], ["IDA"])
        self.assertIsNone(det[("intact", "interacts_with", "111")])
        # PubMed IDs cited by the raw record are added to the row.
        e1 = ev[ev.evidence_id == "e1"].iloc[0]
        self.assertEqual(list(e1.pmids), ["444"])

    def test_raw_unavailable_source_is_skipped(self):
        ev = self.run_paths(f"{T1} -[literature]- {D2}", raw="full").evidence
        self.assertEqual(len(ev), 1)
        self.assertIsNone(ev.iloc[0].details)
        self.assertIsNone(ev.iloc[0].raw_record)

    # -- index-based inputs ------------------------------------------------
    def test_pagelink_paths_and_drift(self):
        f = self.root / "paths.parquet"
        path = ("target#0 -> [target::interacts_with::target] target#1 -> "
                "[target::genetic_association::disease] disease#0")
        pd.DataFrame([dict(target_id=T1, disease_id=D1, rank=0, path=path)]).to_parquet(f)
        q = Q.from_pagelink_paths(f, self.mappings)
        self.assertEqual(q[["src_id", "dst_id"]].values.tolist(), [[T1, T2], [T2, D1]])
        ev = self.ex.extract(q).evidence
        self.assertEqual(set(ev["hop"]), {0, 1})
        pd.DataFrame([dict(target_id=T2, disease_id=D1, rank=0, path=path)]).to_parquet(f)
        with self.assertRaises(SystemExit):
            Q.from_pagelink_paths(f, self.mappings)

    def test_explained_edges_topk_and_drift(self):
        f = self.root / "edges.parquet"
        rows = [dict(target_id=T1, disease_id=D1, edge_type="target::interacts_with::target",
                     src=0, dst=1, ig_total=0.1),
                dict(target_id=T1, disease_id=D1,
                     edge_type="disease::rev_genetic_association::target",
                     src=0, dst=0, ig_total=-0.7)]
        pd.DataFrame(rows).to_parquet(f)
        q = Q.from_explained_edges(f, self.mappings, top_k=1)
        self.assertEqual(q["relation"].tolist(), ["rev_genetic_association"])
        self.assertEqual(len(self.ex.extract(q).evidence), 2)
        bad = [dict(r, target_id=T2, disease_id=D2) for r in rows[:1]]
        bad[0].update(src=0, dst=0, edge_type="target::interacts_with::target")
        pd.DataFrame(bad).to_parquet(f)
        with self.assertRaises(SystemExit):
            Q.from_explained_edges(f, self.mappings)

    # -- CLI ---------------------------------------------------------------
    def test_cli_writes_outputs(self):
        out = self.root / "out"
        main(["--data-root", str(self.root), "--path", f"{M1} -- {T1} -- {D1}",
              "--raw", "key", "--as-of", "2018", "--out-dir", str(out)])
        for name in ("evidence.parquet", "summary.parquet", "evidence.json", "evidence.md"):
            self.assertTrue((out / name).exists(), name)
        doc = json.loads((out / "evidence.json").read_text())
        hop2 = doc["queries"][0]["hops"][1]
        self.assertEqual(hop2["src"]["name"], "GENE1")
        self.assertEqual(hop2["evidence_page"],
                         f"https://platform.opentargets.org/evidence/{T1}/{D1}")
        links = [l for r in hop2["relations"] for d in r["datasources"]
                 for rec in d["records"] for l in rec["links"]]
        self.assertIn("https://clinicaltrials.gov/study/NCT01234567", links)
        trial = [rec for r in hop2["relations"] for d in r["datasources"]
                 for rec in d["records"] if d["datasourceId"] == "clinical_precedence"][0]
        self.assertTrue(trial["links"][0].startswith("https://clinicaltrials.gov/"))
        self.assertIn("https://pubmed.ncbi.nlm.nih.gov/444/", links)
        self.assertIn("disease one", (out / "evidence.md").read_text())


if __name__ == "__main__":
    unittest.main()

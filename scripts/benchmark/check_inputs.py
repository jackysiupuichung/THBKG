#!/usr/bin/env python
"""Check a THBKG release against the Open Targets input it was built from.

check_release.py checks that a release is well formed. This checks that it is
complete: every (source, target) pair the build's rules keep from its own input
is in the graph, first dated in the year of its earliest kept row, and the graph
holds no other pair. A release can be well formed and still have lost input
rows (the 26.09 build lost 11% of IntAct and 48,582 GO rows to a failed PubMed
request); that fails here, without reference to any other release.

    python scripts/benchmark/check_inputs.py --release-dir R --evidence-dir OT/evidenceDated \\
        --trials-dir OT/raw_backup/sourceId=clinical_precedence --pubmed-cache P.parquet \\
        [--edge-schema config/edge_schema_26.09.yaml] [--out R/check_inputs.json]

The build's rules, applied here to the raw Open Targets files (re-implemented,
so a bug in the parser does not hide itself):
  rows       one per input row, or per list element for list fields (go.id,
             pathways.id), as config/edge_schema_<version>.yaml specifies
  endpoints  both ids must be in the Open Targets node index of their type,
             targets protein-coding only (the parser drops the rest); every
             kept pair's endpoints must then be graph nodes
  year       the first non-null column of parsers.parser.YEAR_PRIORITY; a date
             after EVIDENCE_CUTOFF counts as missing (no fallback); a row still
             undated with PMIDs (GO, IntAct) takes its earliest PubMed year from
             the build's PubMed cache; years after time_range.last_year are
             missing, earlier ones count in time_range.first_year
  score      present and > 0 (constant for constant-score specs)
  trials     raw clinical_precedence: NCT trials with a start date. The ongoing
             edge and modulated_by come from every such trial, dated at its
             start; outcome edges only from trials that ended (the cache of
             ClinicalTrials.gov statuses decides), so outcome relations are
             checked one way: no outcome pair outside those trials
  intact     human.txt: protein-protein rows with an Ensembl gene each, no
             self-loops, each pair once as (min, max), dated through PubMed
  static     disease and Reactome parents; GO is_a of non-obsolete terms

Checks (hard)
  inputs.<relation>.missing     kept input pairs absent from the graph
  inputs.<relation>.extra       graph pairs no kept input row supports
  inputs.<relation>.first_year  pairs whose first graph year differs from
                                their earliest kept row
  inputs.relations_covered      every release relation has an input rule
  inputs.pubmed.cached          every PMID a row needs is in the PubMed cache
  inputs.pubmed.no_year         at most inputs.pubmed_no_year_max (contract) of
                                those PMIDs have no year in PubMed
Reported: per input source, how many rows each rule drops.

Heavy (reads every input file): run via sbatch, as bench_04_assemble.sh does.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "preprocessing" / "temporal_graph"))
from src.benchmark.release import Release, edge_key  # noqa: E402
# the build's dating constants; the rules themselves are re-implemented below
from parsers.parser import EVIDENCE_CUTOFF, MAX_EVIDENCE_YEAR, YEAR_PRIORITY  # noqa: E402
from parsers.apply_trial_resolution import NCT_RE  # noqa: E402

# YEAR_PRIORITY columns holding dates (parser._add_props); the others hold years
DATE_COLS = {"resolvedTrialDate", "studyStartDate", "publicationDate", "evidenceDate"}
PMID_RE = re.compile(r"(?:PMID|pubmed)[\s:]*(\d+)", re.IGNORECASE)
# Open Targets node datasets (config/node_schema.yaml), per graph node type
NODE_DIRS = {"target": "targets", "disease": "diseases", "molecule": "molecule", "reactome": "reactome", "go": "go"}
TRIAL_START = "clinical_trial_ongoing"
TRIAL_OUTCOMES = {"clinical_trial_positive", "clinical_trial_Unknown/Operational",
                  "clinical_trial_unmet_efficacy", "clinical_trial_adverse_effects"}
DROP = {  # report labels for the rules, in the order they are applied
    "endpoint": "endpoint not an Open Targets entity",
    "undated": "no date and no PMID",
    "future": "dated after the cutoff",
    "no_pubmed_year": "PMIDs without a PubMed year",
    "uncached": "PMIDs not in the PubMed cache (unverifiable)",
    "score": "no score or score <= 0",
    "self_loop": "self-loop",
    "not_nct": "not a ClinicalTrials.gov trial",
    "no_start": "no start date",
}


def pmids_of(v) -> list[str]:
    """PMIDs in a literature or source value, a single value or a list: bare digits, or the number in
    'PMID:123' / 'pubmed: 123' (as parser.EdgeParser._extract_literature)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    vals = list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v]
    out = []
    for x in vals:
        s = str(x).strip()
        if s.isdigit():
            out.append(s)
        else:
            m = PMID_RE.search(s)
            if m:
                out.append(m.group(1))
    return out


def struct_pmids(t) -> list[str]:
    """PMIDs of one list element, e.g. a GO annotation: its literature, source or pubmedId field, the first
    one set, as the parser reads it."""
    if not isinstance(t, dict):
        return []
    return pmids_of(t.get("literature") or t.get("source") or t.get("pubmedId"))


def parse_dates(s: pd.Series) -> pd.Series:
    """Dates of a column as the parser reads them row by row: pd.to_datetime of each value in any format, NaT
    when unparseable, without time zone. Each distinct value is parsed once."""
    if pd.api.types.is_datetime64_any_dtype(s):
        d = s
    else:
        u = pd.unique(s.dropna())
        parsed = pd.to_datetime(pd.Series(u, dtype=object), errors="coerce", format="mixed")
        d = s.map(dict(zip(u, parsed)))
        d = pd.to_datetime(d, errors="coerce")
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    return d


def row_years(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(year, decided): the first non-null YEAR_PRIORITY column dates a row;
    decided rows whose date is after the cutoff stay NaN (no fallback)."""
    year = np.full(len(df), np.nan)
    decided = np.zeros(len(df), bool)
    for col in YEAR_PRIORITY:
        if col not in df.columns:
            continue
        if col in DATE_COLS:
            d = parse_dates(df[col])
            valid = d.notna().to_numpy() & ~decided
            ok = valid & (d <= EVIDENCE_CUTOFF).fillna(False).to_numpy()
            year[ok] = d[ok].dt.year.to_numpy()
        else:
            v = pd.to_numeric(df[col], errors="coerce")
            valid = df[col].notna().to_numpy() & ~decided
            year[valid] = v[valid].to_numpy()
        decided |= valid
    return year, decided


class PubMed:
    """The build's PubMed cache (PMID -> publication year), and the PMIDs this check needed from it."""

    def __init__(self, path: Path | None):
        """Load the cache at `path` (columns pmid, year; year None when PubMed gave none). No file: an empty
        cache, so every PMID counts as uncached."""
        self.years: dict[str, int | None] = {}
        if path is not None and path.exists():
            c = pd.read_parquet(path)
            self.years = {str(p): (int(y) if y is not None and str(y).isdigit() else None)
                          for p, y in zip(c["pmid"], c["year"])}
        self.needed: set[str] = set()

    def earliest(self, pmid_lists) -> tuple[np.ndarray, np.ndarray]:
        """Per row of PMID lists: the earliest year among its cached PMIDs (NaN if none has one) and whether
        any of its PMIDs is missing from the cache. Records every PMID asked for in `needed`."""
        year = np.full(len(pmid_lists), np.nan)
        uncached = np.zeros(len(pmid_lists), bool)
        for i, ps in enumerate(pmid_lists):
            if not ps:
                continue
            self.needed.update(ps)
            ys = [self.years.get(p) for p in ps if p in self.years]
            ys = [y for y in ys if y is not None]
            if ys:
                year[i] = min(ys)
            uncached[i] = any(p not in self.years for p in ps)
        return year, uncached


class Relation:
    """One graph relation's kept input pairs, in graph node indices, with the year of each kept row; the pairs
    that might be kept (unverifiable); and, per input source, the rows read and the rows each rule dropped."""

    def __init__(self, et, graph_index: dict, symmetric: bool):
        """Empty relation for edge type `et`. `graph_index` maps each node type to a pd.Index of graph node
        ids; `symmetric` orders pairs (min, max)."""
        self.et, self.symmetric = et, symmetric
        self.n_dst = len(graph_index[et[2]])
        self.gi = graph_index
        self.keys, self.years, self.possible = [], [], []
        self.not_node: Counter = Counter()  # kept pairs with an endpoint that is no graph node
        self.not_node_examples: list = []
        self.drops: dict[str, Counter] = defaultdict(Counter)
        self.rows: Counter = Counter()

    def _keys(self, src_ids, dst_ids):
        """Pair keys src * n_dst + dst in graph node indices ((min, max) for symmetric relations), and a mask
        of pairs with an endpoint that is no graph node (key -1)."""
        s = self.gi[self.et[0]].get_indexer(src_ids)
        d = self.gi[self.et[2]].get_indexer(dst_ids)
        if self.symmetric:
            s, d = np.minimum(s, d), np.maximum(s, d)  # graph ids sort as their indices
        bad = (s < 0) | (d < 0)
        return np.where(bad, -1, s * self.n_dst + d), bad

    def add(self, source, src_ids, dst_ids, year, kept, possible, drops: dict):
        """Add one input source's rows: count them and, per rule in `drops`, the rows it removed; store the
        keys and years of the `kept` rows (kept pairs with an endpoint that is no graph node are counted as
        missing instead) and the keys of the `possible` rows."""
        self.rows[source] += len(src_ids)
        for k, m in drops.items():
            self.drops[source][DROP[k]] += int(m.sum())
        for mask, store in ((kept, "kept"), (possible, "possible")):
            if not mask.any():
                continue
            src, dst = np.asarray(src_ids)[mask], np.asarray(dst_ids)[mask]
            keys, bad = self._keys(src, dst)
            if store == "kept":
                self.not_node[source] += int(bad.sum())
                if bad.any() and len(self.not_node_examples) < 5:
                    self.not_node_examples += list(zip(src[bad][:5], dst[bad][:5]))
                self.keys.append(keys[~bad])
                self.years.append(np.asarray(year)[mask][~bad])
            else:
                self.possible.append(keys[~bad])


def node_index(evidence_dir: Path, nt: str) -> pd.Index:
    """Ids of one node type in the Open Targets download, as the parser keeps them
    (NodeParser: protein-coding targets only)."""
    d = ds.dataset(str(evidence_dir / NODE_DIRS[nt]), format="parquet")
    if nt == "target" and "biotype" in d.schema.names:
        t = d.to_table(columns=["id", "biotype"]).to_pandas()
        ids = t.loc[t["biotype"] == "protein_coding", "id"].to_numpy(dtype=object)
    else:
        ids = np.asarray(d.to_table(columns=["id"]).column("id").to_pylist(), dtype=object)
    return pd.Index(pd.unique(ids))


def spec_list(schema: dict):
    """Every (name, spec) of an edge schema, once per spec where a name lists several."""
    for name, spec in schema.items():
        for s in (spec if isinstance(spec, list) else [spec]):
            yield name, s


def constant(props, key):
    """The value of prop `key=constant:value` in a spec's props, or None if the prop is not constant."""
    for p in props:
        if isinstance(p, str) and p.startswith(f"{key}=constant:"):
            return p.split("constant:", 1)[1]
    return None


def evidence_spec(rel: Relation, evidence_dir: Path, name: str, spec: dict, ot: dict, pubmed: PubMed,
                  first: int, last: int) -> None:
    """Apply the rules to the input of one edge-schema spec, a parquet file at a time: expand list fields
    (go.id, pathways.id), keep rows with both endpoints in the Open Targets index, date them (PubMed for
    undated rows with PMIDs), keep a score above 0, and add the rows to `rel`."""
    props = spec.get("props", [])
    src_col, dst_col = spec["sourceId"], spec["targetId"]
    src_t, dst_t = constant(props, "source_type"), constant(props, "target_type")
    const_score = constant(props, "score")
    base = dst_col.split(".", 1)[0]
    d = ds.dataset(str(evidence_dir / spec["input_dir"]), format="parquet")
    names = set(d.schema.names)
    cols = [src_col, base] + [c for c in YEAR_PRIORITY if c in names]
    if const_score is None:
        cols.append("score")
    if "literature" in props and "literature" in names:
        cols.append("literature")
    cols = list(dict.fromkeys(c for c in cols if c in names))
    for frag in d.get_fragments():
        df = frag.to_table(columns=cols).to_pandas()
        if "." in dst_col or isinstance(df[base].dropna().iloc[0] if df[base].notna().any() else None,
                                        (list, np.ndarray)):
            sub = dst_col.split(".", 1)[1] if "." in dst_col else None
            df = df.explode(base, ignore_index=True)
            el = df[base]
            if sub is not None:
                df["_dst"] = el.map(lambda t: t.get(sub) if isinstance(t, dict) else None)
            else:  # plain values, or GO-style dicts with an id
                df["_dst"] = el.map(lambda t: t.get("id") if isinstance(t, dict) else t)
            df["_pmids"] = el.map(struct_pmids)
            df = df[df["_dst"].notna()].reset_index(drop=True)
        else:
            df["_dst"] = df[dst_col]
            df["_pmids"] = df["literature"].map(pmids_of) if "literature" in df.columns else [[] for _ in range(len(df))]
        src_ids, dst_ids = df[src_col].astype(str).to_numpy(), df["_dst"].astype(str).to_numpy()
        rest = (ot[src_t].get_indexer(src_ids) >= 0) & (ot[dst_t].get_indexer(dst_ids) >= 0)
        year, decided = row_years(df)
        # the parser looks up PubMed for any valid row still without a year (or year 0)
        has_pm = df["_pmids"].map(len).to_numpy() > 0
        use_pm = rest & has_pm & (np.isnan(year) | (year == 0))
        pm_year, uncached = pubmed.earliest([p if u else [] for p, u in zip(df["_pmids"], use_pm)])
        year = np.where(use_pm, pm_year, year)
        year[year > min(MAX_EVIDENCE_YEAR, last)] = np.nan
        undated = np.isnan(year)
        if const_score is None:
            sc = pd.to_numeric(df["score"], errors="coerce").to_numpy()
            score_ok = ~np.isnan(sc) & (sc > 0)
        else:
            score_ok = np.full(len(df), float(const_score) > 0)
        unverifiable = rest & use_pm & np.isnan(pm_year) & uncached
        drops = {"endpoint": ~rest,
                 "undated": rest & undated & ~use_pm & ~decided,
                 "future": rest & undated & ~use_pm & decided,
                 "no_pubmed_year": rest & undated & use_pm & ~unverifiable,
                 "uncached": unverifiable,
                 "score": rest & ~undated & ~score_ok}
        kept = rest & ~undated & score_ok
        rel.add(f"{name}:{spec['input_dir']}", src_ids, dst_ids, np.maximum(year, first), kept,
                unverifiable & score_ok, drops)


def trial_relations(rels: dict, trials_dir: Path, ot: dict, last: int, first: int) -> None:
    """Trial relations from the raw clinical_precedence file: the ongoing edge and modulated_by are kept from
    every NCT trial with a start date up to the cutoff, dated at its start; outcome relations get the same
    pairs as possible only, since only trials that ended carry an outcome (checked one way)."""
    t = ds.dataset(str(trials_dir), format="parquet").to_table(
        columns=["targetId", "diseaseId", "drugId", "clinicalReportId", "studyStartDate"]).to_pandas()
    nct = t["clinicalReportId"].astype(str).str.upper().str.match(NCT_RE).fillna(False).to_numpy()
    start = parse_dates(t["studyStartDate"])
    has_start = start.notna().to_numpy()
    ok_date = (start <= EVIDENCE_CUTOFF).fillna(False).to_numpy() & (start.dt.year <= last).fillna(False).to_numpy()
    year = np.maximum(start.dt.year.to_numpy(dtype=float), first)
    for et, s_col, d_col in ((("target", TRIAL_START, "disease"), "targetId", "diseaseId"),
                             (("molecule", "modulated_by", "target"), "drugId", "targetId")):
        if et not in rels:
            continue
        s_ids, d_ids = t[s_col].astype(str).to_numpy(), t[d_col].astype(str).to_numpy()
        ep = (ot[et[0]].get_indexer(s_ids) >= 0) & (ot[et[2]].get_indexer(d_ids) >= 0)
        drops = {"endpoint": ~ep, "not_nct": ep & ~nct, "no_start": ep & nct & ~has_start,
                 "future": ep & nct & has_start & ~ok_date}
        kept = ep & nct & has_start & ok_date
        rels[et].add("clinical_precedence", s_ids, d_ids, year, kept, np.zeros(len(t), bool), drops)
        if et[1] == TRIAL_START:
            for rel_name in TRIAL_OUTCOMES:
                oet = ("target", rel_name, "disease")
                if oet in rels:  # any trial pair with a start date may carry an outcome
                    rels[oet].add("clinical_precedence", s_ids, d_ids, year, np.zeros(len(t), bool),
                                  ep & nct & has_start, drops)


def intact_relation(rel: Relation, evidence_dir: Path, ot: dict, pubmed: PubMed, first: int, last: int) -> None:
    """IntAct from human.txt as the IntAct parser reads it (protein-protein rows with an Ensembl gene each):
    drop self-loops, order each pair (min, max), date rows through PubMed, and add them to `rel`."""
    rows = []
    with open(evidence_dir / "intact" / "human.txt", encoding="utf-8") as f:
        header = f.readline()
        if not header.startswith("#"):
            f.seek(0)
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) < 24:
                continue
            if not ('psi-mi:"MI:0326"(protein)' in row[20] and 'psi-mi:"MI:0326"(protein)' in row[21]):
                continue
            a, b = re.search(r"ensembl:(ENSG\d+)", row[22]), re.search(r"ensembl:(ENSG\d+)", row[23])
            if not a or not b:
                continue
            rows.append((a.group(1), b.group(1), sorted(set(re.findall(r"pubmed:(\d+)", row[8])))))
    df = pd.DataFrame(rows, columns=["a", "b", "pmids"])
    a, b = df["a"].to_numpy(dtype=str), df["b"].to_numpy(dtype=str)
    loop = a == b
    lo, hi = np.where(a < b, a, b), np.where(a < b, b, a)
    ep = (ot["target"].get_indexer(lo) >= 0) & (ot["target"].get_indexer(hi) >= 0)
    has_pm = df["pmids"].map(len).to_numpy() > 0
    year, uncached = pubmed.earliest([p if h else [] for p, h in zip(df["pmids"], has_pm & ep & ~loop)])
    year[year > min(MAX_EVIDENCE_YEAR, last)] = np.nan
    rest = ~loop & ep
    drops = {"self_loop": loop, "endpoint": ~loop & ~ep, "undated": rest & ~has_pm,
             "no_pubmed_year": rest & has_pm & np.isnan(year) & ~uncached}
    unverifiable = rest & has_pm & np.isnan(year) & uncached
    drops["uncached"] = unverifiable
    kept = rest & ~np.isnan(year)
    rel.add("intact:human.txt", lo, hi, np.maximum(year, first), kept, unverifiable, drops)


def static_relations(rels: dict, evidence_dir: Path, ot: dict) -> None:
    """Ontology edges, all kept and undated: disease and Reactome (id, parent) pairs from their node datasets,
    and GO is_a pairs of non-obsolete terms from go.obo."""
    for nt, rel_name, d in (("disease", "is_subtype_of", "diseases"), ("reactome", "is_subpathway_of", "reactome")):
        et = (nt, rel_name, nt)
        if et not in rels:
            continue
        t = ds.dataset(str(evidence_dir / d), format="parquet").to_table(columns=["id", "parents"]).to_pandas()
        t = t.explode("parents").dropna(subset=["parents"])
        s, p = t["id"].astype(str).to_numpy(), t["parents"].astype(str).to_numpy()
        n = len(s)
        rels[et].add(f"{d}.parents", s, p, np.full(n, -1.0), np.ones(n, bool), np.zeros(n, bool), {})
    et = ("go", "is_subtype_of", "go")
    if et in rels:
        text = (evidence_dir / "go_ontology" / "go.obo").read_text()
        pairs = []
        for block in text.split("\n\n"):
            if not block.lstrip().startswith("[Term]"):
                continue
            m = re.search(r"^id:\s*(GO:\d+)", block, re.MULTILINE)
            if not m or re.search(r"^is_obsolete:\s*true", block, re.MULTILINE):
                continue
            pairs += [(m.group(1), p) for p in re.findall(r"^is_a:\s*(GO:\d+)", block, re.MULTILINE)]
        s, p = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])
        rels[et].add("go.obo is_a", s, p, np.full(len(s), -1.0), np.ones(len(s), bool), np.zeros(len(s), bool), {})


def graph_first_years(r: Release, et, symmetric: bool) -> pd.Series:
    """First year (smallest edge_time) of every pair of one release edge type, keyed src * n_dst + dst ((min,
    max) for symmetric relations)."""
    ei, t, _ = r.edges(et)
    n_dst = r.num_nodes(et[2])
    a, b = (np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])) if symmetric else (ei[0], ei[1])
    return pd.DataFrame({"k": a * n_dst + b, "t": t}).groupby("k")["t"].min()


def main() -> int:
    """Read every input, compare each relation's kept pairs and first years with the release, check the PubMed
    cache, write the JSON report and return 1 if any check failed (0 otherwise)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-dir", required=True, type=Path)
    ap.add_argument("--evidence-dir", required=True, type=Path, help="the build's Open Targets download (evidenceDated)")
    ap.add_argument("--trials-dir", required=True, type=Path,
                    help="raw clinical_precedence, before trial resolution (raw_backup/sourceId=clinical_precedence)")
    ap.add_argument("--pubmed-cache", type=Path, default=None, help="the build's PMID -> year cache")
    ap.add_argument("--edge-schema", type=Path, default=_REPO / "config" / "edge_schema_26.09.yaml")
    ap.add_argument("--config", type=Path, default=_REPO / "config" / "event_graph_config.yaml")
    ap.add_argument("--contract", type=Path, default=_REPO / "config" / "release_contract.yaml")
    ap.add_argument("--out", type=Path, default=None, help="JSON report (default <release>/check_inputs.json)")
    args = ap.parse_args()

    contract = yaml.safe_load(args.contract.read_text())
    kinds = {tuple(e[:3]): e[3] for e in contract["edge_types"]}
    tr = yaml.safe_load(args.config.read_text())["time_range"]
    first, last = int(tr["first_year"]), int(tr["last_year"])
    r = Release(args.release_dir)
    graph_index = {nt: pd.Index(r.node_ids(nt)) for nt in r.node_types}
    ot = {nt: node_index(args.evidence_dir, nt) for nt in NODE_DIRS}
    print("Open Targets node index: " + ", ".join(f"{k} {len(v):,}" for k, v in ot.items()), flush=True)
    pubmed = PubMed(args.pubmed_cache)
    print(f"PubMed cache: {len(pubmed.years):,} PMIDs", flush=True)

    rels = {et: Relation(et, graph_index, kinds.get(et) == "symmetric") for et in r.edge_types}
    schema = yaml.safe_load(args.edge_schema.read_text())
    for name, spec in spec_list(schema):
        if "relation" in spec or spec.get("input_dir") == "sourceId=clinical_precedence":
            continue  # trials: from the raw file, below
        props = spec.get("props", [])
        et = (constant(props, "source_type"), spec["relation_name"], constant(props, "target_type"))
        if et not in rels:
            print(f"SKIP  {name}: {edge_key(et)} is not a release relation")
            continue
        if not (args.evidence_dir / spec["input_dir"]).exists():
            print(f"SKIP  {name}: no input {spec['input_dir']}")
            continue
        evidence_spec(rels[et], args.evidence_dir, name, spec, ot, pubmed, first, last)
        print(f"read  {name} -> {et[1]}", flush=True)
    trial_relations(rels, args.trials_dir, ot, last, first)
    if ("target", "interacts_with", "target") in rels:
        intact_relation(rels[("target", "interacts_with", "target")], args.evidence_dir, ot, pubmed, first, last)
    static_relations(rels, args.evidence_dir, ot)
    print("inputs read\n", flush=True)

    report, failures = {}, []

    def check(name, ok, detail=None):
        """Record check `name`: print PASS or FAIL with its detail and store it for the report."""
        report[name] = {"ok": bool(ok), **({"detail": detail} if detail is not None else {})}
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail is not None else ""), flush=True)
        if not ok:
            failures.append(name)

    uncovered = [edge_key(et) for et, rel in rels.items() if not rel.rows]
    check("inputs.relations_covered", not uncovered, {"no input rule": uncovered} if uncovered else None)
    for et, rel in rels.items():
        if not rel.rows:
            continue
        static = kinds.get(et) == "static"
        outcome = et[1] in TRIAL_OUTCOMES
        exp = (pd.DataFrame({"k": np.concatenate(rel.keys), "y": np.concatenate(rel.years)}).groupby("k")["y"].min()
               if rel.keys else pd.Series(dtype=float))
        possible = np.unique(np.concatenate(rel.possible)) if rel.possible else np.zeros(0, np.int64)
        g = graph_first_years(r, et, rel.symmetric)
        missing = int((~exp.index.isin(g.index)).sum()) + sum(rel.not_node.values())
        extra_keys = g.index[~g.index.isin(exp.index) & ~g.index.isin(possible)]
        common = exp.index.intersection(g.index)
        year_diff = 0 if static else int((exp.loc[common].to_numpy() != g.loc[common].to_numpy()).sum())
        label = edge_key(et)
        detail = {"graph_pairs": len(g), "kept_input_pairs": len(exp), "unverifiable_pairs": len(possible)}

        def ids(keys):
            """Node ids of the first five pair keys, for examples in the report."""
            src, dst = r.node_ids(et[0]), r.node_ids(et[2])
            return [(src[k // rel.n_dst], dst[k % rel.n_dst]) for k in keys[:5]]

        if not outcome:
            m_keys = exp.index[~exp.index.isin(g.index)].to_numpy()
            check(f"inputs.{label}.missing", missing == 0,
                  {**detail, "missing": missing, "endpoint_not_a_graph_node": dict(rel.not_node),
                   "examples": ids(m_keys) + rel.not_node_examples[:3]})
        check(f"inputs.{label}.extra", len(extra_keys) == 0,
              {**detail, "extra": len(extra_keys), "examples": ids(extra_keys.to_numpy())})
        if not static and not outcome:
            diff = common[exp.loc[common].to_numpy() != g.loc[common].to_numpy()]
            check(f"inputs.{label}.first_year", year_diff == 0,
                  {"pairs_compared": len(common), "differ": year_diff,
                   "examples": [(a, b, int(exp[k]), int(g[k])) for (a, b), k in zip(ids(diff.to_numpy()), diff[:5])]})
        report[f"inputs.{label}.drops"] = {s: {"rows": rel.rows[s], **{k: v for k, v in c.items() if v}}
                                          for s, c in rel.drops.items()}

    needed = pubmed.needed
    uncached = [p for p in needed if p not in pubmed.years]
    no_year = [p for p in needed if p in pubmed.years and pubmed.years[p] is None]
    limit = float(contract.get("inputs", {}).get("pubmed_no_year_max", 0.02))
    check("inputs.pubmed.cached", not uncached, f"{len(needed):,} PMIDs needed, {len(uncached):,} not in the cache"
          + (f", e.g. {uncached[:5]}" if uncached else ""))
    check("inputs.pubmed.no_year", len(no_year) <= limit * max(len(needed), 1),
          f"{len(no_year):,} of {len(needed):,} PMIDs without a year (limit {limit:.0%})")

    out = args.out or args.release_dir / "check_inputs.json"
    report["failures"] = failures
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n{'❌ ' + str(len(failures)) + ' check(s) failed' if failures else '✅ all checks passed'} -> {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

"""Extract the Open Targets evidence behind a path or subgraph of THBKG.

Given paths or edges on the graph, returns every evidence row of the locally
downloaded Open Targets release that supports each edge, optionally restricted
to evidence dated strictly before a decision year (the graph's masking rule).
No network access: results are pinned to the downloaded release.

Examples (submit real runs via ``scripts/evidence/run_extract_evidence.sh``)::

    # direct pair, every relation, as of 2018
    python -m src.evidence.extract --path "ENSG00000120217 -- MONDO_0002169" \
        --as-of 2018 --out-dir out/cd274_rectum

    # two-hop path with full Open Targets records
    python -m src.evidence.extract \
        --path "ENSG00000120217 -[interacts_with]- ENSG00000188389 -- MONDO_0002169" \
        --raw key --out-dir out/cd274_pdcd1

    # PaGE-Link explanations (graph indices resolved via the mappings file)
    python -m src.evidence.extract --pagelink-paths run/per_pair_paths.parquet \
        --mappings <release>/progression/temporal_graph_datatype_mappings.pt \
        --out-dir out/pagelink

Outputs in ``--out-dir``: ``evidence.parquet`` (one row per query edge and
evidence row), ``summary.parquet`` (per edge, relation and datasource),
``evidence.json`` (nested path, hop, relation, datasource, records with links)
and ``evidence.md`` (readable brief).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import query as Q
from .names import lookup_names
from .parsed import ParsedEvidenceIndex
from .raw import RawEvidenceStore
from .release import DEFAULT_RELEASE, NodeMappings, ReleasePaths, resolve_data_root
from .render import build_json, summarise, write_outputs

YEAR_POLICIES = ("drop", "flag")
RAW_MODES = ("none", "key", "full")


@dataclass
class EvidenceResult:
    queries: pd.DataFrame
    evidence: pd.DataFrame
    summary: pd.DataFrame
    unresolved: pd.DataFrame
    names: dict
    settings: dict = field(default_factory=dict)

    def to_json(self, max_records: int = 50) -> dict:
        return build_json(self.queries, self.evidence, self.summary,
                          self.unresolved, self.names, self.settings, max_records)

    def write(self, out_dir: str | Path, max_records: int = 50) -> None:
        write_outputs(out_dir, self.evidence, self.summary, self.to_json(max_records))


class EvidenceExtractor:
    def __init__(self, paths: ReleasePaths):
        paths.check()
        self.paths = paths
        self.index = ParsedEvidenceIndex([(paths.parsed_edges_dir, False),
                                          (paths.static_edges_dir, True)])
        self.raw = RawEvidenceStore(paths.raw_dir)

    def extract(self, queries: pd.DataFrame, as_of: int | None = None,
                year_policy: str = "drop", raw: str = "none") -> EvidenceResult:
        if year_policy not in YEAR_POLICIES:
            raise ValueError(f"year_policy must be one of {YEAR_POLICIES}")
        if raw not in RAW_MODES:
            raise ValueError(f"raw must be one of {RAW_MODES}")
        queries = queries.reset_index(drop=True)

        plan, unresolved = self.index.plan(queries)
        ev = self.index.lookup(plan)

        # Temporal scope. Evidence dated in or after the decision year is
        # dropped (or flagged); undated evidence is treated the same way,
        # except ontology hierarchy edges, which carry no date by design.
        if as_of is not None and not ev.empty:
            year = ev["year"]
            late = year.notna() & (year >= as_of)
            undated = year.isna() & ~ev["is_static"].astype(bool)
            ev["after_cutoff"] = (late | undated).astype(bool)
            if year_policy == "drop":
                ev = ev[~ev["after_cutoff"]].drop(columns="after_cutoff")

        ev = ev.reset_index(drop=True)
        if raw != "none":
            ev = self.raw.attach(ev, raw)

        # Carry the query context onto each evidence row.
        ctx = queries[["query_id", "hop", "src_type", "src_id", "relation",
                       "dst_type", "dst_id", "weight"]].add_prefix("query_")
        ctx = ctx.rename(columns={"query_query_id": "query_id", "query_hop": "hop",
                                  "query_weight": "weight"})
        ev = ev.merge(ctx, left_on="query_row", right_index=True, how="left")
        lead = ["query_row", "query_id", "hop", "query_src_type", "query_src_id",
                "query_relation", "query_dst_type", "query_dst_id", "weight"]
        ev = ev[lead + [c for c in ev.columns if c not in lead]]

        ids: dict[str, set[str]] = {}
        for side in ("src", "dst"):
            for t, a in zip(queries[f"{side}_type"], queries[f"{side}_id"]):
                ids.setdefault(t, set()).add(a)
        names = lookup_names(self.paths.raw_dir, ids)

        settings = {"release": self.paths.release, "data_root": str(self.paths.data_root),
                    "as_of": as_of, "year_policy": year_policy, "raw": raw}
        return EvidenceResult(queries=queries, evidence=ev, summary=summarise(ev),
                              unresolved=unresolved, names=names, settings=settings)


def check_nodes_in_graph(queries: pd.DataFrame, mappings: NodeMappings) -> None:
    """Warn about query nodes that are not nodes of the graph (wrong ID space)."""
    missing = set()
    for side in ("src", "dst"):
        for t, a in zip(queries[f"{side}_type"], queries[f"{side}_id"]):
            if not mappings.contains(t, a):
                missing.add((t, a))
    for t, a in sorted(missing):
        print(f"[evidence] warning: {t} {a} is not a node of the graph "
              f"({mappings.path}); check its ID space", flush=True)


def build_queries(args, mappings_loader) -> pd.DataFrame:
    frames = []
    paths = list(args.path or [])
    if args.paths_file:
        paths += Q.read_paths_file(args.paths_file)
    if paths:
        frames.append(Q.from_path_strings(paths))
    for f in args.edges or []:
        frames.append(Q.from_edge_list(f))
    for f in args.pagelink_paths or []:
        frames.append(Q.from_pagelink_paths(f, mappings_loader(), max_rank=args.max_rank))
    for f in args.explained_edges or []:
        frames.append(Q.from_explained_edges(f, mappings_loader(), top_k=args.top_k))
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("[evidence] no query edges: give --path, --paths-file, "
                         "--edges, --pagelink-paths or --explained-edges")
    return pd.concat(frames, ignore_index=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract the Open Targets evidence behind THBKG paths or edges.")
    g = p.add_argument_group("inputs (combinable)")
    g.add_argument("--path", action="append",
                   help='Accession path, e.g. "ENSG00000120217 -[literature]- MONDO_0002169". '
                        "Repeatable.")
    g.add_argument("--paths-file", help="Text file with one accession path per line.")
    g.add_argument("--edges", action="append",
                   help="Edge list (csv/tsv/parquet) with src_id, dst_id and optional "
                        "src_type, dst_type, relation, query_id, hop, weight.")
    g.add_argument("--pagelink-paths", action="append",
                   help="PaGE-Link per_pair_paths.parquet (graph indices).")
    g.add_argument("--explained-edges", action="append",
                   help="Per-edge explanation table, e.g. per_pair_edges.parquet.")
    g.add_argument("--max-rank", type=int, default=None,
                   help="Keep PaGE-Link paths with rank below this.")
    g.add_argument("--top-k", type=int, default=None,
                   help="Keep the k strongest explained edges per pair, by |weight|.")

    r = p.add_argument_group("release")
    r.add_argument("--release", default=DEFAULT_RELEASE)
    r.add_argument("--data-root", default=None,
                   help="Directory holding <release>/; defaults to $THBKG_DATA_ROOT.")
    r.add_argument("--mappings", default=None,
                   help="Mappings file for graph-index inputs and the node check; "
                        "defaults to the release's datatype mappings.")

    s = p.add_argument_group("scope and output")
    s.add_argument("--as-of", type=int, default=None,
                   help="Decision year: keep evidence dated strictly before it.")
    s.add_argument("--year-policy", choices=YEAR_POLICIES, default="drop",
                   help="With --as-of: drop later evidence, or keep and flag it.")
    s.add_argument("--raw", choices=RAW_MODES, default="none",
                   help="Attach Open Targets records: key fields, or full records.")
    s.add_argument("--max-records", type=int, default=50,
                   help="Records per datasource per edge in the JSON (parquet keeps all).")
    s.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    paths = ReleasePaths(resolve_data_root(args.data_root), args.release)
    mappings_file = Path(args.mappings) if args.mappings else paths.default_mappings_file

    cache: dict[str, NodeMappings] = {}

    def mappings_loader() -> NodeMappings:
        if "m" not in cache:
            cache["m"] = NodeMappings(mappings_file)
        return cache["m"]

    queries = build_queries(args, mappings_loader)
    print(f"[evidence] {len(queries)} query edges in "
          f"{queries['query_id'].nunique()} queries", flush=True)
    if mappings_file.exists():
        check_nodes_in_graph(queries, mappings_loader())

    extractor = EvidenceExtractor(paths)
    print(f"[evidence] release {paths.release}: {len(extractor.index.triples())} "
          "stored relations indexed", flush=True)
    result = extractor.extract(queries, as_of=args.as_of,
                               year_policy=args.year_policy, raw=args.raw)
    for row in result.unresolved.itertuples(index=False):
        print(f"[evidence] unresolved query edge {row.query_row}: {row.reason}", flush=True)
    result.write(args.out_dir, max_records=args.max_records)
    print(f"[evidence] {len(result.evidence)} evidence rows -> {args.out_dir}", flush=True)
    if len(result.summary):
        print(result.summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

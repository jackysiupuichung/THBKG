"""Fast lookup of the evidence rows behind graph edges, from the parsed tables.

Each parsed parquet holds one (source_type, relation, target_type, datasource)
slice of the release, one row per Open Targets evidence record (or per PubMed
reference for IntAct and GO), with its year and score. A graph edge of
relation R between two nodes is the collapse of every such row across all
datasources that map onto R.

Orientation: query edges are matched against the stored forward orientation.
``rev_`` relations are folded back, and a query written against the stored
direction is flipped automatically. Same-type relations (``interacts_with``,
``is_subtype_of``) are looked up in both orientations, since IntAct stores an
interaction under either participant order.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

Triple = tuple[str, str, str]

EVIDENCE_COLUMNS = ["edge_type", "src_type", "relation", "dst_type", "sourceId",
                    "targetId", "datasourceId", "is_static", "score", "year",
                    "evidence_id", "pmids"]


@dataclass(frozen=True)
class ParsedFile:
    path: Path
    triple: Triple
    datasource: str
    is_static: bool
    columns: frozenset


class ParsedEvidenceIndex:
    """Catalogue of parsed evidence parquets keyed by forward triple."""

    def __init__(self, parsed_dirs: list[tuple[Path, bool]]):
        self.files: dict[Triple, list[ParsedFile]] = {}
        for d, is_static in parsed_dirs:
            if not Path(d).is_dir():
                continue
            for path in sorted(Path(d).glob("*.parquet")):
                pf = pq.ParquetFile(path)
                if pf.metadata.num_rows == 0:
                    continue
                head = next(pf.iter_batches(
                    batch_size=1,
                    columns=["source_type", "relation", "target_type", "datasourceId"],
                )).to_pylist()[0]
                triple = (head["source_type"], head["relation"], head["target_type"])
                self.files.setdefault(triple, []).append(ParsedFile(
                    path=path, triple=triple, datasource=str(head["datasourceId"]),
                    is_static=is_static, columns=frozenset(pf.schema_arrow.names)))

    def triples(self) -> list[Triple]:
        return sorted(self.files)

    # ------------------------------------------------------------------
    # Planning: query edge -> concrete (triple, stored source, stored target).
    # ------------------------------------------------------------------
    def plan(self, queries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Expand query edges into stored-orientation lookups.

        Returns (plan, unresolved). ``plan`` has one row per (query row,
        candidate triple, orientation); ``unresolved`` lists query rows whose
        relation and node types match no stored triple.
        """
        plan_rows, unresolved = [], []
        for qi, q in queries.iterrows():
            st, sid, dt, did = q.src_type, q.src_id, q.dst_type, q.dst_id
            rel = q.relation if isinstance(q.relation, str) and q.relation else None
            if rel is not None and rel.startswith("rev_"):
                rel = rel[4:]
                st, sid, dt, did = dt, did, st, sid
            hits = set()
            for triple in self.files:
                ts, tr, td = triple
                if rel is not None and not fnmatchcase(tr, rel):
                    continue
                if (ts, td) == (st, dt):
                    hits.add((triple, sid, did))
                if (ts, td) == (dt, st):
                    hits.add((triple, did, sid))
            if not hits:
                unresolved.append(dict(query_row=qi, reason=(
                    f"no local evidence table for {q.src_type} -[{q.relation or '*'}]- "
                    f"{q.dst_type} in this release")))
            for triple, a, b in sorted(hits):
                plan_rows.append(dict(query_row=qi, triple=triple,
                                      lookup_src=a, lookup_dst=b))
        plan = pd.DataFrame(plan_rows, columns=["query_row", "triple",
                                                "lookup_src", "lookup_dst"])
        return plan, pd.DataFrame(unresolved, columns=["query_row", "reason"])

    # ------------------------------------------------------------------
    # Lookup: one filtered scan per parquet for all edges at once.
    # ------------------------------------------------------------------
    def lookup(self, plan: pd.DataFrame) -> pd.DataFrame:
        """Evidence rows for every planned lookup, joined back to ``query_row``."""
        chunks = []
        for triple, sub in plan.groupby("triple", sort=True):
            keys = sub[["lookup_src", "lookup_dst"]].drop_duplicates()
            for f in self.files.get(triple, []):
                rows = _scan(f, keys)
                if rows.empty:
                    continue
                rows = rows.merge(sub[["query_row", "lookup_src", "lookup_dst"]],
                                  left_on=["sourceId", "targetId"],
                                  right_on=["lookup_src", "lookup_dst"])
                chunks.append(rows.drop(columns=["lookup_src", "lookup_dst"]))
        if not chunks:
            return pd.DataFrame(columns=["query_row"] + EVIDENCE_COLUMNS)
        return pd.concat(chunks, ignore_index=True)


def _scan(f: ParsedFile, keys: pd.DataFrame) -> pd.DataFrame:
    cols = ["sourceId", "targetId", "datasourceId", "score", "year"]
    cols += [c for c in ("id", "literature") if c in f.columns]
    srcs = sorted(set(keys["lookup_src"]))
    dsts = sorted(set(keys["lookup_dst"]))
    flt = pc.field("sourceId").isin(srcs) & pc.field("targetId").isin(dsts)
    table = ds.dataset(f.path, format="parquet").to_table(columns=cols, filter=flt)
    df = table.to_pandas()
    if df.empty:
        return df
    # isin on each column admits cross pairs; keep the exact requested pairs.
    df = df.merge(keys, left_on=["sourceId", "targetId"],
                  right_on=["lookup_src", "lookup_dst"]).drop(
                      columns=["lookup_src", "lookup_dst"])
    st, rel, dt = f.triple
    out = pd.DataFrame({
        "edge_type": "::".join(f.triple),
        "src_type": st, "relation": rel, "dst_type": dt,
        "sourceId": df["sourceId"].astype(str),
        "targetId": df["targetId"].astype(str),
        "datasourceId": df["datasourceId"].astype(str),
        "is_static": f.is_static,
        "score": pd.to_numeric(df["score"], errors="coerce"),
        "year": pd.to_numeric(df["year"], errors="coerce").astype("Int64"),
        "evidence_id": df["id"].astype(object).where(df["id"].notna(), None)
                       if "id" in df else None,
        "pmids": [_as_pmids(v) for v in df["literature"]] if "literature" in df
                 else [[] for _ in range(len(df))],
    })
    return out


def _as_pmids(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v) for v in value if v is not None and str(v)]
    return [str(value)] if str(value) else []

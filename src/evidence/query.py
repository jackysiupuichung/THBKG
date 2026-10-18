"""Turn paths and graphs into one canonical table of query edges.

Every input form is normalised to ``QUERY_COLUMNS``: one row per edge, with
node types and accessions in the order the user traversed them. Relation may
be ``None`` (any relation), an exact relation name, a ``rev_`` relation as
emitted by PyG's ToUndirected, or an fnmatch glob such as ``clinical_trial_*``.
Orientation against the stored evidence is resolved later, in ``parsed.py``.

Supported inputs
----------------
* Accession path strings::

      ENSG00000120217 -- MONDO_0002169
      ENSG00000120217 -[interacts_with]- ENSG00000188389 -[genetic_association]-> MONDO_0002169
      CHEMBL939 -[modulated_by]-> ENSG00000146648 -[clinical_trial_*]-> EFO_0000571

  Connectors are ``--``, ``->``, ``<-``, ``-[rel]-``, ``-[rel]->`` and
  ``<-[rel]-``. Arrow direction is a hint only; a node may carry an explicit
  type prefix (``disease:GO_0000050``) when its ID space is ambiguous.
* Edge lists (CSV, TSV or parquet) with ``src_id`` and ``dst_id`` columns, and
  optionally ``src_type``, ``dst_type``, ``relation``, ``query_id``, ``hop``,
  ``weight``.
* PaGE-Link ``per_pair_paths.parquet`` and per-edge explanation tables
  (``per_pair_edges.parquet``), whose nodes are graph indices resolved through
  a mappings file. A drift check aborts when the indices no longer resolve to
  the explained pair.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .release import NODE_TYPES, NodeMappings, guess_node_type

QUERY_COLUMNS = ["query_id", "hop", "src_type", "src_id", "relation",
                 "dst_type", "dst_id", "weight", "input_edge_type"]

# Longest connectors first so "<-[rel]-" is not read as "<-".
_CONNECTOR = re.compile(r"\s*(<-\[[^\]]*\]-|-\[[^\]]*\]->?|->|<-|--)\s*")
_REL_IN_CONNECTOR = re.compile(r"\[([^\]]*)\]")


def _parse_node(token: str) -> tuple[str, str]:
    token = token.strip()
    if not token:
        raise ValueError("empty node in path")
    head, sep, tail = token.partition(":")
    if sep and head in NODE_TYPES and tail:
        return head, tail
    return guess_node_type(token), token


def _norm_relation(rel) -> str | None:
    if rel is None or (isinstance(rel, float) and pd.isna(rel)):
        return None
    rel = str(rel).strip()
    return None if rel in ("", "*") else rel


def parse_path(text: str, query_id: str) -> pd.DataFrame:
    """One accession path string -> query edges (one row per hop)."""
    parts = _CONNECTOR.split(text.strip())
    if len(parts) < 3 or len(parts) % 2 == 0:
        raise ValueError(f"cannot parse path {text!r}: need NODE (CONNECTOR NODE)+")
    nodes = [_parse_node(t) for t in parts[0::2]]
    rows = []
    for hop, conn in enumerate(parts[1::2]):
        m = _REL_IN_CONNECTOR.search(conn)
        (st, sid), (dt, did) = nodes[hop], nodes[hop + 1]
        rows.append(dict(query_id=query_id, hop=hop, src_type=st, src_id=sid,
                         relation=_norm_relation(m.group(1) if m else None),
                         dst_type=dt, dst_id=did, weight=None,
                         input_edge_type=None))
    return pd.DataFrame(rows, columns=QUERY_COLUMNS)


def read_paths_file(path: str | Path) -> list[str]:
    """One path per line; blank lines and ``#`` comments are skipped."""
    lines = (ln.strip() for ln in Path(path).read_text().splitlines())
    return [ln for ln in lines if ln and not ln.startswith("#")]


def from_path_strings(paths: list[str], prefix: str = "path") -> pd.DataFrame:
    frames = [parse_path(p, f"{prefix}{i + 1}") for i, p in enumerate(paths)]
    return pd.concat(frames, ignore_index=True) if frames else empty_queries()


def _read_table(path: str | Path) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    sep = "\t" if path.endswith((".tsv", ".tab")) else ","
    return pd.read_csv(path, sep=sep, dtype=str)


def from_edge_list(path: str | Path) -> pd.DataFrame:
    """Edge list with accession columns -> query edges."""
    df = _read_table(path)
    missing = {"src_id", "dst_id"} - set(df.columns)
    if missing:
        raise SystemExit(f"[evidence] edge list {path} lacks columns {sorted(missing)}")
    out = pd.DataFrame({
        "query_id": df["query_id"].astype(str) if "query_id" in df else Path(path).stem,
        "hop": pd.to_numeric(df["hop"]).astype(int) if "hop" in df else range(len(df)),
        "src_id": df["src_id"].astype(str).str.strip(),
        "dst_id": df["dst_id"].astype(str).str.strip(),
    })
    for side in ("src", "dst"):
        col = f"{side}_type"
        given = df[col] if col in df else pd.Series([None] * len(df))
        out[col] = [t if isinstance(t, str) and t else guess_node_type(a)
                    for t, a in zip(given, out[f"{side}_id"])]
    out["relation"] = [_norm_relation(r) for r in df["relation"]] if "relation" in df else None
    out["weight"] = pd.to_numeric(df["weight"]) if "weight" in df else None
    out["input_edge_type"] = None
    return out[QUERY_COLUMNS]


# --------------------------------------------------------------------------
# Index-based inputs (PaGE-Link / IG explanation outputs).
# --------------------------------------------------------------------------
_PATH_HEAD = re.compile(r"^([a-z_]+)#(\d+)$")
_PATH_HOP = re.compile(r"^\[([^\]]+)\]\s+([a-z_]+)#(\d+)$")


def _split_edge_type(et: str) -> tuple[str, str, str]:
    st, rel, dt = str(et).split("::", 2)
    return st, rel, dt


def _resolve(mappings: NodeMappings, node_type: str, idx: int, where: str) -> str:
    acc = mappings.accession(node_type, idx)
    if acc is None:
        raise SystemExit(
            f"[evidence] ABORT: {node_type}#{idx} ({where}) has no accession in "
            f"{mappings.path}. The mappings file does not match the graph these "
            "indices came from; pass the mappings that were in force for that run.")
    return acc


def from_pagelink_paths(path: str | Path, mappings: NodeMappings,
                        max_rank: int | None = None) -> pd.DataFrame:
    """PaGE-Link ``per_pair_paths.parquet`` -> query edges, one query per path."""
    df = pd.read_parquet(path)
    if max_rank is not None and "rank" in df:
        df = df[df["rank"] < max_rank]
    rows = []
    for r in df.itertuples(index=False):
        segs = [s.strip() for s in str(r.path).split(" -> ")]
        head = _PATH_HEAD.match(segs[0])
        if head is None:
            raise SystemExit(f"[evidence] cannot parse PaGE-Link path {r.path!r}")
        qid = f"{r.target_id}|{r.disease_id}|rank{getattr(r, 'rank', 0)}"
        prev_t, prev_i = head.group(1), int(head.group(2))
        prev_acc = _resolve(mappings, prev_t, prev_i, qid)
        if (prev_t, prev_acc) != ("target", r.target_id):
            raise SystemExit(
                f"[evidence] ABORT drift: path {qid} starts at {prev_t}#{prev_i} = "
                f"{prev_acc}, expected target {r.target_id}. Wrong mappings file.")
        for hop, seg in enumerate(segs[1:]):
            m = _PATH_HOP.match(seg)
            if m is None:
                raise SystemExit(f"[evidence] cannot parse hop {seg!r} in {r.path!r}")
            et, nt, ni = m.group(1), m.group(2), int(m.group(3))
            acc = _resolve(mappings, nt, ni, qid)
            _, rel, _ = _split_edge_type(et)
            rows.append(dict(query_id=qid, hop=hop, src_type=prev_t, src_id=prev_acc,
                             relation=rel, dst_type=nt, dst_id=acc, weight=None,
                             input_edge_type=et))
            prev_t, prev_acc = nt, acc
        if (prev_t, prev_acc) != ("disease", r.disease_id):
            raise SystemExit(
                f"[evidence] ABORT drift: path {qid} ends at {prev_t} {prev_acc}, "
                f"expected disease {r.disease_id}. Wrong mappings file.")
    return pd.DataFrame(rows, columns=QUERY_COLUMNS) if rows else empty_queries()


def from_explained_edges(path: str | Path, mappings: NodeMappings,
                         top_k: int | None = None,
                         weight_col: str = "ig_total") -> pd.DataFrame:
    """Per-edge explanation table (``per_pair_edges.parquet``) -> query edges.

    One query per explained (target, disease) pair. Edges are ordered by
    |weight| (IG or PaGE-Link mask), and ``top_k`` keeps the strongest ones.
    """
    df = pd.read_parquet(path)
    if weight_col not in df:
        weight_col = next((c for c in ("ig_total", "mask_weight") if c in df), None)
    rows = []
    for (t_id, d_id), g in df.groupby(["target_id", "disease_id"], sort=False):
        qid = f"{t_id}|{d_id}"
        resolved = []
        for r in g.itertuples(index=False):
            st, rel, dt = _split_edge_type(r.edge_type)
            resolved.append((st, rel, dt, _resolve(mappings, st, r.src, qid),
                             _resolve(mappings, dt, r.dst, qid), r))
        # Drift gate on the FULL subgraph, before top-k truncation. The explained
        # pair's own nodes seed its subgraph, so under the right mappings they
        # appear among the resolved nodes; under drifted mappings a seed
        # resolving to exactly the expected accession is vanishingly unlikely.
        # A direct target-disease edge is NOT required (most pairs lack one),
        # and one seed suffices because a seed can be isolated before cutoff.
        nodes = {(st, sa) for st, _, _, sa, _, _ in resolved}
        nodes |= {(dt, da) for _, _, dt, _, da, _ in resolved}
        found = [("target", t_id) in nodes, ("disease", d_id) in nodes]
        if not any(found):
            raise SystemExit(
                f"[evidence] ABORT drift: neither node of explained pair {qid} "
                f"appears in its own subgraph under {mappings.path}.")
        if not all(found):
            print(f"[evidence] note: only one seed of {qid} appears in its "
                  "explained subgraph", flush=True)
        if weight_col:
            def _strength(x):
                v = getattr(x[5], weight_col)
                return abs(float(v)) if pd.notna(v) else 0.0
            resolved.sort(key=_strength, reverse=True)
        if top_k is not None:
            resolved = resolved[:top_k]
        for hop, (st, rel, dt, sa, da, r) in enumerate(resolved):
            rows.append(dict(query_id=qid, hop=hop, src_type=st, src_id=sa,
                             relation=rel, dst_type=dt, dst_id=da,
                             weight=float(getattr(r, weight_col)) if weight_col else None,
                             input_edge_type=r.edge_type))
    return pd.DataFrame(rows, columns=QUERY_COLUMNS) if rows else empty_queries()


def empty_queries() -> pd.DataFrame:
    return pd.DataFrame(columns=QUERY_COLUMNS)

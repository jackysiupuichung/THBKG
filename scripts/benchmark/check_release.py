#!/usr/bin/env python
"""Check a THBKG release directory against the release contract.

The contract (config/release_contract.yaml) is what every release must
satisfy, whatever the Open Targets version: its files, node and edge types,
tensor shapes and dtypes, id formats, time range, score rules and label
consistency. Counts are not in it: they follow the input, which
check_inputs.py checks. Fails (exit 1) on any check.

    python scripts/benchmark/check_release.py --release-dir R --ot-version 26.09 \\
        [--out R/check_release.json] [--require-snapshots]

Checks
  files      every required file exists, and nothing outside the contract's
             file list (a stray __pycache__, a left-over label file)
  meta       format and edge_attr columns; every tensor metadata.json lists is
             in graph.safetensors with that shape and dtype, and no other;
             lengths agree with num_nodes / num_edges; every edge type has a
             score_max
  config     config/event_graph_config.yaml time_range matches
             src/benchmark/tasks.py; the contract's phase weights match the
             build's (apply_trial_resolution.STAGE_SCORE) and the label code's
             (tasks.WEIGHT_PHASE, tasks.PHASE_GRADE); source_versions.json
             names the Open Targets release and every source the contract lists
  nodes      exactly the contract's node types and tensors; finite features;
             distinct embedding rows; mappings/ index 0..n-1 with unique ids
             matching the id pattern; molecule modality rows are zero exactly
             where the modality is absent
  edges      exactly the contract's edge types, each with rows; endpoints in
             range; per kind (see the contract):
               dated   within the time range, the last year reached; score in
                       (0, score_max]; no duplicate (pair, year) rows; a pair's
                       score never falls over time
               static  edge_time -1, score 1, no self-loops, acyclic
               symmetric  each pair stored once as (min, max)
               trial_* scores are phase weights (halved for trial_failure);
                       a start-dated phase weight that is not one is the
                       ongoing-score leak (halved 0.7 = 0.35, 0.5 = 0.25)
             every pair with a trial outcome has a start-dated trial edge no
             later than its first outcome
  labels     discovery / hypothesis / train / eval: ids and indices agree with mappings/,
             one row per pair, label years inside the file's window, train and
             eval disjoint; each file equals the table src/benchmark/tasks.py
             builds from this graph, so stale labels fail
  loader     the release's own load_thbkg.load_graph: every relation but
             interacts_with gets a rev_ type mirroring it; interacts_with
             becomes both directions with nothing summed (edge_time within the
             time range, score within score_max); max-normalised scores in [0, 1]
  snapshots  (when present, or --require-snapshots) index.json years, edge-type
             order and graph hash; per-year edge counts; static rows at -1, no
             dated row in or after its snapshot year, scores > 0

Heavy (reads the whole release and rebuilds the labels): run via sbatch, as
scripts/benchmark/bench_04_assemble.sh and bench_05_snapshots.sh do.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from safetensors import safe_open

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "preprocessing" / "temporal_graph" / "parsers"))
from src.benchmark import tasks  # noqa: E402
from src.benchmark.release import Release, edge_key  # noqa: E402
from apply_trial_resolution import STAGE_SCORE  # noqa: E402

# metadata.json dtype names -> safetensors header dtypes
SAFETENSORS_DTYPE = {"float32": "F32", "float64": "F64", "float16": "F16", "int64": "I64",
                     "int32": "I32", "int16": "I16", "int8": "I8", "uint8": "U8", "bool": "BOOL"}


class Report:
    """Named check results for the JSON report; each is printed PASS or FAIL as it is recorded."""

    def __init__(self):
        """Start with no results and no failures."""
        self.results: dict = {}
        self.failures: list[str] = []

    def check(self, name: str, ok, detail=None) -> bool:
        """Record check `name`: print PASS or FAIL with its detail, store it for the report and note a
        failure. Returns whether it passed."""
        self.results[name] = {"ok": bool(ok), **({"detail": detail} if detail is not None else {})}
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail is not None else ""), flush=True)
        if not ok:
            self.failures.append(name)
        return bool(ok)


def acyclic(ei: np.ndarray, n: int) -> bool:
    """Whether the directed graph with edges `ei` over `n` nodes has no cycle. Kahn's algorithm: repeatedly
    remove the nodes with no incoming edge; all nodes go only if there is no cycle."""
    src, dst = ei[0], ei[1]
    indeg = np.bincount(dst, minlength=n)
    order = np.argsort(src, kind="stable")
    children = dst[order]
    ptr = np.searchsorted(src[order], np.arange(n + 1))
    frontier = np.flatnonzero(indeg == 0)
    removed = 0
    while frontier.size:
        removed += frontier.size
        lens = ptr[frontier + 1] - ptr[frontier]
        idx = np.repeat(ptr[frontier] - np.r_[0, np.cumsum(lens)[:-1]], lens) + np.arange(lens.sum())
        kids = children[idx]
        np.subtract.at(indeg, kids, 1)
        frontier = np.unique(kids[indeg[kids] == 0])
    return removed == n


def edge_label(et, edge_types) -> str:
    """Short name for edge type `et` in check names: its relation, plus the source node type when two edge
    types share the relation (is_subtype_of on disease and on go)."""
    shared = sum(1 for e in edge_types if e[1] == et[1]) > 1
    return f"{et[1]}.{et[0]}" if shared else et[1]


def check_files(rep: Report, rd: Path, contract: dict, require_snapshots: bool) -> None:
    """Files: every file the contract requires is present (snapshots/index.json too with `require_snapshots`),
    and no file lies outside the contract's required and optional lists. huggingface_hub's .cache/ is
    ignored."""
    fc = contract["files"]
    # .cache/ is huggingface_hub's download bookkeeping, not part of the release
    present = sorted(str(p.relative_to(rd)) for p in rd.rglob("*")
                     if p.is_file() and not str(p.relative_to(rd)).startswith(".cache/"))
    required = fc["required"] + (["snapshots/index.json"] if require_snapshots else [])
    missing = [f for f in required if not (rd / f).is_file()]
    allowed = fc["required"] + fc["optional"]
    extra = [p for p in present if not any(fnmatch.fnmatch(p, g) for g in allowed)]
    rep.check("files.required", not missing, {"missing": missing} if missing else f"{len(required)} files")
    rep.check("files.nothing_else", not extra, {"unexpected": extra[:20]} if extra else f"{len(present)} files")


def check_meta(rep: Report, r: Release, contract: dict) -> None:
    """metadata.json against the contract and graph.safetensors: format and edge_attr columns; every tensor it
    lists is stored with the stated shape and dtype, and nothing else is; tensor lengths agree with
    num_nodes and num_edges; every edge type has a score_max."""
    m = r.meta
    rep.check("meta.format", (m.get("format"), str(m.get("format_version"))) == (
        contract["format"], contract["format_version"]), f"{m.get('format')} {m.get('format_version')}")
    rep.check("meta.edge_attr_columns", m.get("edge_attr_columns") == contract["edge_attr_columns"],
              m.get("edge_attr_columns"))

    listed = {spec["key"]: spec for nm in m["node_types"].values() for spec in nm["tensors"].values()}
    listed.update({spec["key"]: spec for e in m["edge_types"] for spec in e["tensors"].values()})
    bad = []
    with safe_open(str(r.path / "graph.safetensors"), framework="np") as f:
        keys = set(f.keys())
        for k, spec in listed.items():
            if k not in keys:
                bad.append(f"{k}: missing")
                continue
            sl = f.get_slice(k)
            shape, dtype = list(sl.get_shape()), sl.get_dtype()
            if shape != spec["shape"] or dtype != SAFETENSORS_DTYPE.get(spec["dtype"]):
                bad.append(f"{k}: {shape} {dtype}, metadata {spec['shape']} {spec['dtype']}")
    extra = sorted(keys - set(listed))
    rep.check("meta.tensors_match_safetensors", not bad and not extra,
              {"mismatched": bad[:10], "unlisted": extra[:10]} if bad or extra else f"{len(listed)} tensors")

    wrong = []
    for nt, nm in m["node_types"].items():
        wrong += [f"node.{nt}.{a}" for a, s in nm["tensors"].items() if s["shape"][0] != nm["num_nodes"]]
    n_cols = len(m.get("edge_attr_columns") or [])
    for e in m["edge_types"]:
        n, ts = e["num_edges"], e["tensors"]
        want = {"edge_index": [2, n], "edge_time": [n], "edge_attr": [n, n_cols]}
        wrong += [f"{e['rel']}.{a}" for a, shape in want.items() if ts.get(a, {}).get("shape") != shape]
    rep.check("meta.lengths_consistent", not wrong, wrong[:10] if wrong else None)

    no_max = [edge_key(et) for et in r.edge_types if not r.edge_meta(et).get("score_max")]
    rep.check("meta.score_max", not no_max, no_max or None)


def check_config(rep: Report, rd: Path, contract: dict, ot_version: str) -> tuple[int, int]:
    """The build's configuration: config/event_graph_config.yaml time_range equals the range
    src/benchmark/tasks.py hard-codes; the contract's phase weights equal those of trial resolution and the
    label code; source_versions.json names `ot_version` and every source the contract lists. Returns the
    time range (first_year, last_year)."""
    tr = yaml.safe_load((_REPO / "config" / "event_graph_config.yaml").read_text())["time_range"]
    first, last = int(tr["first_year"]), int(tr["last_year"])
    rep.check("config.time_range_matches_tasks", (first, last) == (tasks.FIRST_GRAPH_YEAR, tasks.LAST_GRAPH_YEAR),
              f"event_graph_config {first}-{last}, tasks.py {tasks.FIRST_GRAPH_YEAR}-{tasks.LAST_GRAPH_YEAR}")

    weights = sorted(contract["phase_weights"])
    others = {"apply_trial_resolution.STAGE_SCORE": set(STAGE_SCORE.values()),
              "tasks.WEIGHT_PHASE (+ unknown 0.01)": set(tasks.WEIGHT_PHASE) | {0.01},
              "tasks.PHASE_GRADE": set(tasks.PHASE_GRADE)}
    differ = {k: sorted(v) for k, v in others.items() if sorted(v) != weights}
    rep.check("config.phase_weights_agree", not differ, differ or None)

    p = rd / "source_versions.json"
    sv = json.loads(p.read_text()) if p.exists() else {}
    got = (sv.get("open_targets") or {}).get("release")
    rep.check("config.open_targets_release", got == ot_version, f"source_versions.json {got}, expected {ot_version}")
    missing = [k for k in contract["source_versions"] if k not in sv]
    rep.check("config.source_versions_complete", not missing, {"missing": missing} if missing else None)
    return first, last


def check_nodes(rep: Report, r: Release, contract: dict) -> None:
    """Per node type: exactly the contract's node types, each with exactly its tensors (width and dtype);
    finite float features; enough distinct embedding rows where the contract asks; mappings/ with index
    0..n-1 and unique ids matching the id pattern; molecule modality rows zero exactly where the modality
    is absent."""
    spec_all = contract["node_types"]
    rep.check("nodes.types", set(r.node_types) == set(spec_all),
              {"missing": sorted(set(spec_all) - set(r.node_types)),
               "extra": sorted(set(r.node_types) - set(spec_all))} if set(r.node_types) != set(spec_all) else None)
    for nt, spec in spec_all.items():
        if nt not in r.meta["node_types"]:
            continue
        n = r.num_nodes(nt)
        have = {a: (s["shape"], s["dtype"]) for a, s in r.meta["node_types"][nt]["tensors"].items()}
        want = {a: ([n] if w is None else [n, w], d) for a, (w, d) in spec["tensors"].items()}
        rep.check(f"nodes.{nt}.tensors", have == want,
                  None if have == want else {"have": have, "want": want})

        nonfinite = [a for a, (_, d) in have.items() if d.startswith("float")
                     and not np.isfinite(r.node_tensor(nt, a)).all()]
        rep.check(f"nodes.{nt}.finite", not nonfinite, nonfinite or None)

        if "min_distinct_rows" in spec and "x" in have:
            distinct = int(np.unique(np.round(r.node_tensor(nt, "x"), 4), axis=0).shape[0])
            rep.check(f"nodes.{nt}.distinct_rows", distinct >= spec["min_distinct_rows"] * n,
                      f"{distinct:,} of {n:,} rows distinct (need {spec['min_distinct_rows']:.0%})")

        mp = r.mapping(nt)
        index_ok = np.array_equal(np.sort(mp["index"].to_numpy()), np.arange(n))
        ids = mp["node_id"].astype(str)
        bad_ids = ids[~ids.str.fullmatch(spec["id_pattern"])]
        rep.check(f"nodes.{nt}.mapping", index_ok and ids.is_unique and bad_ids.empty,
                  f"{n:,} nodes; index 0..n-1: {index_ok}; unique ids: {ids.is_unique}; "
                  f"{len(bad_ids)} ids not matching {spec['id_pattern']}"
                  + (f", e.g. {bad_ids.head(5).tolist()}" if len(bad_ids) else ""))

        for mod in spec.get("modalities", []):
            if f"x_{mod}" not in have or f"has_{mod}" not in have:
                continue
            x, has = r.node_tensor(nt, f"x_{mod}"), r.node_tensor(nt, f"has_{mod}").astype(bool)
            zero_absent = not x[~has].any()
            empty_present = int((~x[has].any(axis=1)).sum())
            rep.check(f"nodes.{nt}.{mod}_zero_where_absent", zero_absent and empty_present == 0,
                      f"{int(has.sum()):,} of {n:,} have {mod}; absent rows all zero: {zero_absent}; "
                      f"present rows all zero: {empty_present}")


def check_edges(rep: Report, r: Release, contract: dict, first: int, last: int) -> int:
    """Per edge type: exactly the contract's edge types, each with rows and endpoints in range, then the rules
    of its kind (module docstring); across them, that dated edges reach the time range's last year and
    every trial-outcome pair has a start-dated trial edge no later than its first outcome. Returns the last
    dated edge year."""
    kinds = {tuple(e[:3]): e[3] for e in contract["edge_types"]}
    have = set(r.edge_types)
    rep.check("edges.types", have == set(kinds),
              {"missing": sorted(map(edge_key, set(kinds) - have)),
               "extra": sorted(map(edge_key, have - set(kinds)))} if have != set(kinds) else f"{len(have)} types")
    weights = set(np.round(contract["phase_weights"], 3).tolist())
    last_dated = -1
    trial_start, trial_end = [], []
    for et in r.edge_types:
        kind = kinds.get(et)
        if kind is None:
            continue
        ei, t, s = r.edges(et)
        n_src, n_dst = r.num_nodes(et[0]), r.num_nodes(et[2])
        problems = []
        if ei.shape[1] == 0:
            rep.check(f"edges.{edge_label(et, r.edge_types)}", False, "no edges")
            continue
        if ei.min() < 0 or ei[0].max() >= n_src or ei[1].max() >= n_dst:
            problems.append("endpoint index out of range")
        key = ei[0] * n_dst + ei[1]
        if kind == "static":
            if (t != -1).any():
                problems.append(f"{int((t != -1).sum()):,} rows not at edge_time -1")
            if (s != 1).any():
                problems.append(f"{int((s != 1).sum()):,} rows with score != 1")
            if (ei[0] == ei[1]).any():
                problems.append(f"{int((ei[0] == ei[1]).sum()):,} self-loops")
            if len(np.unique(key)) < len(key):
                problems.append(f"{len(key) - len(np.unique(key)):,} duplicate edges")
            if not acyclic(ei, n_src):
                problems.append("has a cycle")
        else:
            outside = (t < first) | (t > last)
            if outside.any():
                problems.append(f"{int(outside.sum()):,} rows dated outside {first}-{last} "
                                f"(min {int(t.min())}, max {int(t.max())})")
            last_dated = max(last_dated, int(t.max()))
            smax = float(r.edge_meta(et).get("score_max") or np.inf)
            if (s <= 0).any():
                problems.append(f"{int((s <= 0).sum()):,} rows with score <= 0")
            if (s > smax * (1 + 1e-6)).any():
                problems.append(f"{int((s > smax * (1 + 1e-6)).sum()):,} rows above score_max {smax:.4f}")
            order = np.lexsort((t, key))
            k, ts, ss = key[order], t[order], s[order]
            same = k[1:] == k[:-1]
            dup = int((same & (ts[1:] == ts[:-1])).sum())
            falls = int((same & (ss[1:] < ss[:-1] - 1e-6)).sum())
            if dup:
                problems.append(f"{dup:,} duplicate (pair, year) rows")
            if falls:
                problems.append(f"{falls:,} rows where a pair's score falls")
        if kind == "symmetric" and (ei[0] >= ei[1]).any():
            problems.append(f"{int((ei[0] >= ei[1]).sum()):,} rows not stored as (min, max)")
        if kind.startswith("trial_"):
            v = np.round(s * (2 if kind == "trial_failure" else 1), 3)
            odd = sorted(set(np.unique(v).tolist()) - weights)
            if odd:
                problems.append(f"scores that are no phase weight{' when doubled' if kind == 'trial_failure' else ''}: "
                                f"{odd[:8]}")
            (trial_start if kind == "trial_start" else trial_end).append(pd.DataFrame({"k": key, "t": t}))
        rep.check(f"edges.{edge_label(et, r.edge_types)}", not problems,
                  "; ".join(problems) if problems else f"{ei.shape[1]:,} rows ({kind})")

    rep.check("edges.reach_last_year", last_dated == last, f"last dated edge {last_dated}, time_range ends {last}")
    if trial_start and trial_end:
        start = pd.concat(trial_start).groupby("k")["t"].min()
        end = pd.concat(trial_end).groupby("k")["t"].min()
        s_at = start.reindex(end.index)
        no_start = int(s_at.isna().sum())
        before = int((end < s_at).sum())
        rep.check("edges.trial_outcome_after_start", no_start == 0 and before == 0,
                  f"{len(end):,} pairs with a trial outcome; {no_start} with no start-dated trial edge; "
                  f"{before} whose first outcome precedes their first start")
    return last_dated


def check_labels(rep: Report, r: Release, first: int, last: int) -> None:
    """The label files against this graph: discovery, hypothesis, train and eval ids and indices agree with
    mappings/, one row per pair, label years inside each file's window, train and eval disjoint, and each
    file equal to the table src/benchmark/tasks.py builds from the graph (so stale labels fail)."""
    rd = r.path
    idx = {nt: pd.Series(np.arange(r.num_nodes(nt)), index=r.node_ids(nt)) for nt in ("target", "disease")}
    rebuilt = {"discovery": tasks.build_discovery_labels(rd), "hypothesis": tasks.build_hypothesis_labels(rd),
               **tasks.build_advancement_labels(rd)}
    windows = {"discovery": ("first_trial_year", first, last),
               "hypothesis": ("first_direct_year", first, last),
               **{k: ("transition_year", lo, hi) for k, (lo, hi, _) in tasks.ADVANCEMENT_LABEL_FILES.items()}}
    pairs = {}
    for name, built in rebuilt.items():
        p = rd / "labels" / f"{name}.parquet"
        if not p.exists():
            continue  # files.required reports it
        df = pd.read_parquet(p)
        ids_ok = df["target_id"].isin(idx["target"].index).all() and df["disease_id"].isin(idx["disease"].index).all()
        index_ok = ids_ok and all(
            np.array_equal(idx[nt].loc[df[f"{nt}_id"]].to_numpy(), df[f"{nt}_index"].to_numpy())
            for nt in ("target", "disease"))
        dups = int(df.duplicated(["target_id", "disease_id"]).sum())
        col, lo, hi = windows[name]
        outside = int((~df[col].between(lo, hi)).sum())
        rep.check(f"labels.{name}.consistent", ids_ok and index_ok and dups == 0 and outside == 0,
                  f"{len(df):,} pairs; ids in mappings: {ids_ok}; indices agree: {index_ok}; "
                  f"{dups} duplicate pairs; {outside} with {col} outside {lo}-{hi}")
        try:
            pd.testing.assert_frame_equal(df.reset_index(drop=True), built.reset_index(drop=True), check_dtype=False)
            same, detail = True, "equals the table rebuilt from this graph"
        except AssertionError as e:
            same, detail = False, " ".join(str(e).split())[:300]
        rep.check(f"labels.{name}.matches_graph", same, detail)
        pairs[name] = pd.MultiIndex.from_frame(df[["target_id", "disease_id"]])
    if "train" in pairs and "eval" in pairs:
        both = len(pairs["train"].intersection(pairs["eval"]))
        rep.check("labels.train_eval_disjoint", both == 0, f"{both} pairs in both")


def check_loader(rep: Report, r: Release, contract: dict, last: int) -> None:
    """The release's own load_thbkg.py, as a user runs it: the undirected graph has a rev_ type mirroring
    every relation but interacts_with; interacts_with holds both directions with nothing summed (edge_time
    within the time range, score within score_max); max-normalised scores lie in [0, 1]."""
    spec = importlib.util.spec_from_file_location("release_load_thbkg", r.path / "load_thbkg.py")
    mod = importlib.util.module_from_spec(spec)
    # importing would leave __pycache__/ in the release (how one reached the v26.09 upload)
    write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = write_bytecode
    kinds = {tuple(e[:3]): e[3] for e in contract["edge_types"]}
    sym = {et for et in r.edge_types if kinds.get(et) == "symmetric"}

    g = mod.load_graph(r.path, to_undirected=True)
    want = set(r.edge_types) | {(d, f"rev_{rel}", s) for (s, rel, d) in r.edge_types if (s, rel, d) not in sym}
    rep.check("loader.edge_types", set(g.edge_types) == want,
              None if set(g.edge_types) == want else {"missing": sorted(map(edge_key, want - set(g.edge_types))),
                                                      "extra": sorted(map(edge_key, set(g.edge_types) - want))})
    problems = []
    for et in r.edge_types:
        ei, t, s = r.edges(et)
        if et in sym:
            st = g[et]
            tmax, smax = int(st.edge_time.max()), float(st.edge_attr[:, 0].max())
            limit = float(r.edge_meta(et)["score_max"])
            if st.edge_index.size(1) != 2 * ei.shape[1] or tmax > last or smax > limit * (1 + 1e-6):
                problems.append(f"{edge_key(et)}: {st.edge_index.size(1):,} rows for {ei.shape[1]:,} stored, "
                                f"max edge_time {tmax}, max score {smax:.4f} (score_max {limit:.4f})")
        else:
            rev = (et[2], f"rev_{et[1]}", et[0])
            if rev not in g.edge_types:
                continue  # loader.edge_types reports it
            if not (torch.equal(g[rev].edge_index, torch.from_numpy(ei).flip(0))
                    and torch.equal(g[rev].edge_time, torch.from_numpy(t))):
                problems.append(f"{edge_key(rev)} does not mirror {edge_key(et)}")
    rep.check("loader.reverse_edges", not problems, problems or f"{len(want)} edge types")
    del g

    gm = mod.load_graph(r.path, to_undirected=False, normalise="max")
    over = [edge_key(et) for et in gm.edge_types
            if float(gm[et].edge_attr[:, 0].min()) < 0 or float(gm[et].edge_attr[:, 0].max()) > 1 + 1e-6]
    rep.check("loader.max_normalised_in_unit_interval", not over, over or None)


def check_snapshots(rep: Report, r: Release, contract: dict, last_dated: int, required: bool) -> None:
    """The per-year snapshots, when built (a failure if missing with `required`): years, edge-type order and
    source graph hash in index.json; for every year and edge type, the edge count index.json records,
    static rows at -1, no dated row from that year or later, and scores above 0."""
    p = r.path / "snapshots" / "index.json"
    if not p.exists():
        if required:
            rep.check("snapshots.present", False, "no snapshots/index.json")
        else:
            print("SKIP  snapshots (none built yet)")
        return
    index = json.loads(p.read_text())
    years = index["years"]
    want_years = list(range(contract["first_snapshot_year"], last_dated + 2))
    rep.check("snapshots.years", years == want_years, f"{years[0]}-{years[-1]} ({len(years)}), "
              f"expected {want_years[0]}-{want_years[-1]}")
    rep.check("snapshots.edge_type_order", index.get("edge_types") == [edge_key(et) for et in r.edge_types])
    h = hashlib.sha256()
    with open(r.path / "graph.safetensors", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    rep.check("snapshots.source_graph", index.get("source_graph_sha256") == h.hexdigest(),
              "built from this graph.safetensors" if index.get("source_graph_sha256") == h.hexdigest()
              else "built from a different graph.safetensors")
    kinds = {tuple(e[:3]): e[3] for e in contract["edge_types"]}
    static = {i for i, et in enumerate(r.edge_types) if kinds.get(et) == "static"}
    problems = []
    for y in years:
        counts = index["edges_per_year"][str(y)]
        with safe_open(str(r.path / "snapshots" / f"{y}.safetensors"), framework="np") as f:
            for i, et in enumerate(r.edge_types):
                t, s = f.get_tensor(f"edge.{i}.edge_time"), f.get_tensor(f"edge.{i}.score")
                n = f.get_slice(f"edge.{i}.edge_index").get_shape()[1]
                if n != counts[str(i)] or len(t) != n or len(s) != n:
                    problems.append(f"{y} {edge_key(et)}: {n:,} edges, index.json {counts[str(i)]:,}")
                if i in static and (t != -1).any():
                    problems.append(f"{y} {edge_key(et)}: static rows not at -1")
                if i not in static and (t >= y).any():
                    problems.append(f"{y} {edge_key(et)}: {int((t >= y).sum()):,} rows dated {y} or later")
                if (s <= 0).any():
                    problems.append(f"{y} {edge_key(et)}: {int((s <= 0).sum()):,} scores <= 0")
    rep.check("snapshots.content", not problems, problems[:10] or f"{len(years)} snapshots")


def main() -> int:
    """Run every check group on --release-dir, write the JSON report and return 1 if any check failed (0
    otherwise)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-dir", required=True, type=Path)
    ap.add_argument("--ot-version", required=True, help="Open Targets release the build used, e.g. 26.09")
    ap.add_argument("--contract", type=Path, default=_REPO / "config" / "release_contract.yaml")
    ap.add_argument("--out", type=Path, default=None, help="JSON report (default <release>/check_release.json)")
    ap.add_argument("--require-snapshots", action="store_true", help="fail if snapshots/ is missing")
    args = ap.parse_args()

    contract = yaml.safe_load(args.contract.read_text())
    rd = args.release_dir
    rep = Report()
    r = Release(rd)

    def run(group, fn, *a, default=None):
        """Run one check group. If it raises (e.g. label code rejecting a malformed graph), record that as one
        failed check and return `default`, so the other groups still run."""
        try:
            return fn(*a)
        except Exception as e:
            rep.check(f"{group}.ran", False, f"{type(e).__name__}: {e}"[:400])
            return default

    run("files", check_files, rep, rd, contract, args.require_snapshots)
    run("meta", check_meta, rep, r, contract)
    first, last = run("config", check_config, rep, rd, contract, args.ot_version,
                      default=(tasks.FIRST_GRAPH_YEAR, tasks.LAST_GRAPH_YEAR))
    run("nodes", check_nodes, rep, r, contract)
    last_dated = run("edges", check_edges, rep, r, contract, first, last, default=last)
    run("labels", check_labels, rep, r, first, last)
    run("loader", check_loader, rep, r, contract, last)
    run("snapshots", check_snapshots, rep, r, contract, last_dated, args.require_snapshots)

    out = args.out or rd / "check_release.json"
    rep.results["failures"] = rep.failures
    out.write_text(json.dumps(rep.results, indent=2, default=str))
    print(f"\n{'❌ ' + str(len(rep.failures)) + ' check(s) failed' if rep.failures else '✅ all checks passed'}"
          f" -> {out}")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())

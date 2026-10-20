#!/usr/bin/env python
"""Build the discontinuous THBKG: one graph snapshot per decision year.

The continuous graph (graph.safetensors) stores one row per (pair, relation) for
every year its score changed. Snapshot ``Y`` holds, for every pair, its latest
row dated strictly before ``Y`` -- the graph a model may see when deciding in
year ``Y`` -- with two values per edge:

* ``score``: that row's score (raw; normalise at load time with the edge type's
  ``score_max`` from metadata.json, see src/benchmark/snapshots.py);
* ``recency``: how recently the score rose, as of ``Y - 1``:

      recency = max over the pair's score increases (year p, size d) with
                0 <= (Y - 1) - p <= window  of  d / (1 + exp(scale * ((Y - 1 - p) - shift)))

  with the Open Targets novelty constants of build_event_list.py (scale 2,
  shift 3, window 5). Increases come from the cumulative score itself (it never
  decreases), so trial edges get recency too. Static edges: recency 0.

``edge_time`` is the year of the row (year of the pair's last update before
``Y``), -1 for static edges.

Output (``--out-dir``, default <release>/snapshots):
  {Y}.safetensors   per edge type i (metadata.json order):
                    edge.{i}.edge_index int32 [2, E], edge.{i}.score float32 [E],
                    edge.{i}.recency float32 [E], edge.{i}.edge_time int16 [E]
  index.json        years, per-year edge counts, definitions, source graph hash

Read-only on the release graph. Heavy: run via scripts/benchmark/bench_05_snapshots.sh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open
from safetensors.numpy import save_file

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "preprocessing" / "temporal_graph" / "pipeline"))
from build_event_list import NOVELTY_SCALE, NOVELTY_SHIFT, NOVELTY_WINDOW  # noqa: E402

STATIC_T = np.iinfo(np.int64).min
NEVER = np.iinfo(np.int64).max
DECAY = (1.0 / (1.0 + np.exp(NOVELTY_SCALE * (np.arange(NOVELTY_WINDOW + 1) - NOVELTY_SHIFT)))).astype(np.float64)


class Relation:
    """One edge type's rows, sorted by (pair, year), with per-row increments."""

    def __init__(self, ei: np.ndarray, t: np.ndarray, s: np.ndarray):
        self.static = bool((t == STATIC_T).all())
        if not self.static and (t == STATIC_T).any():
            raise ValueError("edge type mixes static and dated rows")
        if self.static:
            self.src, self.dst, self.s = ei[0], ei[1], s.astype(np.float32)
            return
        # sort by (src, dst, t, score); keep the highest score of any (pair, year) duplicate
        order = np.lexsort((s, t, ei[1], ei[0]))
        src, dst, t, s = ei[0][order], ei[1][order], t[order], s[order].astype(np.float64)
        dup = np.r_[(src[1:] == src[:-1]) & (dst[1:] == dst[:-1]) & (t[1:] == t[:-1]), False]
        self.n_dup = int(dup.sum())
        keep = ~dup
        src, dst, t, s = src[keep], dst[keep], t[keep], s[keep]

        new_pair = np.r_[True, (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])]
        last = np.r_[new_pair[1:], True]
        delta = s - np.r_[0.0, s[:-1]]
        delta[new_pair] = s[new_pair]
        self.min_delta = float(delta.min())
        if self.min_delta < -1e-6:
            raise ValueError(f"score decreases within a pair (min increment {self.min_delta})")
        t_next = np.r_[t[1:], NEVER]
        t_next[last] = NEVER

        self.src, self.dst, self.t, self.s = src, dst, t, s.astype(np.float32)
        self.delta = delta.clip(min=0.0)
        self.t_next = t_next
        self.pair_id = np.cumsum(new_pair) - 1
        self.n_pairs = int(self.pair_id[-1]) + 1 if len(src) else 0

    def snapshot(self, year: int) -> dict[str, np.ndarray]:
        if self.static:
            n = len(self.src)
            return {"edge_index": np.stack([self.src, self.dst]).astype(np.int32),
                    "score": self.s, "recency": np.zeros(n, np.float32),
                    "edge_time": np.full(n, -1, np.int16)}
        latest = (self.t < year) & (self.t_next >= year)
        win = np.nonzero((self.t < year) & (self.t >= year - 1 - NOVELTY_WINDOW))[0]
        rec_pair = np.zeros(max(self.n_pairs, 1), np.float64)
        if win.size:
            contrib = self.delta[win] * DECAY[(year - 1 - self.t[win])]
            pid = self.pair_id[win]
            starts = np.r_[0, np.nonzero(np.diff(pid))[0] + 1]
            rec_pair[pid[starts]] = np.maximum.reduceat(contrib, starts)
        idx = np.nonzero(latest)[0]
        return {"edge_index": np.stack([self.src[idx], self.dst[idx]]).astype(np.int32),
                "score": self.s[idx],
                "recency": rec_pair[self.pair_id[idx]].astype(np.float32),
                "edge_time": self.t[idx].astype(np.int16)}


def brute_force_check(raw: dict, rel: Relation, snap: dict, year: int, rng, n: int = 2000) -> None:
    """Recompute a sample of pairs from the unsorted rows and compare."""
    ei, t, s = raw["ei"], raw["t"], raw["s"]
    if rel.static:
        assert snap["edge_index"].shape[1] == ei.shape[1]
        return
    df = pd.DataFrame({"a": ei[0], "b": ei[1], "t": t, "s": s.astype(np.float64)})
    before = df[df.t < year]
    # every pair seen before Y appears exactly once, and nothing dated >= Y
    n_pairs_before = before.groupby(["a", "b"]).ngroups
    assert snap["edge_index"].shape[1] == n_pairs_before, (snap["edge_index"].shape[1], n_pairs_before)
    if n_pairs_before == 0:
        return
    assert int(snap["edge_time"].max()) < year
    got = pd.DataFrame({"a": snap["edge_index"][0], "b": snap["edge_index"][1], "t": snap["edge_time"],
                        "s": snap["score"], "r": snap["recency"]})
    assert not got.duplicated(["a", "b"]).any()
    pairs = got.sample(min(n, len(got)), random_state=int(rng.integers(1 << 31)))
    sub = df.merge(pairs[["a", "b"]], on=["a", "b"])
    for (a, b), g in sub.groupby(["a", "b"]):
        g = g.sort_values(["t", "s"]).drop_duplicates("t", keep="last")
        inc = g.s.diff().fillna(g.s).clip(lower=0).to_numpy()
        gb = g[g.t < year]
        exp_t, exp_s = int(gb.t.iloc[-1]), float(gb.s.iloc[-1])
        ages = (year - 1 - g.t.to_numpy())
        ok = (ages >= 0) & (ages <= NOVELTY_WINDOW)
        exp_r = float((inc[ok] * DECAY[ages[ok]]).max()) if ok.any() else 0.0
        row = pairs[(pairs.a == a) & (pairs.b == b)].iloc[0]
        assert int(row.t) == exp_t and abs(float(row.s) - exp_s) < 1e-6 and abs(float(row.r) - exp_r) < 1e-5, \
            (a, b, year, row.to_dict(), exp_t, exp_s, exp_r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--years", default=None,
                    help="'1995-2026' or '2005,2016' (default: 1995 to the last edge year + 1)")
    ap.add_argument("--check-years", default="2005,2016,last",
                    help="years to brute-force check on a sample of pairs ('' to skip)")
    args = ap.parse_args()

    rd = args.release_dir
    out = args.out_dir or rd / "snapshots"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((rd / "metadata.json").read_text())

    t0 = time.time()
    raw, rels, last_year = {}, {}, 0
    with safe_open(str(rd / "graph.safetensors"), framework="np") as f:
        for i, e in enumerate(meta["edge_types"]):
            ei = f.get_tensor(e["tensors"]["edge_index"]["key"]).astype(np.int64)
            t = f.get_tensor(e["tensors"]["edge_time"]["key"]).astype(np.int64)
            s = f.get_tensor(e["tensors"]["edge_attr"]["key"])[:, 0]
            raw[i] = {"ei": ei, "t": t, "s": s}
            rels[i] = Relation(ei, t, s)
            if not rels[i].static:
                last_year = max(last_year, int(t.max()))
            r = rels[i]
            print(f"  {e['src']}|{e['rel']}|{e['dst']}: {ei.shape[1]:,} rows"
                  + ("" if r.static else f", {r.n_pairs:,} pairs, {r.n_dup} same-year duplicates dropped"), flush=True)
    print(f"loaded in {time.time() - t0:.0f}s; last edge year {last_year}")

    if args.years:
        if "-" in args.years:
            lo, hi = map(int, args.years.split("-"))
            years = list(range(lo, hi + 1))
        else:
            years = [int(y) for y in args.years.split(",")]
    else:
        years = list(range(1995, last_year + 2))
    check = {last_year + 1 if y == "last" else int(y) for y in args.check_years.split(",") if y}
    rng = np.random.default_rng(0)

    index = {
        "format": "thbkg-snapshots-safetensors", "format_version": "1.0",
        "years": years,
        "definition": {
            "snapshot": "for decision year Y: each pair's latest continuous-graph row with edge_time < Y",
            "score": "raw score of that row; normalise with metadata.json edge_types[i].score_max",
            "recency": (f"max over the pair's score increases d in year p with 0 <= Y-1-p <= {NOVELTY_WINDOW} of "
                        f"d / (1 + exp({NOVELTY_SCALE} * ((Y-1-p) - {NOVELTY_SHIFT}))); 0 for static edges"),
            "edge_time": "year of that row; -1 for static edges",
        },
        "keys": "edge.{i}.{edge_index,score,recency,edge_time}, i = metadata.json edge_types order",
        "source_graph_sha256": hashlib.sha256((rd / "graph.safetensors").read_bytes()).hexdigest(),
        "edges_per_year": {},
    }
    for y in years:
        tensors, counts = {}, {}
        for i, rel in rels.items():
            snap = rel.snapshot(y)
            if y in check:
                brute_force_check(raw[i], rel, snap, y, rng)
            for k, v in snap.items():
                tensors[f"edge.{i}.{k}"] = np.ascontiguousarray(v)
            counts[i] = int(snap["edge_index"].shape[1])
        save_file(tensors, str(out / f"{y}.safetensors"))
        index["edges_per_year"][str(y)] = counts
        print(f"  {y}: {sum(counts.values()):>11,} edges" + ("  (checked)" if y in check else ""), flush=True)

    (out / "index.json").write_text(json.dumps(index, indent=2))
    total = sum(sum(c.values()) for c in index["edges_per_year"].values())
    print(f"✅ {len(years)} snapshots, {total:,} edges -> {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

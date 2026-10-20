#!/usr/bin/env python
"""Write each relation's theoretical maximum score into a release's metadata.json.

Scores are on different scales per relation by construction (build_event_list.py):

* evidence relations: harmonic sum over datasources of (datasource weight x
  datasource score), where a datasource score is the harmonic sum of its top-50
  raw scores; raw scores are <= 1, so the maximum depends only on the
  datasource weights (config/datatype_mapping.yaml);
* clinical_trial_* relations: max of the raw trial scores, so at most 1;
* static ontology relations: constant 1.

The maximum is computed from that formula and the configured weights, never from
the data, so dividing by it is the same constant for every decision year and
leaks nothing about later evidence. Normalisation itself happens at load time
(src/benchmark/snapshots.py); this only records the constants.

Which datasources feed a relation is read from the raw edge file names
({src}_{relation}_{dst}_{datasource}.parquet), not from their contents.

    python scripts/benchmark/write_edge_scales.py --release-dir R --raw-edges-dir E [--dry-run]

Fails if any observed score exceeds its computed maximum.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from safetensors import safe_open

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "preprocessing" / "temporal_graph" / "pipeline"))
from build_event_list import harmonic_sum  # noqa: E402  the function that builds the scores

TOP_K = 50  # harmonic_sum keeps the top 50 scores


def _clean(s: str) -> str:
    """The pipeline's file-name normalisation (parsers/parser.py EdgeParser.output_name)."""
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9_]", "_", s).lower()).strip("_")


def relation_datasources(raw_edges_dir: Path, edge_types: list[dict]) -> dict[str, list[str]]:
    """{rel key -> datasources} from file names, matching the longest {src}_{rel}_{dst}_ prefix."""
    files = sorted(p.stem for p in raw_edges_dir.glob("*.parquet"))
    prefixes = sorted(((_clean(f"{e['src']}_{e['rel']}_{e['dst']}") + "_", f"{e['src']}|{e['rel']}|{e['dst']}")
                       for e in edge_types), key=lambda x: -len(x[0]))
    out: dict[str, list[str]] = {}
    for f in files:
        for prefix, key in prefixes:
            if f.startswith(prefix):
                out.setdefault(key, []).append(f[len(prefix):])
                break
    return out


def theoretical_max(datasources: list[str], ds_map: dict) -> float:
    """Max over datatype groups of the weighted harmonic sum at full datasource scores."""
    ds_max = harmonic_sum(np.ones(TOP_K))
    groups: dict[str, list[float]] = {}
    for ds in datasources:
        info = ds_map.get(ds, {"datatype": ds, "weight": 1.0})  # unmapped: weight 1 (add_datatype_info)
        groups.setdefault(info["datatype"], []).append(info["weight"] * ds_max)
    return max(harmonic_sum(np.array(w)) for w in groups.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-dir", required=True, type=Path)
    ap.add_argument("--raw-edges-dir", required=True, type=Path)
    ap.add_argument("--datatype-mapping", type=Path, default=_REPO / "config" / "datatype_mapping.yaml")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write metadata.json")
    args = ap.parse_args()

    meta_path = args.release_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    cfg = yaml.safe_load(args.datatype_mapping.read_text())
    ds_map = {ds: {"datatype": dt, "weight": float(w)}
              for dt, info in cfg["datatypes"].items() for ds, w in info["datasources"].items()}
    rel_ds = relation_datasources(args.raw_edges_dir, meta["edge_types"])

    bad = []
    with safe_open(str(args.release_dir / "graph.safetensors"), framework="np") as f:
        for e in meta["edge_types"]:
            key = f"{e['src']}|{e['rel']}|{e['dst']}"
            t = f.get_tensor(e["tensors"]["edge_time"]["key"])
            static = bool((t == np.iinfo(np.int64).min).all())
            if static:
                kind, smax, sources = "static", 1.0, []
            elif e["rel"].startswith("clinical_trial"):
                kind, smax, sources = "trial_max", 1.0, rel_ds.get(key, [])
            else:
                sources = rel_ds.get(key, [])
                if not sources:
                    raise SystemExit(f"no raw edge files found for {key} in {args.raw_edges_dir}")
                kind, smax = "harmonic", float(theoretical_max(sources, ds_map))
            observed = float(f.get_tensor(e["tensors"]["edge_attr"]["key"])[:, 0].max())
            e["score_max"] = round(smax, 6)
            e["score_kind"] = kind
            e["score_sources"] = sources
            flag = "" if observed <= smax + 1e-6 else "  <-- EXCEEDS"
            if flag:
                bad.append(key)
            print(f"{key:56s} {kind:9s} max {smax:.4f}  observed {observed:.4f}  {','.join(sources)}{flag}")

    if bad:
        raise SystemExit(f"observed score above theoretical max for: {bad}")
    meta["score_normalisation"] = (
        "divide edge_attr score (and snapshot recency) by the edge type's score_max: a constant of the "
        "aggregation formula and config/datatype_mapping.yaml weights, independent of the data")
    if args.dry_run:
        print("dry run: metadata.json not written")
    else:
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"✅ score_max written to {meta_path}")


if __name__ == "__main__":
    main()

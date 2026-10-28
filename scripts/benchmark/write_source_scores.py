#!/usr/bin/env python
"""Write per-source target-disease evidence scores into a release directory.

The graph stores one target->disease edge per data type (genetic association,
literature, ...), combining that type's sources. Baselines that need the
sources apart (the Open Targets association score, per-source features) read
this table instead:

  scores/target_disease_by_source.parquet
      target_id, disease_id, relation, datasource, year, score

One row per (pair, source) for every year the source's score changed: score is
the source's harmonic sum over its top-50 evidence dated up to that year, as in
the graph build; a pair's score before decision year Y is its latest row with
year < Y. Trial evidence is left out (it is in the graph as trial edges).

    python scripts/benchmark/write_source_scores.py \\
        --events $EVENT_OUTPUT_DIR/events_datasource.parquet --release-dir $RELEASE_DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True, type=Path, help="per-datasource event list of the graph build")
    ap.add_argument("--release-dir", required=True, type=Path)
    args = ap.parse_args()

    ev = pq.read_table(args.events, columns=["sourceId", "targetId", "source_type", "target_type", "relation",
                                             "datasourceId", "edge_weight", "edge_time"]).to_pandas()
    ev = ev[(ev["source_type"] == "target") & (ev["target_type"] == "disease")
            & ~ev["relation"].str.startswith("clinical_trial")]
    out = ev.rename(columns={"sourceId": "target_id", "targetId": "disease_id", "datasourceId": "datasource",
                             "edge_weight": "score", "edge_time": "year"})[
        ["target_id", "disease_id", "relation", "datasource", "year", "score"]]
    out = out.astype({"year": "int16", "score": "float32"}).sort_values(
        ["target_id", "disease_id", "datasource", "year"], ignore_index=True)
    dest = args.release_dir / "scores" / "target_disease_by_source.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    print(f"{dest}: {len(out):,} rows, {out['datasource'].nunique()} sources, "
          f"{out[['target_id', 'disease_id']].drop_duplicates().shape[0]:,} pairs")


if __name__ == "__main__":
    main()

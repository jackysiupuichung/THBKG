#!/usr/bin/env python3
"""Provenance check: is each context edge's `edge_time` the evidence's OWN year,
or a constant / a copy of the advancement transition_year?

Evidence for per-evidence dating:
  - different edge types span DIFFERENT, sensible year ranges
  - no edge type is a single constant
  - context edge years are NOT all equal to the advancement transition_year
Read-only; loads the graph and prints per-edge-type edge_time stats.
"""
import torch
from collections import Counter

G = "/gpfs/scratch/bty414/opentarget_evidences/26.03/graph/hetero_graph_with_features_datatype.pt"
data = torch.load(G, weights_only=False)

ADV = ("target", "advancement", "disease")

print("=== per-edge-type edge_time stats ===")
print(f"{'edge_type':45s} {'n_edges':>9} {'min':>6} {'max':>6} {'n_years':>7} {'top-3 years (count)'}")
for et in data.edge_types:
    store = data[et]
    if "edge_time" not in store:
        print(f"{str(et):45s} {'--- no edge_time ---'}")
        continue
    t = store.edge_time
    yrs = t.tolist()
    c = Counter(yrs)
    top = ", ".join(f"{y}:{n}" for y, n in c.most_common(3))
    print(f"{str(et):45s} {len(yrs):>9} {int(t.min()):>6} {int(t.max()):>6} {len(c):>7}  {top}")

# Is context edge_time ever just a copy of the advancement transition_year?
adv_years = set(data[ADV].edge_time.tolist())
print(f"\nadvancement (transition_year) range: {min(adv_years)}..{max(adv_years)}, n_distinct={len(adv_years)}")
print("\n=== do context edge types have years OUTSIDE the advancement year range? ===")
amin, amax = min(adv_years), max(adv_years)
for et in data.edge_types:
    if et == ADV or "edge_time" not in data[et]:
        continue
    t = data[et].edge_time
    outside = int(((t < amin) | (t > amax)).sum())
    print(f"{str(et):45s} {outside:>8} / {len(t)} edges dated outside [{amin},{amax}]")

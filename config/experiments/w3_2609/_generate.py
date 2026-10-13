#!/usr/bin/env python3
"""Generate the OT 26.09 w3 configs by cloning every config/experiments/w3_retrain
YAML (12 model families x 5 seeds) and changing ONLY:

- graph_file / mappings_file -> the 26.09 build (scripts/assembling_graph_04.sh,
  building_event_graph_02.sh outputs; the build is already w3-labelled, so the
  file names carry no _w3 suffix)
- advancement_train_csv / advancement_test_csv -> the 26.09_w3 label CSVs, set
  explicitly so the loader never falls back to the 26.03_w3 defaults (disease
  IDs differ between releases)
- experiment name and output_dir -> *_w3_2609_* under 26.09/runs/w3/

Recipe (model, optimiser, lambdarank, seeds) is untouched.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "config" / "experiments" / "w3_retrain"
OUT = REPO / "config" / "experiments" / "w3_2609"

_DR = "${oc.env:THBKG_DATA_ROOT,/gpfs/scratch/bty414/opentarget_evidences}"
GRAPH = f"{_DR}/26.09/graph/hetero_graph_with_features_datatype.pt"
MAPPINGS = f"{_DR}/26.09/progression/temporal_graph_datatype_mappings.pt"
ADV_DIR = "data/clinical_trial_advancement/26.09_w3"


def convert(text: str) -> str:
    text, n = re.subn(r"(\n  graph_file: ).*", rf"\g<1>{GRAPH}", text)
    assert n == 1, "graph_file"
    text, n = re.subn(
        r"(\n  mappings_file: ).*",
        rf"\g<1>{MAPPINGS}\n  advancement_train_csv: {ADV_DIR}/train_dataset.csv"
        rf"\n  advancement_test_csv: {ADV_DIR}/test_dataset.csv",
        text,
    )
    assert n == 1, "mappings_file"
    text, n = re.subn(r"(experiment:\n\s*name: \S*?)_w3_", r"\g<1>_w3_2609_", text)
    assert n == 1, "experiment name"
    text, n = re.subn(
        r"(output_dir: .*?)/26\.03/runs/w3_retrain/(.*?)_w3_",
        r"\g<1>/26.09/runs/w3/\g<2>_w3_2609_",
        text,
    )
    assert n == 1, "output_dir"
    assert "26.03" not in text, "leftover 26.03 reference"
    return text


n = 0
for src in sorted(SRC.rglob("*.yaml")):
    rel = src.relative_to(SRC)
    dst = OUT / rel.parent / src.name.replace("_w3_", "_w3_2609_")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(convert(src.read_text()))
    n += 1
print(f"wrote {n} configs under {OUT}")

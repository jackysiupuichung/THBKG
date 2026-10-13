#!/usr/bin/env python3
"""Export the w3 clinical-advancement pipeline labels as train/test CSVs
for build_event_graph.py (columns: target_id, disease_id, transition_year, outcome).

train_dataset.csv  <- datasets_<OT_VERSION>_w<ADV_WINDOW>/training_dataset.zarr
test_dataset.csv   <- datasets_<OT_VERSION>_w<ADV_WINDOW>/evaluation_dataset.zarr

OT_VERSION (default 26.09) and ADV_WINDOW (default 3) come from the
environment; output goes to this repo's data/clinical_trial_advancement/.
"""
import os
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

OT_VERSION = os.environ.get("OT_VERSION", "26.09")
ADV_WINDOW = os.environ.get("ADV_WINDOW", "3")
TAG = f"{OT_VERSION}_w{ADV_WINDOW}"
SRC = Path(f"/gpfs/scratch/bty414/clinical_advancement_paper/data/datasets_{TAG}")
OUT = Path(__file__).resolve().parents[2] / "data" / "clinical_trial_advancement" / TAG
OUT.mkdir(parents=True, exist_ok=True)


def export(zarr_name, out_csv):
    ds = xr.open_zarr(SRC / zarr_name).load()
    tid = [str(x) for x in ds.coords["target_id"].values]
    did = [str(x) for x in ds.coords["disease_id"].values]
    outcome = np.asarray(ds.outcome.values).astype(float).ravel().astype(bool)
    # transition_year lives in the descriptor coord
    dd = list(ds.coords["descriptors"].values)
    ty = np.asarray(ds.descriptor.values)[:, dd.index("transition_year")].astype(int)
    df = pd.DataFrame({
        "target_id": tid,
        "disease_id": did,
        "transition_year": ty,
        "outcome": outcome,
    })
    df.to_csv(out_csv, index=False)
    print(f"{out_csv.name}: {len(df)} rows, pos={int(outcome.sum())} "
          f"({outcome.mean()*100:.1f}%), years {ty.min()}-{ty.max()}")


export("training_dataset.zarr", OUT / "train_dataset.csv")
export("evaluation_dataset.zarr", OUT / "test_dataset.csv")
print(f"\nWrote to {OUT}")

"""Result records for the THBKG benchmark: one per run of a setting (model, task, recipe, seed), as TGB keeps them.

    save_result("results.json", task="discovery", model="my_gnn", recipe="single", seed=0,
                val=val_summary, test=test_summary, train_seconds=812)
    summarise(load_results("results.json"))   # mean, std and seeds of the primary metric per setting

The protocol (tasks.py) asks for ``SEEDS`` runs of the setting chosen on val,
each refitted on everything before the test cutoff, reported as mean ± std of
the task's ``PRIMARY_METRIC``. A record keeps the run's val and test metrics
whole, so other metrics can be summarised later: T1/T2 ``evaluate_task``'s
summary; T3 the "summary" table of ``evaluate_advancement`` on the
year-stratified predictions (its val: the mean over ``cv_folds``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .tasks import LABEL_YEAR, PRIMARY_METRIC, RECIPES

SEEDS = 5


def primary(task: str, metrics) -> float:
    """The task's ``PRIMARY_METRIC`` out of one split's metrics (a summary dict, or T3's summary table)."""
    if task in LABEL_YEAR:
        return float(metrics[PRIMARY_METRIC[task]])
    df = pd.DataFrame(metrics)
    row = df[(df["scope"] == "TA mean") & (df["N"] == 10)]
    if len(row) != 1:
        raise ValueError("T3 metrics need the evaluate_advancement summary with scope 'TA mean' and N 10")
    return float(row["RS"].iloc[0])


def _plain(metrics):
    if isinstance(metrics, pd.DataFrame):
        return metrics.to_dict(orient="records")
    return metrics


def load_results(path) -> list[dict]:
    """Every record in a results file (none if it does not exist)."""
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else []


def save_result(path, *, task: str, model: str, recipe: str | None, seed: int, val, test=None, **extra) -> dict:
    """Append one run's record to a JSON results file and return it.

    ``recipe`` is one of ``RECIPES`` for T1/T2 and None for T3 (each pair has its
    own cutoff). ``test`` is None for a run scored on val only. ``extra`` holds
    anything else worth keeping: timings, the release version, hyperparameters.
    """
    if task in LABEL_YEAR and recipe not in RECIPES:
        raise ValueError(f"{task} results need a recipe, one of {sorted(RECIPES)}")
    if task not in LABEL_YEAR and recipe is not None:
        raise ValueError(f"{task} has no recipe")
    record = {"task": task, "model": model, "recipe": recipe, "seed": seed,
              "primary_metric": PRIMARY_METRIC[task], "val_primary": primary(task, val),
              "test_primary": None if test is None else primary(task, test),
              "val": _plain(val), "test": _plain(test), **extra}
    records = load_results(path) + [record]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(records, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    return record


def summarise(records: list[dict]) -> pd.DataFrame:
    """Mean, std and number of seeds of the primary metric per setting (task, model, recipe), on val and test;
    ``complete`` marks settings with at least ``SEEDS`` seeds."""
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["recipe"] = df["recipe"].fillna("-")
    out = (df.groupby(["task", "model", "recipe"])
           .agg(seeds=("seed", "nunique"), val_mean=("val_primary", "mean"), val_std=("val_primary", "std"),
                test_mean=("test_primary", "mean"), test_std=("test_primary", "std"))
           .reset_index())
    out["complete"] = out["seeds"] >= SEEDS
    return out

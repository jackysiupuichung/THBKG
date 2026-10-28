"""Baselines computed from the release without training.

Both read the per-source target-disease scores (scores/target_disease_by_source.parquet)
as of a decision year Y (each (pair, source)'s latest row dated before Y):

* ``open_targets_score``: Open Targets' direct association score. Each
  source's score is multiplied by its weight (config/datatype_mapping.yaml:
  literature, RNA expression and animal models 0.2), the weighted scores are
  sorted and harmonic-summed (the i-th counts 1/i^2) and divided by pi^2/6.
  Trial evidence is not part of the table, so it never contributes.
* ``genetics_score``: the same over the genetic-association sources alone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parents[2]
HARMONIC_MAX = np.pi ** 2 / 6


def source_weights(mapping_file: str | Path = _REPO / "config" / "datatype_mapping.yaml") -> tuple[dict, set]:
    """{datasource: weight} and the set of genetic-association sources."""
    cfg = yaml.safe_load(Path(mapping_file).read_text())["datatypes"]
    weights = {ds: float(w) for dt in cfg.values() for ds, w in (dt.get("datasources") or {}).items()}
    genetic = set((cfg["genetic_association"].get("datasources") or {}))
    return weights, genetic


class SourceScores:
    """Per-source target-disease scores, readable as of any decision year."""

    def __init__(self, release_dir: str | Path):
        release_dir = Path(release_dir)
        df = pd.read_parquet(release_dir / "scores" / "target_disease_by_source.parquet")
        t2i = _index(release_dir, "target")
        d2i = _index(release_dir, "disease")
        t, d = df["target_id"].map(t2i), df["disease_id"].map(d2i)
        if t.isna().any() or d.isna().any():
            raise ValueError("per-source scores reference nodes absent from the graph")
        self.sources = sorted(df["datasource"].unique())
        ds = df["datasource"].map({s: i for i, s in enumerate(self.sources)}).to_numpy(np.int64)
        self.t = t.to_numpy(np.int64)
        self.d = d.to_numpy(np.int64)
        self.ds = ds
        self.year = df["year"].to_numpy(np.int64)
        self.score = df["score"].to_numpy(np.float64)
        n_d, n_s = int(self.d.max()) + 1, len(self.sources)
        self.key = (self.t * n_d + self.d) * n_s + self.ds
        order = np.lexsort((self.year, self.key))
        for a in ("t", "d", "ds", "year", "score", "key"):
            setattr(self, a, getattr(self, a)[order])

    def as_of(self, year: int) -> pd.DataFrame:
        """Each (pair, source)'s latest score dated before ``year``."""
        m = self.year < year
        key = self.key[m]
        last = np.r_[key[1:] != key[:-1], True] if key.size else np.zeros(0, bool)
        idx = np.flatnonzero(m)[last]
        return pd.DataFrame({"target_index": self.t[idx], "disease_index": self.d[idx],
                             "datasource": np.asarray(self.sources)[self.ds[idx]], "score": self.score[idx]})


def _index(release_dir: Path, node_type: str) -> dict:
    df = pd.read_parquet(release_dir / "mappings" / f"{node_type}.parquet")
    return dict(zip(df["node_id"].astype(str), df["index"].astype(int)))


def harmonic_score(per_source: pd.DataFrame, weights: dict, sources: set | None = None) -> pd.DataFrame:
    """Weighted harmonic sum per (target, disease) over ``sources`` (default: all)."""
    df = per_source if sources is None else per_source[per_source["datasource"].isin(sources)]
    unknown = sorted(set(df["datasource"]) - set(weights))
    if unknown:
        raise ValueError(f"sources without a weight in datatype_mapping.yaml: {unknown}")
    w = df["score"].to_numpy() * df["datasource"].map(weights).to_numpy()
    out = pd.DataFrame({"target_index": df["target_index"].to_numpy(), "disease_index": df["disease_index"].to_numpy(),
                        "w": w}).sort_values(["target_index", "disease_index", "w"], ascending=[True, True, False])
    rank = out.groupby(["target_index", "disease_index"]).cumcount().to_numpy() + 1
    out["contrib"] = out["w"].to_numpy() / rank ** 2
    agg = out.groupby(["target_index", "disease_index"], as_index=False)["contrib"].sum()
    return agg.assign(score=agg["contrib"] / HARMONIC_MAX).drop(columns="contrib")


def open_targets_score(scores: SourceScores, year: int, weights: dict) -> pd.DataFrame:
    return harmonic_score(scores.as_of(year), weights)


def genetics_score(scores: SourceScores, year: int, weights: dict, genetic: set) -> pd.DataFrame:
    return harmonic_score(scores.as_of(year), weights, genetic)

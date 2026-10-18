"""Human-readable node names from the downloaded release (local only)."""

from __future__ import annotations

from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds

# node type -> (directory under evidenceDated/, id column, name column)
_SOURCES = {
    "target": ("targets", "id", "approvedSymbol"),
    "disease": ("diseases", "id", "name"),
    "molecule": ("molecule", "id", "name"),
    "go": ("go", "id", "label"),
    "reactome": ("reactome", "id", "label"),
}


def lookup_names(raw_dir: str | Path, ids_by_type: dict[str, set[str]]
                 ) -> dict[tuple[str, str], str]:
    """{(node_type, accession): name} for the requested nodes."""
    out: dict[tuple[str, str], str] = {}
    for node_type, ids in ids_by_type.items():
        spec = _SOURCES.get(node_type)
        if spec is None or not ids:
            continue
        sub, id_col, name_col = spec
        d = Path(raw_dir) / sub
        if not d.is_dir() or not any(d.rglob("*.parquet")):
            continue
        dset = ds.dataset(d, format="parquet")
        if not {id_col, name_col} <= set(dset.schema.names):
            continue
        table = dset.to_table(
            filter=pc.field(id_col).isin(sorted(ids)), columns=[id_col, name_col])
        for acc, name in zip(table[id_col].to_pylist(), table[name_col].to_pylist()):
            if name:
                out[(node_type, acc)] = str(name)
    return out

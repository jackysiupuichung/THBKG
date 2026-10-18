"""Locate one THBKG release's local evidence files and its node mappings.

Everything the evidence extractor reads lives under ``<data_root>/<release>/``,
as produced by ``scripts/collecting_edges_01.sh``:

    evidences/edges/          parsed per-(relation, datasource) evidence rows
    evidences/static_edges/   ontology hierarchy edges (no evidence records)
    evidenceDated/            the downloaded Open Targets release (full records)
    progression/              node mappings (graph index <-> accession)

No network access is used anywhere in this package: results are pinned to the
locally downloaded release.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

NODE_TYPES = ("target", "disease", "molecule", "go", "reactome")
DEFAULT_RELEASE = "26.09"


def resolve_data_root(data_root: str | None) -> Path:
    """``--data-root`` wins, then ``$THBKG_DATA_ROOT``; otherwise fail loudly."""
    root = data_root or os.environ.get("THBKG_DATA_ROOT")
    if not root:
        raise SystemExit(
            "[evidence] no data root: pass --data-root or set THBKG_DATA_ROOT to "
            "the directory that holds <release>/evidences and <release>/evidenceDated")
    return Path(root)


def guess_node_type(accession: str) -> str:
    """Node type from the accession's ID space.

    The five node types use disjoint ID spaces in THBKG. Note the disease
    ontology also contains ``GO_`` (underscore) terms, which are diseases;
    only ``GO:`` (colon) is a GO term node.
    """
    if accession.startswith("ENSG"):
        return "target"
    if accession.startswith("CHEMBL"):
        return "molecule"
    if accession.startswith("GO:"):
        return "go"
    if accession.startswith("R-"):
        return "reactome"
    return "disease"


@dataclass(frozen=True)
class ReleasePaths:
    data_root: Path
    release: str = DEFAULT_RELEASE

    @property
    def base(self) -> Path:
        return Path(self.data_root) / self.release

    @property
    def parsed_edges_dir(self) -> Path:
        return self.base / "evidences" / "edges"

    @property
    def static_edges_dir(self) -> Path:
        return self.base / "evidences" / "static_edges"

    @property
    def raw_dir(self) -> Path:
        return self.base / "evidenceDated"

    @property
    def nodes_dir(self) -> Path:
        return self.base / "evidences" / "nodes"

    @property
    def default_mappings_file(self) -> Path:
        # The 26.09 training configs use the datatype-level mappings; the
        # datasource-level file has an identical node_mapping.
        return self.base / "progression" / "temporal_graph_datatype_mappings.pt"

    def check(self) -> None:
        missing = [p for p in (self.parsed_edges_dir, self.raw_dir) if not p.is_dir()]
        if missing:
            raise SystemExit(
                "[evidence] release files not found: "
                + ", ".join(str(p) for p in missing))


class NodeMappings:
    """Graph index <-> accession, per node type, from a ``*_mappings.pt`` file."""

    def __init__(self, mappings_file: str | Path):
        import torch  # only index-based inputs need torch

        node_mapping = torch.load(mappings_file, weights_only=False)["node_mapping"]
        self.path = str(mappings_file)
        self.acc_to_idx = {nt: dict(m) for nt, m in node_mapping.items()}
        self.idx_to_acc = {nt: {int(v): k for k, v in m.items()}
                           for nt, m in node_mapping.items()}

    def accession(self, node_type: str, idx: int) -> str | None:
        return self.idx_to_acc.get(node_type, {}).get(int(idx))

    def contains(self, node_type: str, accession: str) -> bool:
        return accession in self.acc_to_idx.get(node_type, {})

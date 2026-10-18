"""Extract the Open Targets evidence behind THBKG paths and subgraphs.

Command line: ``python -m src.evidence.extract --help``. Python::

    from src.evidence import EvidenceExtractor, ReleasePaths
    from src.evidence.query import from_path_strings

    ex = EvidenceExtractor(ReleasePaths(data_root, "26.09"))
    res = ex.extract(from_path_strings(["ENSG00000120217 -- MONDO_0002169"]), as_of=2018)
"""

__all__ = ["EvidenceExtractor", "EvidenceResult", "ReleasePaths"]


def __getattr__(name):
    # Lazy, so ``python -m src.evidence.extract`` does not import the module twice.
    if name in ("EvidenceExtractor", "EvidenceResult"):
        from . import extract
        return getattr(extract, name)
    if name == "ReleasePaths":
        from .release import ReleasePaths
        return ReleasePaths
    raise AttributeError(name)

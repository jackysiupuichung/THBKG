#!/usr/bin/env python
"""Write croissant.json into a release directory, with the sha256 of every file it lists.

    python scripts/hf_release/write_croissant.py --release-dir $RELEASE_DIR

The repository's croissant.json is the template: descriptions, record sets and
file paths relative to the release root. Croissant needs a checksum on every
FileObject, and those change with each build, so they are filled in here, once
every listed file is in place (stage 4, after the labels and scores). FileSets
(mappings/, snapshots/) carry no checksum. scripts/benchmark/check_release.py
checks the result against the files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    """Hex sha256 of a file, read in 16 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-dir", required=True, type=Path)
    ap.add_argument("--template", type=Path, default=_REPO_ROOT / "croissant.json")
    args = ap.parse_args()
    c = json.loads(args.template.read_text())
    missing = []
    for d in c["distribution"]:
        if d["@type"] != "cr:FileObject":
            continue
        p = args.release_dir / d["contentUrl"]
        if not p.is_file():
            missing.append(d["contentUrl"])
            continue
        d["sha256"] = sha256(p)
    if missing:
        raise SystemExit(f"croissant.json lists files the release lacks: {missing}")
    out = args.release_dir / "croissant.json"
    out.write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n")
    n = sum(d["@type"] == "cr:FileObject" for d in c["distribution"])
    print(f"✅ {out}: {n} files with sha256, version {c.get('version')}")


if __name__ == "__main__":
    main()

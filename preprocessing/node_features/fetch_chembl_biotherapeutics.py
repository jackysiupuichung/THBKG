#!/usr/bin/env python3
"""Fetch structure inputs for every trialled drug from the ChEMBL REST API.

Open Targets' molecule table carries a SMILES string but no biologic sequence,
so antibodies, proteins and oligonucleotides have nothing to featurise. ChEMBL
stores their chains as ``biotherapeutic.biocomponents``; this script caches
them, together with ChEMBL's own SMILES and HELM, for every drug that appears
in the clinical_precedence evidence.

Needs outbound internet, so run it on a login node (compute nodes are offline):

    python preprocessing/node_features/fetch_chembl_biotherapeutics.py \
        --clinical-precedence-dir $BASE/evidenceDated/sourceId=clinical_precedence \
        --output $OUT/features/raw/chembl_biotherapeutics.parquet

Output is one row per (molecule, biocomponent); a molecule with no components
appears once with null component fields. A ``.json`` sidecar records the
ChEMBL release the API served, for source_versions.json.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

API = "https://www.ebi.ac.uk/chembl/api/data"
FIELDS = "molecule_chembl_id,molecule_type,biotherapeutic,helm_notation,molecule_structures"
CHUNK = 40


def _get_json(url: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # network blips: back off, then give up loudly
            if attempt == retries - 1:
                raise RuntimeError(f"ChEMBL request failed after {retries} tries: {url}") from e
            time.sleep(2 ** attempt)


def load_drug_ids(clin_prec_dir: Path) -> list[str]:
    files = sorted(clin_prec_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {clin_prec_dir}")
    ids = set()
    for f in files:
        ids.update(pd.read_parquet(f, columns=["drugId"])["drugId"].dropna().astype(str))
    return sorted(ids)


def fetch(ids: list[str]) -> pd.DataFrame:
    rows, seen = [], set()
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i : i + CHUNK]
        q = urllib.parse.urlencode(
            {"molecule_chembl_id__in": ",".join(chunk), "only": FIELDS, "limit": 100}
        )
        for m in _get_json(f"{API}/molecule.json?{q}")["molecules"]:
            mid = m["molecule_chembl_id"]
            seen.add(mid)
            bio = m.get("biotherapeutic") or {}
            base = {
                "molecule_chembl_id": mid,
                "molecule_type": m.get("molecule_type"),
                "canonical_smiles": (m.get("molecule_structures") or {}).get("canonical_smiles"),
                "helm_notation": bio.get("helm_notation") or m.get("helm_notation"),
            }
            comps = bio.get("biocomponents") or []
            if not comps:
                rows.append({**base, "component_type": None, "component_description": None, "sequence": None})
            for c in comps:
                rows.append({
                    **base,
                    "component_type": c.get("component_type"),
                    "component_description": c.get("description"),
                    "sequence": c.get("sequence"),
                })
        print(f"  fetched {min(i + CHUNK, len(ids)):,}/{len(ids):,}", flush=True)
        time.sleep(0.3)  # stay well under the EBI rate limit

    missing = sorted(set(ids) - seen)
    if missing:
        print(f"⚠️  {len(missing)} ids not returned by ChEMBL (kept as empty rows): {missing[:10]}")
        rows.extend({"molecule_chembl_id": mid} for mid in missing)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clinical-precedence-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    status = _get_json(f"{API}/status.json")
    ids = load_drug_ids(args.clinical_precedence_dir)
    print(f"ChEMBL {status['chembl_db_version']} ({status['chembl_release_date']}); {len(ids):,} drugs")

    df = fetch(ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    per_mol = df.groupby("molecule_chembl_id").agg(
        has_seq=("sequence", lambda s: s.notna().any()),
        has_smiles=("canonical_smiles", lambda s: s.notna().any()),
        has_helm=("helm_notation", lambda s: s.notna().any()),
    )
    summary = {
        "chembl_db_version": status["chembl_db_version"],
        "chembl_release_date": status["chembl_release_date"],
        "fetched_on": date.today().isoformat(),
        "n_drugs": len(ids),
        "n_with_sequence": int(per_mol["has_seq"].sum()),
        "n_with_smiles": int(per_mol["has_smiles"].sum()),
        "n_with_helm": int(per_mol["has_helm"].sum()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"✅ {len(df):,} rows -> {args.output}")


if __name__ == "__main__":
    main()

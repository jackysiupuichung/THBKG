#!/usr/bin/env python3
"""Molecule input features for every drug type, one tensor per modality.

Morgan fingerprints only exist for molecules with a structure RDKit can parse,
which leaves antibodies, proteins and oligonucleotides out. Each molecule node
therefore gets per-modality tensors, row-aligned with the node ids and never
concatenated:

* ``x_morgan`` [N, 1024]  Morgan fingerprint (same generator and settings as
                          molecule_structure.py). Structure source, in order: Open
                          Targets canonicalSmiles, ChEMBL canonical_smiles, ChEMBL
                          HELM via RDKit for drugs with no protein sequence.
* ``x_esm``    [N, 64]    ESM-2 residue embeddings, mean-pooled over every protein
                          chain the drug has in ChEMBL (a two-chain antibody
                          averages heavy + light by residue count), then reduced
                          by PCA over the biologics. Nucleic-acid chains are not
                          embedded. The full 1,280-d vectors go to a side file.
* ``has_morgan``, ``has_esm`` [N] bool: which nodes have each modality (rows of
                          nodes without it are zero and never reach the model).
* ``drug_type`` [N] long: Open Targets drugType index (DRUG_TYPES).

Models project each modality with its own layer into one shared hidden space and
add a learned drug-type embedding (src/models/node_input.py); drugs with no
structure (cells, plasma products, ~14% of trialled drugs) get the drug-type
embedding alone.

Writes ``molecule_modalities.pt``, ``molecule_esm2_full.safetensors`` (+ ``_ids.parquet``)
and ``molecule_features.yaml`` (shapes, PCA and coverage). GPU recommended for ESM;
run via scripts/collecting_node_features_03.sh or scripts/benchmark/.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")

# Open Targets 26.09 drugType values; anything else falls into "Other".
DRUG_TYPES = [
    "Small molecule", "Antibody", "Antibody drug conjugate", "Protein", "Enzyme",
    "Oligonucleotide", "Oligosaccharide", "Gene", "Cell", "Vaccine component",
    "Unknown", "Other",
]

THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}
_THREE_LETTER = re.compile(r"^(H-)?([A-Z][a-z]{1,2}-)+[A-Z][a-z]{1,2}(-OH|-NH2)?$")


def clean_protein_sequence(seq: str) -> str:
    """Reduce a ChEMBL protein sequence to ESM residue letters.

    ChEMBL mixes plain one-letter chains with annotated ones: crosslink digits
    (``C1``), chain breaks (``||``), terminal groups (``H-...-OH``) and the odd
    three-letter peptide (``H-Asp-Arg-Val-...-OH``). Annotations are dropped,
    unknown three-letter codes become X, and J (not in the ESM vocabulary)
    becomes X.
    """
    s = seq.strip()
    if _THREE_LETTER.match(s):
        core = re.sub(r"^H-|-(OH|NH2)$", "", s)
        return "".join(THREE_TO_ONE.get(tok, "X") for tok in core.split("-"))
    s = re.sub(r"^H-|-(OH|NH2)$", "", s)
    s = re.sub(r"[^A-Z]", "", s)  # drops digits, punctuation, spaces, lowercase notes
    return s.replace("J", "X")


def mol_to_morgan(mol, fp_dim: int, radius: int) -> np.ndarray | None:
    if mol is None:
        return None
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_dim)
    arr = np.zeros((fp_dim,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
    return arr


def _valid_str(v) -> bool:
    return isinstance(v, str) and v.strip() != "" and v.strip().lower() not in {"none", "null", "nan"}


def morgan_for(ot_smiles, chembl_smiles, helm, fp_dim, radius):
    """Return (fingerprint, source) using the first structure RDKit can parse."""
    for src, val, parse in (
        ("ot_smiles", ot_smiles, Chem.MolFromSmiles),
        ("chembl_smiles", chembl_smiles, Chem.MolFromSmiles),
        ("chembl_helm", helm, Chem.MolFromHELM),
    ):
        if _valid_str(val):
            fp = mol_to_morgan(parse(val.strip()), fp_dim, radius)
            if fp is not None:
                return fp, src
    return None, None


@torch.no_grad()
def esm_embed_chains(chains: list[str], model_name: str, device: str,
                     max_residues: int = 1022, batch_size: int = 8) -> list[tuple[np.ndarray, int]]:
    """Mean residue embedding per chain, as (vector, n_residues).

    Chains longer than ESM-2's 1022-residue window are split into windows and
    combined by residue count, so every residue weighs the same.
    """
    from transformers import AutoTokenizer, EsmModel

    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, add_pooling_layer=False).to(device).eval()

    windows, owner = [], []
    for ci, seq in enumerate(chains):
        for start in range(0, len(seq), max_residues):
            windows.append(seq[start : start + max_residues])
            owner.append(ci)

    dim = model.config.hidden_size
    sums = np.zeros((len(chains), dim), dtype=np.float64)
    counts = np.zeros(len(chains), dtype=np.int64)
    order = np.argsort([len(w) for w in windows])  # length-sorted batches pad less
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        batch = tok([windows[i] for i in idx], return_tensors="pt", padding=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        hidden = model(**batch).last_hidden_state  # [B, L+2, D]
        # residue positions only: drop <cls>, <eos> and padding
        mask = batch["attention_mask"].clone()
        mask[:, 0] = 0
        mask[torch.arange(mask.size(0)), batch["attention_mask"].sum(1) - 1] = 0
        res_sum = (hidden * mask.unsqueeze(-1)).sum(1).double().cpu().numpy()
        n_res = mask.sum(1).cpu().numpy()
        for j, wi in enumerate(idx):
            sums[owner[wi]] += res_sum[j]
            counts[owner[wi]] += n_res[j]
        print(f"   ESM windows {min(b + batch_size, len(order)):,}/{len(order):,}", flush=True)

    return [((sums[i] / counts[i]).astype(np.float32), int(counts[i])) for i in range(len(chains))]


def build(args: argparse.Namespace) -> None:
    kg_ids = pd.read_parquet(args.kg_ids_file)["id"].astype(str).tolist()
    print(f"Molecule nodes: {len(kg_ids):,}")

    ot = pd.concat(
        [pd.read_parquet(p, columns=["id", "canonicalSmiles", "drugType"])
         for p in sorted(Path(args.drug_dir).glob(args.parquet_glob))],
        ignore_index=True,
    ).drop_duplicates("id").set_index("id")

    chembl = pd.read_parquet(args.chembl_cache)
    per_mol = chembl.groupby("molecule_chembl_id").agg(
        chembl_smiles=("canonical_smiles", "first"), helm=("helm_notation", "first")
    )
    prot = chembl[(chembl["component_type"] == "Protein") & chembl["sequence"].notna()].copy()
    prot["clean"] = prot["sequence"].map(clean_protein_sequence)
    prot = prot[prot["clean"].str.len() > 0]
    prot = prot[prot["molecule_chembl_id"].isin(set(kg_ids))]

    # --- ESM block: one embedding per protein chain, then residue-weighted per drug
    esm_by_drug: dict[str, np.ndarray] = {}
    if len(prot):
        device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
        print(f"ESM-2 ({args.esm_model}) on {device}: {len(prot):,} chains, "
              f"{prot['molecule_chembl_id'].nunique():,} drugs")
        chain_vecs = esm_embed_chains(prot["clean"].tolist(), args.esm_model, device,
                                      batch_size=args.batch_size)
        prot["_vec"] = [v for v, _ in chain_vecs]
        prot["_n"] = [n for _, n in chain_vecs]
        for mid, g in prot.groupby("molecule_chembl_id"):
            w = g["_n"].to_numpy(dtype=np.float64)
            esm_by_drug[mid] = (np.stack(g["_vec"].to_list()) * w[:, None]).sum(0) / w.sum()
    # --- per-molecule inputs, kept per modality (never concatenated)
    n = len(kg_ids)
    x_morgan = np.zeros((n, args.fp_dim), dtype=np.float32)
    has_morgan = np.zeros(n, dtype=bool)
    has_esm = np.array([mid in esm_by_drug for mid in kg_ids], dtype=bool)
    drug_type = np.zeros(n, dtype=np.int64)
    morgan_src: dict[str, int] = {}
    type_counts: dict[str, dict[str, int]] = {}
    for r, mid in enumerate(kg_ids):
        ot_row = ot.loc[mid] if mid in ot.index else None
        cm = per_mol.loc[mid] if mid in per_mol.index else None
        # HELM only stands in for drugs without a protein sequence (peptides with
        # non-natural monomers). For a full antibody RDKit parses the HELM, but
        # the fingerprint saturates: ~265 bits set and Tanimoto 0.97-1.0 between
        # unrelated antibodies, i.e. no signal beyond drugType.
        fp, src = morgan_for(
            ot_row["canonicalSmiles"] if ot_row is not None else None,
            cm["chembl_smiles"] if cm is not None else None,
            cm["helm"] if cm is not None and mid not in esm_by_drug else None,
            args.fp_dim, args.radius,
        )
        if fp is not None:
            x_morgan[r], has_morgan[r] = fp, True
            morgan_src[src] = morgan_src.get(src, 0) + 1
        dtype = ot_row["drugType"] if ot_row is not None and _valid_str(ot_row["drugType"]) else "Unknown"
        drug_type[r] = DRUG_TYPES.index(dtype) if dtype in DRUG_TYPES else DRUG_TYPES.index("Other")
        c = type_counts.setdefault(DRUG_TYPES[drug_type[r]], {"n": 0, "morgan": 0, "esm": 0, "neither": 0})
        c["n"] += 1
        c["morgan"] += int(has_morgan[r])
        c["esm"] += int(has_esm[r])
        c["neither"] += int(not has_morgan[r] and not has_esm[r])

    # ESM-2: 1,280 dense, highly correlated values per biologic (8 principal
    # components carry 90% of their variance; raw antibody vectors have median
    # cosine 0.997). Reduce to the top components of a PCA over the biologics
    # (sequences only: no labels or dates), without per-component whitening, which
    # amplifies the low-variance tail; one global scale makes the values O(1).
    esm_full = (np.stack([esm_by_drug[m] for m, h in zip(kg_ids, has_esm) if h]).astype(np.float64)
                if has_esm.any() else np.zeros((0, args.esm_dim)))
    k = min(args.esm_pca_dim, max(len(esm_full) - 1, 1))
    x_esm = np.zeros((n, k), dtype=np.float32)
    pca_meta = {"components": 0}
    if len(esm_full) > 1:
        mu = esm_full.mean(0)
        _, sv, vt = np.linalg.svd(esm_full - mu, full_matrices=False)
        proj = (esm_full - mu) @ vt[:k].T
        scale = float(proj.std())
        x_esm[has_esm] = (proj / scale).astype(np.float32)
        var = sv ** 2 / (sv ** 2).sum()
        pca_meta = {"components": k, "fit_on": int(len(esm_full)), "variance_kept": round(float(var[:k].sum()), 4),
                    "global_scale": scale, "whitened": False}

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stale = out / "molecule_features.pt"  # concatenated format of earlier builds
    if stale.exists():
        stale.unlink()
    torch.save({
        "node_id": list(kg_ids),
        "x_morgan": torch.from_numpy(x_morgan),
        "x_esm": torch.from_numpy(x_esm),
        "has_morgan": torch.from_numpy(has_morgan),
        "has_esm": torch.from_numpy(has_esm),
        "drug_type": torch.from_numpy(drug_type),
    }, out / "molecule_modalities.pt")
    # full-width ESM-2 embeddings (pickle-free), for the release side file and other reductions
    from safetensors.numpy import save_file
    save_file({"esm2": np.ascontiguousarray(esm_full.astype(np.float32))}, str(out / "molecule_esm2_full.safetensors"),
              metadata={"model": args.esm_model, "pooling": "residue mean over all protein chains"})
    pd.DataFrame({"node_id": [m for m, h in zip(kg_ids, has_esm) if h]}).to_parquet(
        out / "molecule_esm2_full_ids.parquet", index=False)

    meta = {
        "format": "per-modality tensors, row-aligned with node_id (no concatenation)",
        "tensors": {"x_morgan": [n, args.fp_dim], "x_esm": [n, k], "has_morgan": [n], "has_esm": [n],
                    "drug_type": [n]},
        "drug_types": DRUG_TYPES,
        "morgan": {"radius": args.radius, "fp_dim": args.fp_dim, "n": int(has_morgan.sum()),
                   "source_counts": morgan_src},
        "esm": {"model": args.esm_model, "pooling": "residue mean over all protein chains",
                "n": int(has_esm.sum()), "n_chains": int(len(prot)), "pca": pca_meta,
                "full_width_files": ["molecule_esm2_full.safetensors", "molecule_esm2_full_ids.parquet"]},
        "chembl_cache": str(args.chembl_cache),
        "num_molecules": n,
        "coverage_by_drug_type": type_counts,
    }
    (out / "molecule_features.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

    print(f"\n✅ {n:,} molecules -> {out / 'molecule_modalities.pt'}: x_morgan [{n}, {args.fp_dim}] "
          f"({int(has_morgan.sum()):,} present), x_esm [{n}, {k}] ({int(has_esm.sum()):,} present, "
          f"PCA keeps {pca_meta.get('variance_kept', 0):.1%})")
    print(f"   {'drugType':26s} {'n':>5s} {'morgan':>7s} {'esm':>5s} {'neither':>8s}")
    for t, c in sorted(type_counts.items(), key=lambda kv: -kv[1]["n"]):
        print(f"   {t:26s} {c['n']:5d} {c['morgan']:7d} {c['esm']:5d} {c['neither']:8d}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg-ids-file", required=True, help="parquet with an 'id' column of molecule node ids")
    ap.add_argument("--drug-dir", required=True, help="Open Targets molecule parquet directory")
    ap.add_argument("--parquet-glob", default="*.parquet")
    ap.add_argument("--chembl-cache", required=True, help="output of fetch_chembl_biotherapeutics.py")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--fp-dim", type=int, default=1024)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--esm-dim", type=int, default=1280, help="only used if no drug has a sequence")
    ap.add_argument("--esm-pca-dim", type=int, default=64, help="principal components kept of the ESM-2 embeddings")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    return ap.parse_args()


if __name__ == "__main__":
    build(parse_args())

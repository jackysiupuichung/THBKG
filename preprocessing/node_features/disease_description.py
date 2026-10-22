#!/usr/bin/env python3
"""Disease node features: PubMedBERT embedding of each disease's name and description.

Input text is "<name>. <description>", or the name alone for the ~10% of diseases
with no description (every disease has a name). The embedding is the mean of the
last hidden layer over real tokens (padding excluded) at the model's full width
(768 for PubMedBERT); the models' per-type input layer projects it to the hidden
size.

Earlier builds sliced the embedding to its first 256 coordinates (transformer
dimensions are not ordered by importance), averaged padding into the mean (so an
embedding depended on the longest text in its batch), and embedded a missing
description as the literal string "None" (4,411 diseases in 26.09, collapsing to
11 distinct vectors).
"""

from pathlib import Path
from typing import Dict, List
import argparse
import yaml

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel


# -------------------------------------------------
# PARQUET INGESTION
# -------------------------------------------------
def _clean_text(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    v = str(v).strip()
    return "" if v.lower() in {"none", "nan", "null"} else v


def load_disease_parquets(
    disease_dir: str,
    parquet_glob: str,
    id_col: str,
    text_col: str,
    kg_ids: list = None,
    name_col: str = "name",
) -> pd.DataFrame:
    """One row per disease with ``text`` = "<name>. <description>" (name alone if no description)."""
    parquet_files = sorted(Path(disease_dir).glob(parquet_glob))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {disease_dir} matching {parquet_glob}"
        )

    dfs = [
        pd.read_parquet(p, columns=[id_col, name_col, text_col])
        for p in parquet_files
    ]

    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=[id_col], keep="first")
    names = df[name_col].map(_clean_text)
    descs = df[text_col].map(_clean_text)
    df["has_description"] = descs != ""
    df["text"] = np.where(df["has_description"], names + ". " + descs, names)
    df = df[df["text"] != ""]
    
    # Filter to KG IDs if provided
    if kg_ids:
        kg_set = set(kg_ids)
        df = df[df[id_col].isin(kg_set)]
        print(f"   Filtered to {len(df):,} diseases (from {len(kg_ids):,} KG IDs)")

    return df.reset_index(drop=True)


# -------------------------------------------------
# BATCH ENCODING
# -------------------------------------------------
@torch.no_grad()
def encode_batch_mean_pool(
    texts: List[str],
    tokenizer,
    model,
    max_length: int,
    device: str,
) -> np.ndarray:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs)
    # mean over real tokens only: padding must not change an embedding
    mask = inputs["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
    pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled.cpu().numpy()


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
def build_disease_embeddings(args: argparse.Namespace, kg_ids: list = None) -> Dict[str, np.ndarray]:
    df = load_disease_parquets(
        disease_dir=args.disease_dir,
        parquet_glob=args.parquet_glob,
        id_col=args.id_col,
        text_col=args.text_col,
        kg_ids=kg_ids,
        name_col=args.name_col,
    )
    print(f"   {len(df):,} diseases, {int(df['has_description'].sum()):,} with a description "
          f"({int((~df['has_description']).sum()):,} encoded from the name alone)")
    args.n_name_only = int((~df["has_description"]).sum())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Configure padding token if not present (e.g., GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   ⚠️  No pad_token found, using eos_token: '{tokenizer.eos_token}'")
    
    model = AutoModel.from_pretrained(args.model_name).to(args.device)
    model.eval()

    ids = df[args.id_col].tolist()
    texts = df["text"].tolist()

    embeddings: Dict[str, np.ndarray] = {}

    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i : i + args.batch_size]
        batch_ids = ids[i : i + args.batch_size]

        batch_embs = encode_batch_mean_pool(
            batch_texts,
            tokenizer,
            model,
            args.max_length,
            args.device,
        )

        for did, emb in zip(batch_ids, batch_embs):
            embeddings[did] = emb

    return embeddings


# -------------------------------------------------
# SAVE OUTPUTS
# -------------------------------------------------
def save_outputs(
    embeddings: Dict[str, np.ndarray],
    args: argparse.Namespace,
):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {k: torch.tensor(v) for k, v in embeddings.items()},
        output_dir / "disease_embeddings.pt",
    )

    with open(output_dir / "disease_embeddings.yaml", "w") as f:
        yaml.safe_dump(
            {
                "meta": {
                    "model": args.model_name,
                    "embedding_dim": int(next(iter(embeddings.values())).shape[0]),
                    "text": "<name>. <description>, or <name> alone when there is no description",
                    "pooling": "mean of last hidden layer over non-padding tokens",
                    "max_length": args.max_length,
                    "num_diseases": len(embeddings),
                    "num_name_only": getattr(args, "n_name_only", None),
                    "source_dir": args.disease_dir,
                },
            },
            f,
            sort_keys=False,
        )


# -------------------------------------------------
# ARGPARSE
# -------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GPT-based disease embeddings from parquet directory"
    )

    parser.add_argument("--disease-dir", required=True, help="Evidence directory with disease parquets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kg-ids-file", default=None, help="Parquet file with KG disease IDs to filter to")

    parser.add_argument("--parquet-glob", default="part-*.parquet")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--text-col", default="description")
    parser.add_argument("--name-col", default="name")

    parser.add_argument("--model-name", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext") 
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
def main():
    args = parse_args()
    
    # Load KG IDs if provided
    kg_ids = None
    if args.kg_ids_file and Path(args.kg_ids_file).exists():
        print(f"Loading KG disease IDs from {args.kg_ids_file}")
        kg_df = pd.read_parquet(args.kg_ids_file)
        kg_ids = kg_df['id'].tolist()
        print(f"Filtering to {len(kg_ids):,} KG disease IDs")
    
    embeddings = build_disease_embeddings(args, kg_ids=kg_ids)
    save_outputs(embeddings, args)


if __name__ == "__main__":
    main()

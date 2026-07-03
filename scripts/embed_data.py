# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""Embed a BEIR-style HF dataset into deduplicated query and key arrays."""

import argparse
import pathlib

import numpy as np
from datasets import DatasetDict, concatenate_datasets, load_dataset
from sentence_transformers import SentenceTransformer


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="BeIR/fiqa-generated-queries")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--query_text_column", type=str, default="query")
    parser.add_argument("--key_text_column", type=str, default="text")
    parser.add_argument("--encoder", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--output_path", type=str, default="./fiqa_embeddings")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    return parser


def embed(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.astype(np.float32, copy=False)


def main(args: argparse.Namespace) -> None:
    data = load_dataset(args.dataset, args.dataset_config)
    merged = (
        concatenate_datasets(list(data.values()))
        if isinstance(data, DatasetDict)
        else data
    )
    print(f"Loaded {args.dataset}: {len(merged)} rows")

    queries = list(dict.fromkeys(merged[args.query_text_column]))
    keys = list(dict.fromkeys(merged[args.key_text_column]))
    print(f"Unique queries: {len(queries)}")
    print(f"Unique keys:    {len(keys)}")

    model = SentenceTransformer(args.encoder, device=args.device)
    query_emb = embed(model, queries, args.batch_size)
    key_emb = embed(model, keys, args.batch_size)

    out_dir = pathlib.Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "queries.npy", query_emb)
    np.save(out_dir / "keys.npy", key_emb)
    print(f"Wrote queries={len(query_emb)} keys={len(key_emb)} to {out_dir}")


if __name__ == "__main__":
    main(_get_parser().parse_args())

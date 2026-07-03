# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
from typing import Literal, TypeAlias

import dataclasses
import pathlib

import grain
import jax
import numpy as np

__all__ = [
    "Batch",
    "QueryKeySource",
    "DotProduct",
    "build_dataloader",
]

Batch: TypeAlias = dict[
    Literal["query", "score", "target", "topk_indices"], np.ndarray | jax.Array
]


@dataclasses.dataclass(frozen=True)
class KeysMetadata:
    keys: np.memmap
    num_cls: int

    @property
    def num_keys(self) -> int:
        return self.keys.shape[0]

    @property
    def dim(self) -> int:
        return self.keys.shape[1]

    @property
    def numel(self) -> int:
        return np.prod(self.keys.shape)


class QueryKeySource(grain.sources.RandomAccessDataSource[dict[str, np.ndarray]]):
    """Random-access source returning a query and its corresponding top-k keys."""

    def __init__(self, *, queries_path: str, topk_path: str, keys_path: str):
        super().__init__()
        self._queries = np.load(queries_path, mmap_mode="r")
        self._topk = np.load(topk_path, mmap_mode="r")
        self._keys = np.load(keys_path, mmap_mode="r")
        assert len(self._queries) == len(self._topk), (
            len(self._queries),
            len(self._topk),
        )

    def __len__(self) -> int:
        return len(self._queries)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        topk_indices = self._topk[index]
        return {
            "query": np.asarray(self._queries[index]),
            "keys": np.asarray(self._keys[topk_indices]),
            "topk_indices": topk_indices,
        }

    @property
    def keys(self) -> np.memmap:
        return self._keys

    @property
    def num_cls(self) -> int:
        # topk_indices has shape [num_queries, num_cls]
        return self._topk.shape[1]


class DotProduct(grain.transforms.Map):
    """Compute per-query dot products between query and its per-cluster top-1 key."""

    def map(self, batch: dict[str, np.ndarray]) -> Batch:
        # query: [B, dim], keys: [B, num_cls, dim] -> score: [B, num_cls]
        score = np.einsum("bcd,bd->bc", batch["keys"], batch["query"])
        return {
            "query": batch["query"],
            "score": score,
            "target": batch["keys"],
            "topk_indices": batch["topk_indices"],
        }


def build_dataloader(
    path: str,
    *,
    split: Literal["train", "test"],
    batch_size: int,
    num_epochs: int | None,
    shuffle: bool,
    drop_remainder: bool,
    seed: int,
    num_workers: int = 0,
    per_worker_buffer_size: int = 4,
    return_key_metadata: bool = False,
) -> tuple[grain.IterDataset[Batch], KeysMetadata | None]:
    queries_path = pathlib.Path(path) / split / "queries.npy"
    topk_path = pathlib.Path(path) / split / "topk_indices.npy"
    keys_path = pathlib.Path(path) / "keys.npy"

    source = QueryKeySource(
        queries_path=str(queries_path),
        topk_path=str(topk_path),
        keys_path=str(keys_path),
    )
    ds = grain.MapDataset.source(source)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    ds = ds.repeat(num_epochs)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(DotProduct())
    ds = ds.to_iter_dataset()
    if num_workers:
        mp_options = grain.MultiprocessingOptions(
            num_workers=num_workers,
            per_worker_buffer_size=per_worker_buffer_size,
        )
        ds = ds.mp_prefetch(mp_options)

    if return_key_metadata:
        keys_meta = KeysMetadata(keys=source.keys, num_cls=source.num_cls)
        return ds, keys_meta
    return ds, None

# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""Cluster keys, augment all queries, compute per-cluster top-1, then split.

Augments every query (original + ``num_noise_samples`` renormalized noisy copies),
computes the argmax key within each cluster (remapped to a global index into
``keys.npy``), pools all samples, shuffles, and splits ``test_size`` rows off the
pool. ``topk_indices.npy`` has shape ``(num_samples, num_clusters)``.
"""

import argparse
import functools
import os
import pathlib
import shutil
import tempfile

import grain
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from ott.geometry import pointcloud, costs
from ott.tools import k_means


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_noise_samples", type=int, default=0)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--test_size", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=0)
    # clustering
    parser.add_argument("--num_clusters", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    return parser


def get_default_device() -> jax.Device:
    try:
        return jax.devices("gpu")[0]
    except RuntimeError:
        return jax.devices("cpu")[0]


def cluster_keys(
    keys: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray]:
    @functools.partial(
        jax.jit, static_argnames=("k", "max_iterations", "tolerance", "n_init")
    )
    def cluster_fn(
        rng: jax.Array,
        x: jnp.ndarray,
        k: int,
        max_iterations: int,
        tolerance: float,
        n_init: int,
    ) -> tuple[jax.Array, jax.Array, float, bool]:
        geom = pointcloud.PointCloud(x, cost_fn=costs.SqEuclidean())
        result = k_means.k_means(
            geom,
            k=k,
            init="k-means++",
            n_init=n_init,
            tol=tolerance,
            max_iterations=max_iterations,
            rng=rng,
        )
        return result.centroids, result.assignment, result.error, result.converged

    device = get_default_device()
    num_keys, _ = keys.shape
    assert num_keys >= args.num_clusters, (num_keys, args.num_clusters)

    if args.num_clusters == 1:
        print("num_clusters=1: skipping kmeans")
        assignments_np = np.zeros(num_keys, dtype=np.uint32)
        centroids_np = keys.mean(axis=0, keepdims=True)
        return centroids_np, assignments_np

    print(f"Clustering {num_keys} keys into {args.num_clusters} clusters")

    centroids, assignments, cost, converged = cluster_fn(
        jax.random.key(args.seed),
        jax.device_put(keys, device),
        args.num_clusters,
        args.max_iterations,
        args.tolerance,
        20,
    )
    assert bool(converged), f"Clustering did not converge (cost={float(cost):.4f})."

    assignments_np = jax.device_get(assignments)
    centroids_np = jax.device_get(centroids)
    sizes = np.bincount(assignments_np, minlength=args.num_clusters)
    print(
        f"Clustering converged: cost={float(cost):.4f}, "
        f"sizes min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.1f}"
    )
    return centroids_np, assignments_np


class QueriesSource(grain.sources.RandomAccessDataSource[np.ndarray]):
    def __init__(self, path: str):
        super().__init__()
        self._path = path
        self._queries = np.load(path, mmap_mode="r")

    def __getitem__(self, index: int) -> np.ndarray:
        return np.asarray(self._queries[index])

    def __len__(self) -> int:
        return len(self._queries)


class AddNoise(grain.experimental.FlatMapTransform):
    """Expand each query into the clean query plus `num_noise_samples` noisy copies."""

    def __init__(self, num_noise_samples: int, *, noise_std: float, seed: int):
        assert num_noise_samples >= 0, num_noise_samples
        super().__init__()
        self._num_noise_samples = num_noise_samples
        self._noise_std = noise_std
        self._rng = np.random.default_rng(seed)

    def flat_map(self, query: np.ndarray) -> list[np.ndarray]:
        noises = self._rng.normal(
            0.0,
            self._noise_std,
            size=(self._num_noise_samples, *query.shape),
        ).astype(query.dtype)
        return [query] + [query + noise for noise in noises]


@jax.jit
def compute_top1(query: jax.Array, cluster_keys: jax.Array) -> jax.Array:
    scores = query @ cluster_keys.T
    return jnp.argmax(scores, axis=-1)


class PerClusterTop1(grain.transforms.Map):
    """L2-normalize the query, then compute per-cluster argmax and remap to global indices."""

    def __init__(
        self,
        keys: np.ndarray,
        assignments: np.ndarray,
        *,
        num_clusters: int,
    ):
        super().__init__()
        self._num_clusters = num_clusters
        self._device = get_default_device()
        self._cluster_keys: list[jax.Array] = []
        self._global_indices: list[np.ndarray] = []
        for cluster_id in range(num_clusters):
            mask = assignments == cluster_id
            cluster_keys = keys[mask]
            assert cluster_keys.shape[0] >= 1, (cluster_id, cluster_keys.shape)
            self._cluster_keys.append(jax.device_put(cluster_keys, self._device))
            self._global_indices.append(np.where(mask)[0].astype(np.uint32))

    def map(self, query: np.ndarray) -> dict[str, np.ndarray]:
        norms = np.linalg.norm(query, axis=-1, keepdims=True)
        norms[norms == 0] = 1e-10
        query = (query / norms).astype(np.float32, copy=False)
        query_jax = jax.device_put(query, self._device)

        top1_global = np.empty((query.shape[0], self._num_clusters), dtype=np.uint32)

        for cluster_id, (cluster_keys, global_indices) in enumerate(
            zip(self._cluster_keys, self._global_indices)
        ):
            top1_local = compute_top1(query_jax, cluster_keys)
            top1_local = jax.device_get(top1_local)
            top1_global[:, cluster_id] = global_indices[top1_local]

        return {
            "query": query,
            "topk_indices": top1_global,
        }


def build_dataloader(
    queries_path: str,
    *,
    keys: np.ndarray,
    assignments: np.ndarray,
    num_clusters: int,
    num_noise_samples: int,
    noise_std: float,
    batch_size: int,
    seed: int,
    read_threads: int = 4,
    prefetch_buffer_size: int = 4,
) -> tuple[grain.IterDataset, tuple[int, int]]:
    source = QueriesSource(queries_path)
    shape = (len(source), *source[0].shape)

    ds = grain.MapDataset.source(source)
    ds = ds.to_iter_dataset(
        read_options=grain.ReadOptions(
            num_threads=read_threads, prefetch_buffer_size=prefetch_buffer_size
        )
    )
    if num_noise_samples > 0:
        ds = grain.experimental.FlatMapIterDataset(
            ds,
            AddNoise(
                num_noise_samples,
                noise_std=noise_std,
                seed=seed,
            ),
        )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.map(
        PerClusterTop1(
            keys,
            assignments,
            num_clusters=num_clusters,
        )
    )
    return ds, shape


def _augment_pool(
    args: argparse.Namespace,
    *,
    keys: np.ndarray,
    assignments: np.ndarray,
    pool_dir: pathlib.Path,
) -> tuple[int, int]:
    queries_path = os.path.join(args.input_path, "queries.npy")
    ds, (num_queries, dim) = build_dataloader(
        queries_path,
        keys=keys,
        assignments=assignments,
        num_clusters=args.num_clusters,
        num_noise_samples=args.num_noise_samples,
        noise_std=args.noise_std,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    num_total = num_queries * (args.num_noise_samples + 1)
    q_out = np.lib.format.open_memmap(
        pool_dir / "queries.npy", mode="w+", dtype=np.float32, shape=(num_total, dim)
    )
    i_out = np.lib.format.open_memmap(
        pool_dir / "topk_indices.npy",
        mode="w+",
        dtype=np.uint32,
        shape=(num_total, args.num_clusters),
    )
    offset = 0
    for record in tqdm.tqdm(ds, total=max(num_total // args.batch_size, 1)):
        bs = len(record["query"])
        q_out[offset : offset + bs] = record["query"]
        i_out[offset : offset + bs] = record["topk_indices"]
        offset += bs
    assert offset == num_total, (offset, num_total)
    q_out.flush()
    i_out.flush()
    return num_total, dim


def _write_split(
    split: str,
    idx: np.ndarray,
    *,
    pool_dir: pathlib.Path,
    out_dir: pathlib.Path,
    dim: int,
    num_clusters: int,
    chunk: int = 1_000_000,
) -> None:
    idx = np.sort(idx)
    pool_q = np.load(pool_dir / "queries.npy", mmap_mode="r")
    pool_i = np.load(pool_dir / "topk_indices.npy", mmap_mode="r")
    sdir = out_dir / split
    sdir.mkdir(parents=True, exist_ok=True)
    q_out = np.lib.format.open_memmap(
        sdir / "queries.npy", mode="w+", dtype=np.float32, shape=(len(idx), dim)
    )
    i_out = np.lib.format.open_memmap(
        sdir / "topk_indices.npy",
        mode="w+",
        dtype=np.uint32,
        shape=(len(idx), num_clusters),
    )
    for s in tqdm.tqdm(range(0, len(idx), chunk), desc=f"write {split}"):
        c = idx[s : s + chunk]
        q_out[s : s + len(c)] = pool_q[c]
        i_out[s : s + len(c)] = pool_i[c]
    q_out.flush()
    i_out.flush()


def main(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(pathlib.Path(args.input_path) / "keys.npy", out_dir / "keys.npy")
    keys = np.load(out_dir / "keys.npy")

    centroids, assignments = cluster_keys(keys, args)
    np.save(out_dir / "cluster_assignments.npy", assignments)
    np.save(out_dir / "centroids.npy", centroids)

    with tempfile.TemporaryDirectory() as tmp_dir:
        pool_dir = pathlib.Path(tmp_dir)
        num_total, dim = _augment_pool(
            args, keys=keys, assignments=assignments, pool_dir=pool_dir
        )
        perm = np.random.default_rng(args.seed).permutation(num_total)
        test_size = min(args.test_size, num_total)

        _write_split(
            "test",
            perm[:test_size],
            pool_dir=pool_dir,
            out_dir=out_dir,
            dim=dim,
            num_clusters=args.num_clusters,
        )
        _write_split(
            "train",
            perm[test_size:],
            pool_dir=pool_dir,
            out_dir=out_dir,
            dim=dim,
            num_clusters=args.num_clusters,
        )


if __name__ == "__main__":
    main(_get_parser().parse_args())

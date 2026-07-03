# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""Training and evaluation kernels."""

from typing import Iterable, Callable

import chex
import faiss
from flax import nnx
import jax
import jax.numpy as jnp
import optax
import tqdm

from amips import metrics, utils, data, models

__all__ = [
    "train_step",
    "eval_step",
    "faiss_eval_step",
    "evaluate",
]


@chex.assert_max_traces(n=1)
def train_step(
    model: models.ModelType,
    optimizer: nnx.Optimizer,
    ema: models.EMA | None,
    batch: data.Batch,
    score_weight: float = 1.0,
    target_weight: float = 1.0,
) -> dict[str, jax.Array]:
    def compute_loss(model: models.ModelType) -> tuple[jax.Array, dict[str, jax.Array]]:
        query = batch["query"]
        pred_score = model(query)  # [batch, num_cls?]
        pred_target = model.gradient(query)  # [batch, num_cls?, dim]
        if pred_score.ndim == 1:
            pred_score = pred_score[:, None]
            pred_target = pred_target[:, None]
        gt_score = batch["score"]  # [batch, num_cls]
        gt_target = batch["target"]  # [batch, num_cls, dim]

        score_loss = optax.squared_error(pred_score, gt_score).mean()
        target_loss = optax.squared_error(pred_target, gt_target).mean()
        total_loss = score_weight * score_loss + target_weight * target_loss
        aux = {"score_loss": score_loss, "target_loss": target_loss}
        return total_loss, aux

    (loss, aux), grads = nnx.value_and_grad(compute_loss, has_aux=True)(model)
    optimizer.update(model, grads)
    if ema is not None:
        ema.update(model)
    aux["total_loss"] = loss
    aux["grad_norm"] = optax.tree.norm(grads)
    return aux


@chex.assert_max_traces(n=2)
def eval_step(
    model: models.ModelType,
    batch: data.Batch,
    all_keys: jax.Array,
) -> dict[str, jax.Array]:
    # [batch_size, num_cls, dim]
    batch_size, num_cls, dim = batch["target"].shape

    query = batch["query"]
    topk_indices = batch["topk_indices"]  # [batch, num_cls]
    gt_score = batch["score"]  # [batch, num_cls]
    gt_target = batch["target"]  # [batch, num_cls, dim]

    pred_score = model(query)  # [batch, num_cls?]
    pred_target = model.gradient(query)  # [batch, num_cls?, dim]
    if pred_score.ndim == 1:
        pred_score = pred_score[:, None]
        pred_target = pred_target[:, None]

    routed_cls = jnp.argmax(pred_score, axis=-1)
    # for the retrieval metrics, use gradient from predicted cluster
    retrieval_grads = pred_target[jnp.arange(batch_size), routed_cls]
    assert retrieval_grads.shape == (batch_size, dim)
    similarity_scores = retrieval_grads @ all_keys.T

    # per-cluster metrics: vmap over clusters (axis 1), compare pred_i vs target_i
    in_axes = (1, 1, 1, 1, 1, None, None)
    regression_metrics_fn = jax.vmap(
        metrics.regression_metrics, in_axes=in_axes, out_axes=1
    )
    eval_metrics = regression_metrics_fn(
        pred_score,
        pred_target,
        gt_score,
        gt_target,
        topk_indices,
        query,
        similarity_scores,
    )
    # average over clusters, sum over batch
    eval_metrics = jax.tree.map(lambda x: jnp.mean(x, axis=1), eval_metrics)

    if num_cls > 1:
        routing_metrics = metrics.routing_metrics(pred_score, gt_score)
        eval_metrics.update(routing_metrics)

    return jax.tree.map(lambda x: jnp.sum(x, axis=0), eval_metrics)


@nnx.jit
def _project_queries(model: models.ModelType, query: jax.Array) -> jax.Array:
    return model.gradient(query)


def faiss_eval_step(
    model: models.ModelType,
    batch: data.Batch,
    _all_keys: None,
    *,
    faiss_index: faiss.IndexIVF,
    natural_nprobe: int,
    projected_nprobe: int,
    k_values: tuple[int, ...],
) -> dict[str, jax.Array]:
    max_k = max(k_values)
    # assuming only 1 cluster, squeeze
    gt_indices = batch["topk_indices"].squeeze(-1)

    natural_query = batch["query"]  # [batch_size, dim]
    projected_query = _project_queries(model, natural_query)
    assert projected_query.shape == natural_query.shape
    natural_query_np = jax.device_get(natural_query)
    projected_query_np = jax.device_get(projected_query)

    _, natural_indices_np = utils.search_index(
        faiss_index, natural_query_np, k=max_k, nprobe=natural_nprobe
    )
    _, projected_indices_np = utils.search_index(
        faiss_index, projected_query_np, k=max_k, nprobe=projected_nprobe
    )
    natural_indices = jax.device_put(natural_indices_np)
    projected_indices = jax.device_put(projected_indices_np)

    natural_metrics = {
        f"natural_recall@{k}": metrics.recall_at_k(
            natural_indices, gt_indices, start=0, end=k
        )
        for k in k_values
    }
    projected_metrics = {
        f"projected_recall@{k}": metrics.recall_at_k(
            projected_indices, gt_indices, start=0, end=k
        )
        for k in k_values
    }
    faiss_metrics = {**natural_metrics, **projected_metrics}
    return jax.tree.map(lambda x: jnp.sum(x, axis=0), faiss_metrics)


def evaluate(
    model: models.ModelType,
    dataloader: Iterable[data.Batch],
    *,
    eval_step_fn: Callable[
        [models.ModelType, data.Batch, jax.Array], dict[str, jax.Array]
    ],
    all_keys: jax.Array | None,
    num_steps: int | None = None,
) -> dict[str, jax.Array]:
    model.eval()
    all_metrics, num_samples = None, 0
    for step, batch in enumerate(tqdm.tqdm(dataloader, desc="Evaluation")):
        if num_steps and step >= num_steps:
            break
        num_samples += batch["query"].shape[0]
        batch_metrics = eval_step_fn(model, batch, all_keys)
        all_metrics = (
            batch_metrics
            if all_metrics is None
            else jax.tree.map(lambda a, b: a + b, all_metrics, batch_metrics)
        )
    all_metrics = jax.tree.map(lambda x: (x / num_samples).item(), all_metrics)
    model.train()
    return all_metrics

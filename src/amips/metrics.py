# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
import jax
import jax.numpy as jnp
import optax


__all__ = [
    "mrr",
    "recall_at_k",
    "retrieval_metrics",
    "routing_accuracy",
    "routing_mrr",
    "routing_recall_at_k",
    "regression_metrics",
    "routing_metrics",
]


def mrr(similarity_scores: jax.Array, ground_truth_indices: jax.Array) -> jax.Array:
    """Compute sum of reciprocal ranks across batch.

    Args:
        similarity_scores: [batch_size, n_keys] similarity score
        ground_truth_indices: [batch_size] ground truth key indices

    Returns:
        Sum of reciprocal ranks across batch, array of shape ``[batch_size,]``.
    """
    batch_size, n_keys = similarity_scores.shape
    # Get the similarity score for ground truth items
    batch_indices = jnp.arange(batch_size)
    ground_truth_score = similarity_scores[batch_indices, ground_truth_indices]
    # Count how many items have higher score than ground truth
    # Shape: [batch_size, n_keys] -> [batch_size]
    higher_score = (similarity_scores > ground_truth_score[:, None]).sum(axis=1)
    # Rank is count of higher score + 1
    ranks = higher_score + 1.0
    # Reciprocal rank, then sum across batch
    return 1.0 / ranks


def recall_at_k(
    sorted_indices: jax.Array,
    ground_truth_indices: jax.Array,
    *,
    start: int,
    end: int,
) -> jax.Array:
    """Compute sum of recall@k hits across batch using pre-sorted indices.

    Args:
        sorted_indices: [batch_size, n_keys] indices sorted by similarity (descending order)
        ground_truth_indices: [batch_size] ground truth key indices
        start: star index, inclusive.
        end: end index, exclusive.

    Returns:
        Recall, array of shape ``[batch_size,]``.
    """
    assert 0 <= start <= sorted_indices.shape[1], start
    assert 0 <= end <= sorted_indices.shape[1], end
    assert start < end, (start, end)
    # Get top-k indices (order is descending)
    top_k_indices = sorted_indices[:, start:end]
    # Check if ground truth is in top-k for each query
    return (top_k_indices == ground_truth_indices[:, None]).any(axis=1)


def retrieval_metrics(
    similarity_scores: jax.Array,
    ground_truth_indices: jax.Array,
) -> dict[str, jax.Array]:
    """Compute all retrieval metrics: sum of accuracy, MRR and Recall@k for multiple k values across batch.

    Args:
        similarity_scores: [batch_size, n_keys] similarity score
        ground_truth_indices: [batch_size] ground truth key indices

    Returns:
        Dictionary with metric names as keys and values as arrays of shape ``[batch_size,]``.
    """
    _, n_keys = similarity_scores.shape

    # Compute MRR
    metrics = {"mrr_value": mrr(similarity_scores, ground_truth_indices)}

    # Sort indices once for all recall@k computations
    sorted_indices = jnp.argsort(similarity_scores, descending=True, axis=-1)
    for pct in [0.01]:
        # Define window size values: 1%, 5%, and 10% of key set size (with minimum of 1)
        end = min(max(1, int(pct * n_keys)), n_keys)
        # Compute all recall@k values with a single sort
        metrics[f"recall@{pct:.2f}"] = recall_at_k(
            sorted_indices, ground_truth_indices, start=0, end=end
        )

    for k in [5, 10, 50]:
        if k > n_keys:
            continue
        metrics[f"recall@{k}"] = recall_at_k(
            sorted_indices, ground_truth_indices, start=0, end=k
        )

    # Compute accuracy (recall@1)
    metrics["accuracy"] = recall_at_k(
        sorted_indices, ground_truth_indices, start=0, end=1
    )
    return metrics


def routing_accuracy(
    pred_score: jax.Array,
    true_cluster: jax.Array,
) -> jax.Array:
    """Compute routing accuracy: best-scoring model == true cluster.

    Args:
        pred_score: Predicted score from all K models [batch_size, num_cls]
        true_cluster: Ground truth cluster indices [batch_size]

    Returns:
        Binary accuracy per query [batch_size]
    """
    best_model = jnp.argmax(pred_score, axis=-1)
    return (best_model == true_cluster).astype(jnp.float32)


def routing_mrr(
    pred_score: jax.Array,
    true_cluster: jax.Array,
) -> jax.Array:
    """Compute routing MRR: reciprocal rank of true cluster.

    Args:
        pred_score: Predicted score from all K models [batch_size, num_cls]
        true_cluster: Ground truth cluster indices [batch_size]

    Returns:
        Reciprocal rank per query [batch_size]
    """
    sorted_indices = jnp.argsort(pred_score, axis=-1, descending=True)
    batch_size = true_cluster.shape[0]
    ranks = (
        jnp.argwhere(
            sorted_indices == true_cluster[:, None], size=batch_size, fill_value=-1
        )[:, 1]
        + 1
    )
    return 1.0 / ranks.astype(jnp.float32)


def routing_recall_at_k(
    pred_score: jax.Array,
    true_cluster: jax.Array,
) -> jax.Array:
    """Routing recall@k for all k=1..num_cls.

    Args:
        pred_score: [batch_size, num_cls]
        true_cluster: [batch_size]

    Returns:
        [K, batch_size] where [k-1] is recall@k
    """
    _, num_cls = pred_score.shape
    sorted_indices = jnp.argsort(pred_score, axis=-1, descending=True)
    ranks = jnp.argmax(sorted_indices == true_cluster[:, None], axis=1) + 1
    k_vals = jnp.arange(1, num_cls + 1)
    return (ranks[None, :] <= k_vals[:, None]).astype(jnp.float32)


def regression_metrics(
    pred_score: jax.Array,
    pred_target: jax.Array,
    gt_score: jax.Array,
    gt_target: jax.Array,
    topk_indices: jax.Array,
    # below args are not vmapped
    query: jax.Array,
    similarity_scores: jax.Array,
) -> dict[str, jax.Array]:
    def ratio_error(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
        eps = jnp.finfo(numerator.dtype).eps
        return numerator / (denominator + eps)

    def log_ratio_error(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
        eps = jnp.finfo(numerator.dtype).eps
        return 0.5 * (jnp.log(numerator + eps) - jnp.log(denominator + eps))

    batch_size, dim = gt_target.shape
    assert pred_score.shape == (batch_size,), pred_score.shape
    assert pred_target.shape == (batch_size, dim), pred_target.shape
    assert gt_score.shape == (batch_size,), gt_score.shape
    assert query.shape == (batch_size, dim), query.shape
    assert topk_indices.shape == (batch_size,), topk_indices.shape
    assert similarity_scores.shape[0] == batch_size, similarity_scores.shape
    norm_pred_target = jnp.linalg.norm(pred_target, axis=-1)

    score_error = optax.squared_error(pred_score, gt_score)
    target_error = optax.squared_error(pred_target, gt_target).mean(-1)
    query_to_gt = optax.squared_error(query, gt_target).mean(-1)

    # Relative error: |y-∇|/|y-x|
    target_error_rn2 = ratio_error(target_error, query_to_gt)
    log_target_error_rn2 = log_ratio_error(target_error, query_to_gt)

    retrieval_results = retrieval_metrics(similarity_scores, topk_indices)

    assert score_error.shape == (batch_size,)
    assert target_error.shape == (batch_size,)
    assert target_error_rn2.shape == (batch_size,)
    assert log_target_error_rn2.shape == (batch_size,)

    return {
        "score_error": score_error,
        "target_error": target_error,
        "target_error_rn2": target_error_rn2,
        "log_target_error_rn2": log_target_error_rn2,
        "norm_pred_target": norm_pred_target,
        **retrieval_results,
    }


def routing_metrics(pred_score: jax.Array, gt_score: jax.Array) -> dict[str, jax.Array]:
    _, num_cls = gt_score.shape  # [batch, num_cls]
    true_cluster = jnp.argmax(gt_score, axis=-1)  # [batch,]

    metrics = {
        "routing_accuracy": routing_accuracy(pred_score, true_cluster),
        "routing_mrr": routing_mrr(pred_score, true_cluster),
    }
    routing_recall = routing_recall_at_k(pred_score, true_cluster)
    for k in range(num_cls):
        metrics[f"routing_acc@{k + 1}"] = routing_recall[k]
    return metrics

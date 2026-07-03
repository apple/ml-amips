# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
from flax import nnx
from ott.neural.networks.icnn import ICNN as SupportNet
import jax
import jax.numpy as jnp
import numpy as np
import faiss


__all__ = [
    "Homogenize",
    "rectangular_dims",
    "soft_leaky_relu",
    "create_ivf_index",
    "search_index",
]


class Homogenize(nnx.Module):
    """A wrapper that turns a model (`h`) into a positive homogeneous `f`.

    `f(x) = ||x|| * h(x/||x||)`.
    """

    def __init__(self, model: SupportNet):
        assert isinstance(model, SupportNet), type(model)
        super().__init__()
        self.model = model

    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate f(x) = ||x|| * h(x/||x||)."""
        is_batched = x.ndim > 1
        x = jnp.atleast_2d(x)
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        out = self.model(x / norm)
        out = (norm if self._is_multi_output else norm.squeeze(-1)) * out
        return out if is_batched else out.squeeze(0)

    def gradient(self, x: jax.Array, *, sphere_normalize: bool = False) -> jax.Array:
        def forward(x: jax.Array) -> jax.Array:
            return nnx.merge(graphdef, state)(x)

        graphdef, state = nnx.split(self)
        fn = jax.jacobian(forward) if self._is_multi_output else jax.grad(forward)
        z = jax.vmap(fn)(x)
        return _normalize(z, axis=-1) if sphere_normalize else z

    @property
    def _is_multi_output(self) -> bool:
        return self.model._output_dim > 1


def _normalize(x: jax.Array, axis: int = -1) -> jax.Array:
    """L2 normalize along the specified axis."""
    eps = jnp.finfo(x).tiny
    return x / (jnp.linalg.norm(x, keepdims=True, axis=axis) + eps)


def _compute_width(
    total_params: int,
    input_dim: int,
    depth: int,
    per_elem_params: int,
    num_wx_layers: int,
    output_dim: int | None = 1,
    width_multiple: int | None = 4,
) -> int:
    # Quadratic formula: w = (-b ± sqrt(b^2 - 4ac)) / (2a)
    if output_dim is None:
        # Special case: output_dim = width
        # The w * output_dim term becomes w^2, so a increases by 1
        # bias_correction = (w - 1) if per_elem_params > 0, which adds 1 to b and -1 to c
        a = depth
        if per_elem_params > 0:
            b = input_dim + per_elem_params * depth + num_wx_layers * input_dim + 1
            c = -1 - total_params
        else:
            b = input_dim + per_elem_params * depth + num_wx_layers * input_dim
            c = -total_params
    else:
        # Standard case with fixed output_dim
        # Bias correction: final layer has output_dim biases instead of 1
        # This is (output_dim - 1) extra constant params if bias is used
        bias_correction = (output_dim - 1) if per_elem_params > 0 else 0

        a = depth - 1
        b = input_dim + output_dim + per_elem_params * depth + num_wx_layers * input_dim
        c = bias_correction - total_params

    discriminant = b**2 - 4 * a * c
    if a == 0:
        width = -c / b if b != 0 else 0
    elif discriminant < 0:
        raise ValueError("No valid width solution exists.")
    else:
        width = (-b + np.sqrt(discriminant)) / (2 * a)

    assert width > 0, width
    # check that number of params not too far off
    assert np.abs(a * width**2 + b * width + c) < 10_000, width

    if width_multiple is not None:
        width = round(width / width_multiple) * width_multiple
        width = max(width, width_multiple)  # ensure at least one multiple

    return int(np.ceil(width))


def _compute_trapezoid_width(
    total_params: int,
    input_dim: int,
    depth: int,
    per_elem_params: int,
    num_wx_layers: int,
    multiplier: float,
    output_dim: int = 1,
    width_multiple: int | None = 4,
) -> int:
    assert 0 < multiplier <= 1.0, f"multiplier must be in (0, 1], got {multiplier}"

    # Bias correction: final layer has output_dim biases instead of 1
    bias_correction = (output_dim - 1) if per_elem_params > 0 else 0

    if depth == 1:
        # Single layer: just solve linear equation
        # total_params = input_dim * w + per_elem_params * w + multiplier * w * output_dim + bias_correction
        width = (total_params - bias_correction) / (
            input_dim + per_elem_params + multiplier * output_dim
        )
        if width_multiple is not None:
            width = round(width / width_multiple) * width_multiple
            width = max(width, width_multiple)
        return int(np.ceil(width))

    # For depth > 1, compute taper factors
    # w_i = w_1 * (1 - (i-1)/(depth-1) * (1 - multiplier))
    # where i ranges from 1 to depth
    def taper_factor(i: int) -> float:
        return 1.0 - (i - 1) / (depth - 1) * (1.0 - multiplier)

    # C1: sum of all taper factors (for bias and per-element params)
    # C1 = sum(taper_factor(i) for i in 1..depth)
    taper_factors = np.array([taper_factor(i) for i in range(1, depth + 1)])
    C1 = np.sum(taper_factors)

    # C2: sum of products of adjacent taper factors (for wz layers)
    # C2 = sum(taper_factor(i) * taper_factor(i+1) for i in 1..depth-1)
    C2 = np.sum(taper_factors[:-1] * taper_factors[1:])

    # C_wx: sum of taper factors for wx injection layers
    # This depends on the wx_inject pattern, which we'll need to compute
    # For now, approximate by distributing evenly: num_wx_layers / depth * C1
    # This is an approximation; exact computation would require knowing the injection pattern
    if num_wx_layers > 0:
        # Average taper factor for wx layers
        C_wx = C1 * num_wx_layers / (depth - 1)
    else:
        C_wx = 0.0

    # Quadratic formula: C2 * w^2 + b * w + c = 0
    a = C2
    b = input_dim + C_wx * input_dim + per_elem_params * C1 + multiplier * output_dim
    c = bias_correction - total_params

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("No valid width solution exists for trapezoid architecture.")

    width = (-b + np.sqrt(discriminant)) / (2 * a)

    assert width > 0, width
    # Check that number of params not too far off
    residual = np.abs(a * width**2 + b * width + c)
    assert residual < 10_000, f"Width solution residual too large: {residual}"

    if width_multiple is not None:
        width = round(width / width_multiple) * width_multiple
        width = max(width, width_multiple)

    return int(np.ceil(width))


def rectangular_dims(
    depth: int,
    parameters_fraction: float,
    input_dim: int,
    num_keys: int,
    use_bias: bool = True,
    wx_inject: bool | tuple[bool, ...] | int = False,
    multiplier: float = 1.0,
    output_dim: int = 1,
    param_adjust_factor: float = 1.0,
    width_multiple: int | None = 4,
) -> tuple[int, ...]:
    """Create rectangular or trapezoid architecture dimensions.

    Args:
        depth: Number of hidden layers.
        parameters_fraction: Fraction of (num_keys * input_dim) to use as parameter budget.
        input_dim: Input dimension.
        num_keys: Number of keys in the dataset.
        use_bias: Whether bias terms are used.
        wx_inject: Controls input re-injection. Can be:
            - bool: True means inject at all layers, False means no injection
            - tuple[bool, ...]: explicit mask for each layer (length = depth - 1)
            - int: frequency (e.g., 3 means inject every 3rd layer)
        multiplier: Ratio of rightmost to leftmost hidden layer width.
            - 1.0 (default): rectangular architecture with uniform width
            - < 1.0: trapezoid architecture with linear taper from left to right
        output_dim: Output dimension (1 for SupportNet, input_dim for KeyNet).
        param_adjust_factor: Factor to adjust parameter budget (e.g., 8.0 for blocks that
            use 8x more parameters). Budget will be divided by this factor.
        width_multiple: If set, round hidden width to the nearest multiple of this value
            (e.g. 16) for better XLA/tensor-core alignment. None disables rounding.

    Returns:
        Tuple of dimensions for the architecture.
    """
    assert depth >= 1, depth
    assert 0 < multiplier <= 1.0, f"multiplier must be in (0, 1], got {multiplier}"
    assert param_adjust_factor > 0, (
        f"param_adjust_factor must be positive, got {param_adjust_factor}"
    )

    # Adjust parameter budget based on adjustment factor
    base_params = num_keys * input_dim * parameters_fraction
    adjusted_params = base_params / param_adjust_factor

    total_params = np.ceil(adjusted_params)
    # Compute number of wx layers based on wx_inject
    num_hidden_layers = depth - 1  # layers after the first where wx can be injected
    if isinstance(wx_inject, bool):
        num_wx_layers = num_hidden_layers if wx_inject else 0
    elif isinstance(wx_inject, int):
        # Frequency: count how many layers have injection, accounting for wx0
        # wx0 is at position 0, wx layers are at positions 1, 2, ..., num_hidden_layers
        # Inject when (position) % wx_inject == 0
        num_wx_layers = sum(
            1 for i in range(num_hidden_layers) if (i + 1) % wx_inject == 0
        )
    else:
        # Tuple of bools
        num_wx_layers = sum(wx_inject)

    # Use trapezoid architecture if multiplier < 1.0
    if multiplier < 1.0:
        # Compute first layer width for trapezoid
        width_1 = _compute_trapezoid_width(
            total_params=total_params,
            input_dim=input_dim,
            depth=depth,
            per_elem_params=use_bias,
            num_wx_layers=num_wx_layers,
            multiplier=multiplier,
            output_dim=output_dim,
            width_multiple=width_multiple,
        )

        # Generate tapering widths
        def taper_factor(i: int) -> float:
            return (
                1.0 - (i - 1) / (depth - 1) * (1.0 - multiplier) if depth > 1 else 1.0
            )

        widths = tuple(
            int(np.ceil(width_1 * taper_factor(i))) for i in range(1, depth + 1)
        )
        return (input_dim,) + widths + (output_dim,)

    # Rectangular architecture with uniform width
    width = _compute_width(
        total_params=total_params,
        input_dim=input_dim,
        depth=depth,
        per_elem_params=use_bias,
        num_wx_layers=num_wx_layers,
        output_dim=output_dim,
        width_multiple=width_multiple,
    )
    return (input_dim,) + (width,) * depth + (output_dim,)


def soft_leaky_relu(
    x: jax.Array, negative_slope: float = 0.1, beta: float = 10.0
) -> jax.Array:
    return negative_slope * x + (1 - negative_slope) * jax.nn.softplus(beta * x) / beta


def create_ivf_index(
    keys: np.ndarray, n_clusters: int, metric: int = faiss.METRIC_INNER_PRODUCT
) -> faiss.IndexIVF:
    _, dim = keys.shape
    quantizer = (
        faiss.IndexFlatIP(dim)
        if metric == faiss.METRIC_INNER_PRODUCT
        else faiss.IndexFlatL2(dim)
    )
    index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, metric)
    index.train(keys.astype(np.float32))
    index.add(keys.astype(np.float32))
    return index


def search_index(
    index: faiss.IndexIVF, queries: np.ndarray, *, k: int, nprobe: int
) -> tuple[np.ndarray, np.ndarray]:
    index.nprobe = nprobe
    scores, indices = index.search(queries.astype(np.float32), k)
    return scores, indices

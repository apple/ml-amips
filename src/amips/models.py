# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
from typing import TypeAlias
import functools

from ott.neural.networks.icnn import ICNN as SupportNet, KeyNet
from flax import nnx
import jax
import numpy as np

from amips import config, utils

__all__ = ["ModelType", "EMA", "create_model", "count_params"]

ModelType: TypeAlias = SupportNet | KeyNet


class EMA(nnx.Module):
    def __init__(self, model: nnx.Module, *, decay: float):
        assert 0.0 < decay < 1.0, decay
        super().__init__()
        graphdef, state = nnx.split(model)
        state = jax.tree.map(lambda x: x.copy(), state)
        self.model = nnx.merge(graphdef, state)
        self.decay = decay

    def update(self, model: nnx.Module) -> None:
        def update_fn(p_ema: jax.Array, p_model: jax.Array) -> jax.Array:
            return self.decay * p_ema + (1.0 - self.decay) * p_model

        params = nnx.state(model)
        ema_params = nnx.state(self.model)
        ema_params = jax.tree.map(update_fn, ema_params, params)
        nnx.update(self.model, ema_params)


def create_model(
    cfg: config.Config,
    *,
    num_keys: int,
    dim: int,
    num_cls: int,
    rngs: nnx.Rngs,
) -> ModelType:
    model_config = cfg.model
    _, *hidden_dims, _ = utils.rectangular_dims(
        depth=model_config.depth,
        parameters_fraction=model_config.parameters_fraction,
        input_dim=dim,
        num_keys=num_keys,
        use_bias=model_config.use_bias,
        wx_inject=model_config.wx_inject,
        multiplier=model_config.multiplier,
        output_dim=num_cls if model_config.model == "supportnet" else dim,
    )

    if model_config.act_fn == "soft_leaky_relu":
        act_fn = functools.partial(
            utils.soft_leaky_relu,
            negative_slope=model_config.act_fn_negative_slope,
            beta=model_config.act_fn_beta,
        )
    else:
        act_fn = getattr(jax.nn, model_config.act_fn)

    if model_config.model == "supportnet":
        rectifier_fn = getattr(jax.nn, model_config.supportnet_rectifier_fn)
        model = SupportNet(
            hidden_dims,
            input_dim=dim,
            output_dim=num_cls,
            act_fn=act_fn,
            use_bias=model_config.use_bias,
            wx_inject=model_config.wx_inject,
            principled_init=model_config.supportnet_principled_init,
            rectifier_fn=rectifier_fn,
            rngs=rngs,
        )
        return utils.Homogenize(model) if model_config.supportnet_homogenize else model
    if model_config.model == "keynet":
        return KeyNet(
            hidden_dims,
            input_dim=dim,
            output_dim=dim,
            num_outputs=None if num_cls == 1 else num_cls,
            use_bias=model_config.use_bias,
            wx_inject=model_config.wx_inject,
            resnet=model_config.keynet_resnet,
            act_fn=act_fn,
            rngs=rngs,
        )
    raise NotImplementedError(model_config.model)


def count_params(model: nnx.Module) -> int:
    model_params = nnx.state(model, nnx.Param)
    return int(
        sum(jax.tree.map(lambda x: np.prod(x.shape), jax.tree.leaves(model_params)))
    )

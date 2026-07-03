# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
import dataclasses
import pathlib

import omegaconf

__all__ = [
    "DataloaderConfig",
    "LossConfig",
    "CheckpointConfig",
    "ModelConfig",
    "OptimizerConfig",
    "FaissConfig",
    "Config",
    "load_config",
]


@dataclasses.dataclass
class DataloaderConfig:
    trn_batch_size: int = 1024
    val_batch_size: int = 1024
    num_trn_steps: int = 100_000
    num_val_steps: int | None = None
    eval_every: int | None = 10_000


@dataclasses.dataclass
class LossConfig:
    score_weight: float = 1.0
    target_weight: float = 1.0


@dataclasses.dataclass
class CheckpointConfig:
    dir: str | None = None
    save_every: int = 10_000
    max_to_keep: int | None = 3


@dataclasses.dataclass
class ModelConfig:
    model: str = "supportnet"  # supportnet or keynet
    ema_decay: float | None = 0.999
    # common
    depth: int = 8
    parameters_fraction: float = 0.05
    use_bias: bool = True
    wx_inject: bool | int = True
    multiplier: float = 1.0
    # keynet only
    keynet_resnet: bool = False
    # supportnet only
    supportnet_principled_init: bool = True
    supportnet_rectifier_fn: str = "identity"
    supportnet_homogenize: bool = False
    # act_fn
    act_fn: str = "soft_leaky_relu"
    act_fn_negative_slope: float = 0.1
    act_fn_beta: float = 50.0


@dataclasses.dataclass
class OptimizerConfig:
    init_value: float = 1.0e-5
    peak_value: float = 1.0e-3
    end_value: float = 1.0e-4
    warmup_steps: int = 1_000
    grad_clip: float = 1.0


@dataclasses.dataclass
class FaissConfig:
    # any field left as null is derived from the number of keys below
    n_clusters: int | None = None
    natural_nprobe: int | None = None
    projected_nprobe: int | None = None
    k_values: list[int] | None = None
    num_faiss_steps: int | None = None

    def resolve(self, num_keys: int) -> "FaissConfig":
        n_clusters = (
            self.n_clusters if self.n_clusters is not None else int(num_keys**0.5)
        )
        natural_nprobe = (
            self.natural_nprobe
            if self.natural_nprobe is not None
            else int(0.05 * n_clusters)
        )
        projected_nprobe = (
            self.projected_nprobe
            if self.projected_nprobe is not None
            else int(0.05 * n_clusters)
        )
        k_values = (
            self.k_values
            if self.k_values is not None
            else [int(k_frac * n_clusters) for k_frac in (0.0001, 0.001, 0.005)]
        )
        k_values = sorted({max(k, 1) for k in k_values})
        return dataclasses.replace(
            self,
            n_clusters=n_clusters,
            natural_nprobe=natural_nprobe,
            projected_nprobe=projected_nprobe,
            k_values=k_values,
        )


@dataclasses.dataclass
class Config:
    seed: int = 0
    dataloader: DataloaderConfig = dataclasses.field(default_factory=DataloaderConfig)
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    checkpoint: CheckpointConfig = dataclasses.field(default_factory=CheckpointConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    faiss: FaissConfig = dataclasses.field(default_factory=FaissConfig)


def load_config(path: str | pathlib.Path) -> Config:
    schema = omegaconf.OmegaConf.structured(Config)
    user_cfg = omegaconf.OmegaConf.load(path)
    cfg = omegaconf.OmegaConf.merge(schema, user_cfg)
    return omegaconf.OmegaConf.to_object(cfg)

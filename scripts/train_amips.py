# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""Train an AMIPS support-function approximator on precomputed top-1 targets."""

import argparse
import contextlib
import functools
import pathlib

from flax import nnx
import jax
import optax
import orbax.checkpoint as ocp
import tqdm

from amips import config, utils, data, models, training


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Directory produced by precompute_top1.py.",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config."
    )
    return parser


def main(args: argparse.Namespace) -> None:
    """Train a model end-to-end from a config and a stage-2 output directory."""
    cfg = config.load_config(args.config)
    input_path = args.input_path

    train_dl, keys_meta = data.build_dataloader(
        input_path,
        split="train",
        batch_size=cfg.dataloader.trn_batch_size,
        num_epochs=None,
        shuffle=True,
        drop_remainder=True,
        seed=cfg.seed,
        return_key_metadata=True,
    )
    test_dl, _ = data.build_dataloader(
        input_path,
        split="test",
        batch_size=cfg.dataloader.val_batch_size,
        num_epochs=1,
        shuffle=False,
        drop_remainder=False,
        seed=cfg.seed,
        return_key_metadata=False,
    )

    mesh = jax.make_mesh((jax.device_count(),), "data")
    param_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    keys_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    if cfg.checkpoint.dir is not None:
        ckpt_dir = pathlib.Path(cfg.checkpoint.dir).resolve()
        ckpt_options = ocp.CheckpointManagerOptions(
            save_interval_steps=cfg.checkpoint.save_every,
            max_to_keep=cfg.checkpoint.max_to_keep,
        )
        ckpt_mngr_ctx = ocp.CheckpointManager(ckpt_dir, options=ckpt_options)
    else:
        ckpt_mngr_ctx = contextlib.nullcontext()

    with jax.set_mesh(mesh), ckpt_mngr_ctx as mngr:
        model = models.create_model(
            cfg,
            num_keys=keys_meta.num_keys,
            dim=keys_meta.dim,
            num_cls=keys_meta.num_cls,
            rngs=nnx.Rngs(cfg.seed + 1),
        )
        model = jax.device_put(model, param_sharding)
        nnx.display(model)
        num_params = models.count_params(model)
        print(
            f"Number of model params: {num_params}, "
            f"fraction of database size: {num_params / keys_meta.numel:.5f}"
        )

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=cfg.optimizer.init_value,
            peak_value=cfg.optimizer.peak_value,
            end_value=cfg.optimizer.end_value,
            warmup_steps=cfg.optimizer.warmup_steps,
            decay_steps=cfg.dataloader.num_trn_steps,
        )
        optimizer = optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(cfg.optimizer.grad_clip),
            optax.adam(lr_schedule),
        )
        optimizer = nnx.Optimizer(model, optimizer, wrt=nnx.Param)

        if cfg.model.ema_decay is not None:
            ema = models.EMA(model, decay=cfg.model.ema_decay)
            ema_sharding = param_sharding
            eval_model = ema.model
        else:
            ema = ema_sharding = None
            eval_model = model

        train_step_fn = nnx.jit(
            training.train_step,
            in_shardings=(param_sharding, param_sharding, ema_sharding, data_sharding),
            static_argnames=("score_weight", "target_weight"),
        )
        eval_step_fn = nnx.jit(
            training.eval_step,
            in_shardings=(param_sharding, data_sharding, keys_sharding),
        )
        all_keys = jax.device_put(keys_meta.keys[:], keys_sharding)

        for step, batch in enumerate(
            tqdm.tqdm(train_dl, desc="Training", total=cfg.dataloader.num_trn_steps)
        ):
            if step >= cfg.dataloader.num_trn_steps:
                break
            batch = jax.device_put(batch, data_sharding)
            _batch_metrics = train_step_fn(
                model,
                optimizer,
                ema,
                batch,
                cfg.loss.score_weight,
                cfg.loss.target_weight,
            )
            if (
                cfg.dataloader.eval_every
                and (step + 1) % cfg.dataloader.eval_every == 0
            ):
                eval_metrics = training.evaluate(
                    eval_model,
                    test_dl,
                    eval_step_fn=eval_step_fn,
                    all_keys=all_keys,
                    num_steps=cfg.dataloader.num_val_steps,
                )
                print("Eval metrics:", step, {k: v for k, v in eval_metrics.items()})

            if mngr is not None:
                mngr.save(
                    step,
                    args=ocp.args.StandardSave(nnx.state(eval_model)),
                )

        if mngr is not None:
            mngr.save(
                step,
                args=ocp.args.StandardSave(nnx.state(eval_model)),
                force=True,
            )
            mngr.wait_until_finished()

    if keys_meta.num_cls == 1:
        faiss_cfg = cfg.faiss.resolve(keys_meta.num_keys)
        faiss_index = utils.create_ivf_index(
            keys_meta.keys, n_clusters=faiss_cfg.n_clusters
        )
        faiss_eval_step_fn = functools.partial(
            training.faiss_eval_step,
            faiss_index=faiss_index,
            natural_nprobe=faiss_cfg.natural_nprobe,
            projected_nprobe=faiss_cfg.projected_nprobe,
            k_values=tuple(faiss_cfg.k_values),
        )

        faiss_eval_metrics = training.evaluate(
            eval_model,
            test_dl,
            eval_step_fn=faiss_eval_step_fn,
            all_keys=None,  # not needed
            num_steps=faiss_cfg.num_faiss_steps,
        )
        print(
            "FAISS eval metrics:", step, {k: v for k, v in faiss_eval_metrics.items()}
        )


if __name__ == "__main__":
    main(_get_parser().parse_args())

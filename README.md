# Amortizing Maximum Inner Product Search with Learned Support Functions (AMIPS)

<div>
<a href="https://arxiv.org/abs/2603.08001" target="_blank"><img alt="AMIPS arXiv" src="https://img.shields.io/badge/arXiv-AMIPS-red?logo=arxiv"/></a>
</div>

This software project accompanies the research paper:
[Amortizing Maximum Inner Product Search with Learned Support Functions](https://arxiv.org/abs/2603.08001)
by *Theo X. Olausson, João Monteiro, Michal Klein and Marco Cuturi*.

Three-stage pipeline that turns a [BEIR-style](https://github.com/beir-cellar/beir) HuggingFace dataset into a trained
AMIPS support-function approximator:

1. [scripts/embed_data.py](scripts/embed_data.py) — embed queries and keys into d-dimensional vectors.
2. [scripts/precompute_top1.py](scripts/precompute_top1.py) — cluster keys, then store the per-cluster top-1 index for every query.
3. [scripts/train_amips.py](scripts/train_amips.py) — train a
   [SupportNet](https://ott-jax.readthedocs.io/neural/_autosummary/ott.neural.networks.icnn.ICNN.html#ott.neural.networks.icnn.ICNN)
   or a [KeyNet](https://ott-jax.readthedocs.io/neural/_autosummary/ott.neural.networks.icnn.KeyNet.html#ott.neural.networks.icnn.KeyNet)
   on top of the cached targets.

## Installation

```bash
bash scripts/setup.sh
```

This installs the `amips` package (with `ott-jax` pinned to the commit carrying the
KeyNet final-layer fix) and then the matched GPU JAX stack on top.

Then run each stage with `python scripts/<stage>.py` as shown below.

## 1. Embed the dataset

Loads a HuggingFace dataset, deduplicates the query/key text columns and encodes
them into L2-normalized d-dimensional vectors.

```bash
python scripts/embed_data.py \
    --dataset BeIR/nq-generated-queries \
    --query_text_column query \
    --key_text_column text \
    --encoder all-MiniLM-L6-v2 \
    --output_path /tmp/amips/embeddings \
    --batch_size 256
```

Outputs (in `--output_path`):

```
queries.npy   # [n, d]
keys.npy      # [m, d]
```

## 2. Cluster keys, augment & split

Runs k-means on `keys.npy` (skipped when `--num_clusters 1`), augments every query
with `--num_noise_samples` renormalized noisy copies, computes the argmax key
*within each cluster* (remapped to a global index into `keys.npy`), then pools,
shuffles, and splits `--test_size` rows into the test set. The resulting
`topk_indices` array has shape `(num_samples, num_clusters)`.

```bash
python scripts/precompute_top1.py \
    --input_path /tmp/amips/embeddings \
    --output_path /tmp/amips/targets \
    --num_clusters 1 \
    --num_noise_samples 10 \
    --noise_std 0.02 \
    --test_size 16_384 \
    --batch_size 1024
```

Outputs (in `--output_path`):

```
keys.npy                      # copied from --input_path
centroids.npy                 # [num_clusters, d]
cluster_assignments.npy       # [m] int — cluster id per key
train/queries.npy             # [n_train, d]
train/topk_indices.npy        # [n_train, num_clusters]
test/queries.npy              # [n_test,  d]
test/topk_indices.npy         # [n_test,  num_clusters]
```

## 3. Train the approximator

`train_amips.py` is configured via a YAML file and the path to stage 2's output is
passed separately as `--input_path`.
See [`train_amips.yaml`](configs/train_amips.yaml) for the default config.

```bash
python scripts/train_amips.py \
    --input_path /tmp/amips/targets \
    --config configs/train_amips.yaml
```

For quicker experiments, [`train_amips_fiqa.yaml`](configs/train_amips_fiqa.yaml) and
[`train_amips_fiqa_router.yaml`](configs/train_amips_fiqa_router.yaml) are lighter configs
tuned for smaller datasets such as `BeIR/fiqa-generated-queries` (~56k keys); pass one of
them via `--config`. The [`quickstart_fiqa.ipynb`](notebooks/quickstart_fiqa.ipynb) notebook runs the
whole FIQA pipeline end to end.

## Citation

If you find our work useful, please consider citing us as:

```bibtex
@misc{
    olausson2026amips,
    title={Amortizing Maximum Inner Product Search with Learned Support Functions},
    author={Theo X. Olausson and João Monteiro and Michal Klein and Marco Cuturi},
    year={2026},
    eprint={2603.08001},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2603.08001},
}
```

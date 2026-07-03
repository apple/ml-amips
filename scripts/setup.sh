#!/usr/bin/env bash
set -euo pipefail

pip install -e .
pip install --force-reinstall "jax[cuda12]==0.8.2" "flax==0.12.2" "optax==0.2.6"

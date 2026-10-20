#!/usr/bin/env bash
set -euo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
root=/home/scv/shared/phase300_20261020
mkdir -p "$root/cache" "$root/tmp"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export HF_HOME="$root/cache" TORCH_HOME="$root/cache"
export XDG_CACHE_HOME="$root/cache" CUDA_CACHE_PATH="$root/cache"
export MPLCONFIGDIR="$root/cache" TMPDIR="$root/tmp"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -B src/tools/check_phase300.py

#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/light/cloud_adapter_dinov2_s_light_l1c.py}"
work_dir="${2:-work_dirs/cloud_adapter_dinov2_s_light_l1c}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

python tools/check_light_setup.py --config "${config}"

python tools/train.py \
  "${config}" \
  --work-dir "${work_dir}"

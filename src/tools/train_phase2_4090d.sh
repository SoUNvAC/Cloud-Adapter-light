#!/usr/bin/env bash
set -euo pipefail

experiment="${1:-v2}"
if (($# > 0)); then shift; fi

case "${experiment}" in
  v2)
    config="configs/light/cloud_adapter_dinov2_s_light_v2_l1c.py"
    work_dir="work_dirs/cloud_adapter_dinov2_s_light_v2_l1c"
    ;;
  control)
    config="configs/control/cloud_adapter_dinov2_s_mask2former_4x_l1c.py"
    work_dir="work_dirs/cloud_adapter_dinov2_s_mask2former_4x_l1c"
    ;;
  *)
    echo "Usage: bash tools/train_phase2_4090d.sh [v2|control]"
    exit 2
    ;;
esac

bash tools/train_light_4090d.sh "${config}" "${work_dir}" "$@"

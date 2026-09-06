#!/usr/bin/env bash
set -euo pipefail

experiment="${1:-fpn}"
if (($# > 0)); then shift; fi

case "${experiment}" in
  fpn)
    config="configs/light/cloud_adapter_dinov2_s_light_v3_fpn_l1c.py"
    work_dir="work_dirs/cloud_adapter_dinov2_s_light_v3_fpn_l1c"
    ;;
  fpn_aux)
    config="configs/light/cloud_adapter_dinov2_s_light_v3_fpn_aux_l1c.py"
    work_dir="work_dirs/cloud_adapter_dinov2_s_light_v3_fpn_aux_l1c"
    ;;
  *)
    echo "Usage: bash tools/train_phase3_4090d.sh [fpn|fpn_aux]"
    exit 2
    ;;
esac

bash tools/train_light_4090d.sh "${config}" "${work_dir}" "$@"

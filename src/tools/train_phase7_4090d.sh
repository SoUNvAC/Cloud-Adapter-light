#!/usr/bin/env bash
set -euo pipefail

config="configs/light/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_l1c.py"
work_dir="work_dirs/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_l1c"

bash tools/train_light_4090d.sh "${config}" "${work_dir}" "$@"

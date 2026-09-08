#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

config="configs/light/cloud_adapter_dinov2_s_mask2former_export_fpn_distill_l1c.py"
work_dir="work_dirs/cloud_adapter_dinov2_s_mask2former_export_fpn_distill_l1c"
student_dir="work_dirs/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c"
teacher_dir="work_dirs/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c"
student_checkpoints=("${student_dir}"/best_mIoU_iter_*.pth)
teacher_checkpoints=("${teacher_dir}"/best_mIoU_iter_*.pth)

if ((${#student_checkpoints[@]} != 1)); then
  echo "Expected one Phase 12 best checkpoint in ${student_dir}, found ${#student_checkpoints[@]}"
  exit 1
fi
if ((${#teacher_checkpoints[@]} != 1)); then
  echo "Expected one V8 best checkpoint in ${teacher_dir}, found ${#teacher_checkpoints[@]}"
  exit 1
fi

bash tools/train_light_4090d.sh \
  "${config}" \
  "${work_dir}" \
  --cfg-options \
  "load_from=${student_checkpoints[0]}" \
  "model.teacher_checkpoint=${teacher_checkpoints[0]}" \
  "$@"

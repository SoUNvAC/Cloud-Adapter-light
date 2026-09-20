#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

seed="${1:?seed is required}"
if [[ "${seed}" != "123" && "${seed}" != "3407" ]]; then
  echo "Phase 27 only trains preregistered seeds 123 and 3407" >&2
  exit 2
fi
config="configs/protocol/phase26_resnet18_v8_l1c.py"
work_dir="work_dirs/phase27_resnet18_3seed/seed${seed}"
pretrained="work_dirs/phase26_resnet18_pilot/pretrained/resnet18_v1c-b5776b93.pth"
pretrained_sha256="b5776b937a850b71ede79225265fabb5665b4641008e9b840929a94a919de707"
resume_args=()

echo "${pretrained_sha256}  ${pretrained}" | sha256sum --check --status || {
  echo "Missing or invalid Phase 26 ResNet-18 pretrained checkpoint" >&2
  exit 7
}
export PHASE26_RESNET18_PRETRAINED="${pretrained}"
if [[ -d "data/cloudsen12_high_l1c" ]]; then
  export CLOUD_ADAPTER_DATA_ROOT="data/cloudsen12_high_l1c"
elif [[ -d "../data/cloudsen12_high_l1c" ]]; then
  export CLOUD_ADAPTER_DATA_ROOT="../data/cloudsen12_high_l1c"
else
  echo "CloudSEN12 High L1C is missing" >&2
  exit 6
fi

if [[ -e "${work_dir}" ]]; then
  mapfile -t completed < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
  if [[ "${#completed[@]}" -eq 1 && -f "${work_dir}/val_eval.log" && -f "${work_dir}/VAL_EVAL_COMPLETE" ]]; then
    echo "Reusing completed Phase 27 seed ${seed}"
    exit 0
  fi
  if [[ -f "${work_dir}/last_checkpoint" ]]; then
    resume_args=(--resume)
  elif ! find "${work_dir}" -maxdepth 2 -type f -name '*.pth' -print -quit | grep -q .; then
    echo "Restarting zero-checkpoint Phase 27 seed ${seed}"
  else
    echo "Refusing checkpoint-bearing run without last_checkpoint" >&2
    exit 3
  fi
fi

bash tools/train_light_4090d.sh "${config}" "${work_dir}" "${resume_args[@]}" \
  --cfg-options "randomness.seed=${seed}" "randomness.deterministic=True"
mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "Expected exactly one best checkpoint for seed ${seed}" >&2
  exit 4
fi
python tools/test.py "${config}" "${checkpoints[0]}" \
  --work-dir "${work_dir}/val_eval" \
  --cfg-options \
  "test_dataloader.dataset.data_prefix.img_path=img_dir/val" \
  "test_dataloader.dataset.data_prefix.seg_map_path=ann_dir/val" \
  "randomness.seed=${seed}" "randomness.deterministic=True" \
  2>&1 | tee "${work_dir}/val_eval.log"
touch "${work_dir}/VAL_EVAL_COMPLETE"

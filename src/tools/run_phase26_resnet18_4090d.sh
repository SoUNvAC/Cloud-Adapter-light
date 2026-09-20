#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

root="work_dirs/phase26_resnet18_pilot"
work_dir="${root}/seed42"
config="configs/protocol/phase26_resnet18_v8_l1c.py"
phase22_config="configs/protocol/phase22_clean_v8_l1c.py"
phase22_summary="work_dirs/phase22_clean_v8/summary.json"
phase22_checkpoint="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth"
pretrained="${root}/pretrained/resnet18_v1c-b5776b93.pth"
pretrained_url="https://download.openmmlab.com/pretrain/third_party/resnet18_v1c-b5776b93.pth"
pretrained_sha256="b5776b937a850b71ede79225265fabb5665b4641008e9b840929a94a919de707"
resume_args=()

python - "${phase22_summary}" <<'PY'
import json
from pathlib import Path
import sys
row = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if row.get("phase") != 22 or row.get("passed") is not True or row.get("test_evaluated") is not False:
    raise SystemExit("Phase 26 requires the passed sealed-test Phase 22 anchor")
PY

mkdir -p "$(dirname "${pretrained}")"
if [[ ! -f "${pretrained}" ]]; then
  curl -L --fail --retry 3 --output "${pretrained}.part" "${pretrained_url}"
  mv "${pretrained}.part" "${pretrained}"
fi
echo "${pretrained_sha256}  ${pretrained}" | sha256sum --check --status || {
  echo "ResNet-18 pretrained checkpoint SHA-256 mismatch" >&2
  exit 7
}
export PHASE26_RESNET18_PRETRAINED="${pretrained}"

if [[ -d "data/cloudsen12_high_l1c" ]]; then
  export CLOUD_ADAPTER_DATA_ROOT="data/cloudsen12_high_l1c"
elif [[ -d "../data/cloudsen12_high_l1c" ]]; then
  export CLOUD_ADAPTER_DATA_ROOT="../data/cloudsen12_high_l1c"
else
  echo "CloudSEN12 High L1C is missing from src/data and repository data" >&2
  exit 6
fi

if [[ -e "${work_dir}" ]]; then
  mapfile -t completed < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
  if [[ "${#completed[@]}" -eq 1 && -f "${work_dir}/val_eval.log" && -f "${work_dir}/VAL_EVAL_COMPLETE" ]]; then
    echo "Reusing completed Phase 26 seed-42 pilot"
  elif [[ -f "${work_dir}/last_checkpoint" ]]; then
    echo "Resuming interrupted Phase 26 pilot"
    resume_args=(--resume)
  elif ! find "${work_dir}" -maxdepth 2 -type f -name '*.pth' -print -quit | grep -q .; then
    echo "Restarting zero-checkpoint Phase 26 directory"
  else
    echo "Refusing checkpoint-bearing pilot without last_checkpoint" >&2
    exit 2
  fi
fi

if [[ ! -f "${work_dir}/VAL_EVAL_COMPLETE" ]]; then
  bash tools/train_light_4090d.sh "${config}" "${work_dir}" "${resume_args[@]}" \
    --cfg-options "randomness.seed=42" "randomness.deterministic=True"
  mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
  if [[ "${#checkpoints[@]}" -ne 1 ]]; then
    echo "Expected exactly one Phase 26 best checkpoint, found ${#checkpoints[@]}" >&2
    exit 3
  fi
  python tools/test.py "${config}" "${checkpoints[0]}" \
    --work-dir "${work_dir}/val_eval" \
    --cfg-options \
    "test_dataloader.dataset.data_prefix.img_path=img_dir/val" \
    "test_dataloader.dataset.data_prefix.seg_map_path=ann_dir/val" \
    "randomness.seed=42" "randomness.deterministic=True" \
    2>&1 | tee "${work_dir}/val_eval.log"
  touch "${work_dir}/VAL_EVAL_COMPLETE"
fi

mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "Expected exactly one completed Phase 26 checkpoint" >&2
  exit 4
fi

if [[ ! -f "${root}/baseline_benchmark.json" ]]; then
  python tools/benchmark_deployment.py \
    --config "${phase22_config}" --checkpoint "${phase22_checkpoint}" \
    --batch-size 1 --input-size 512 --warmup 20 --iters 100 \
    --precision fp16 --weight-dtype fp16 --output-format json --quiet \
    | tee "${root}/baseline_benchmark.json"
fi
if [[ ! -f "${work_dir}/benchmark.json" ]]; then
  python tools/benchmark_deployment.py \
    --config "${config}" --checkpoint "${checkpoints[0]}" \
    --batch-size 1 --input-size 512 --warmup 20 --iters 100 \
    --precision fp16 --weight-dtype fp16 --output-format json --quiet \
    | tee "${work_dir}/benchmark.json"
fi

python tools/summarize_phase26_resnet18.py \
  --phase22-summary "${phase22_summary}" --root "${root}" \
  --min-miou 68.0 --min-speedup 1.25 --max-parameters-m 18.0 \
  --output "${root}/summary.json"

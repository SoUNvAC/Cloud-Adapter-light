#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

root="work_dirs/phase29_mobilenetv2_pilot"
work_dir="${root}/seed42"
config="configs/protocol/phase29_mobilenetv2_v8_l1c.py"
external_config="configs/protocol/phase29_l8_mobilenetv2_l1c.py"
phase26_summary="work_dirs/phase26_resnet18_pilot/summary.json"
phase28_summary="work_dirs/phase28_l8_external/summary.json"
pretrained="${root}/pretrained/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth"
converted_pretrained="${root}/pretrained/mobilenet_v2_backbone_only.pth"
pretrained_sha256="3b2dc3afee0b94e52b357a60851f1ac8ec95cf9318762e785812edf7f6736b14"
resume_args=()

python - "${phase26_summary}" "${phase28_summary}" <<'PY'
import json
from pathlib import Path
import sys
p26 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
p28 = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if p26.get("passed") is not True or p28.get("phase") != 28 or p28.get("passed") is not False:
    raise SystemExit("Phase 29 requires passed Phase 26 and failed Phase 28")
if p28.get("internal_test_evaluated") is not False:
    raise SystemExit("Internal test was not sealed")
PY
echo "${pretrained_sha256}  ${pretrained}" | sha256sum --check --status || {
  echo "Missing or invalid MobileNetV2 pretrained checkpoint" >&2
  exit 7
}
if [[ ! -f "${converted_pretrained}" ]]; then
  python tools/prepare_phase29_mobilenetv2_checkpoint.py \
    "${pretrained}" "${converted_pretrained}"
fi
export PHASE29_MOBILENETV2_PRETRAINED="${converted_pretrained}"
export PHASE28_L8_ROOT="data/l8_biome"
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
  if [[ "${#completed[@]}" -eq 1 && -f "${work_dir}/VAL_EVAL_COMPLETE" ]]; then
    echo "Reusing completed Phase 29 training"
  elif [[ -f "${work_dir}/last_checkpoint" ]]; then
    resume_args=(--resume)
  elif ! find "${work_dir}" -maxdepth 2 -type f -name '*.pth' -print -quit | grep -q .; then
    echo "Restarting zero-checkpoint Phase 29 directory"
  else
    echo "Refusing checkpoint-bearing run without last_checkpoint" >&2
    exit 2
  fi
fi

if [[ ! -f "${work_dir}/VAL_EVAL_COMPLETE" ]]; then
  bash tools/train_light_4090d.sh "${config}" "${work_dir}" "${resume_args[@]}" \
    --cfg-options "randomness.seed=42" "randomness.deterministic=True"
  mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
  if [[ "${#checkpoints[@]}" -ne 1 ]]; then exit 3; fi
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
if [[ "${#checkpoints[@]}" -ne 1 ]]; then exit 4; fi
if [[ ! -f "${work_dir}/benchmark.json" ]]; then
  python tools/benchmark_deployment.py \
    --config "${config}" --checkpoint "${checkpoints[0]}" \
    --batch-size 1 --input-size 512 --warmup 20 --iters 100 \
    --precision fp16 --weight-dtype fp16 --output-format json --quiet \
    | tee "${work_dir}/benchmark.json"
fi
if [[ ! -f "${work_dir}/L8_EVAL_COMPLETE" ]]; then
  python tools/test.py "${external_config}" "${checkpoints[0]}" \
    --work-dir "${work_dir}/l8_eval" \
    --cfg-options "randomness.seed=42" "randomness.deterministic=True" \
    2>&1 | tee "${work_dir}/l8_eval.log"
  touch "${work_dir}/L8_EVAL_COMPLETE"
fi

python tools/summarize_phase29_mobilenetv2.py \
  --phase26-summary "${phase26_summary}" --phase28-summary "${phase28_summary}" \
  --root "${root}" --min-val-miou 66.0 --min-external-miou 35.0 \
  --max-external-drop 6.5 --min-speedup 2.0 --max-parameters-m 7.5 \
  --output "${root}/summary.json"

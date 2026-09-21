#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

root="work_dirs/phase36_mobilenetv3_litefpn"
work_dir="${root}/seed42"
config="configs/protocol/phase36_mobilenetv3_litefpn_l1c.py"
pretrained="${root}/pretrained/mobilenet_v3_large_imagenet1k_v2_features.pth"
phase31_summary="work_dirs/phase31_mobilenetv2_litefpn/summary.json"
phase35_summary="work_dirs/phase35_bilitefpn/summary.json"
resume_args=()

python - "${phase31_summary}" "${phase35_summary}" <<'PY'
import json
from pathlib import Path
import sys
p31 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
p35 = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if p31.get("phase") != 31 or p31.get("passed") is not True:
    raise SystemExit("Phase 36 requires Phase 31 benchmark evidence")
if p35.get("phase") != 35 or p35.get("passed") is not False:
    raise SystemExit("Phase 36 requires the failed Phase 35 stop-loss")
if p35.get("internal_test_evaluated") is not False:
    raise SystemExit("CloudSEN internal test is not sealed")
PY

if [[ ! -f "${pretrained}" ]]; then
  export TORCH_HOME="${root}/pretrained/torch_home"
  python tools/prepare_phase36_mobilenetv3_checkpoint.py --output "${pretrained}"
fi
export PHASE36_MOBILENETV3_PRETRAINED="${pretrained}"
if [[ -d "data/cloudsen12_high_l1c" ]]; then
  export CLOUD_ADAPTER_DATA_ROOT="data/cloudsen12_high_l1c"
elif [[ -d "../data/cloudsen12_high_l1c" ]]; then
  export CLOUD_ADAPTER_DATA_ROOT="../data/cloudsen12_high_l1c"
else
  exit 6
fi

if [[ -e "${work_dir}" ]]; then
  mapfile -t completed < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
  if [[ "${#completed[@]}" -eq 1 && -f "${work_dir}/VAL_EVAL_COMPLETE" ]]; then
    echo "Reusing completed Phase 36 training"
  elif [[ -f "${work_dir}/last_checkpoint" ]]; then
    resume_args=(--resume)
  elif ! find "${work_dir}" -maxdepth 2 -type f -name '*.pth' -print -quit | grep -q .; then
    echo "Restarting zero-checkpoint Phase 36 directory"
  else
    echo "Refusing checkpoint-bearing run without last_checkpoint" >&2
    exit 2
  fi
fi

if [[ ! -f "${work_dir}/VAL_EVAL_COMPLETE" ]]; then
  bash tools/train_light_4090d.sh "${config}" "${work_dir}" "${resume_args[@]}" \
    --cfg-options "randomness.seed=42" "randomness.deterministic=True"
  mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
  if [[ "${#checkpoints[@]}" -ne 1 ]]; then exit 4; fi
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
if [[ "${#checkpoints[@]}" -ne 1 ]]; then exit 5; fi
if [[ ! -f "${work_dir}/benchmark.json" ]]; then
  python tools/benchmark_deployment.py \
    --config "${config}" --checkpoint "${checkpoints[0]}" \
    --batch-size 1 --input-size 512 --warmup 20 --iters 100 \
    --precision fp16 --weight-dtype fp16 --output-format json --quiet \
    | tee "${work_dir}/benchmark.json"
fi
python tools/summarize_phase36_mobilenetv3_litefpn.py \
  --phase31-summary "${phase31_summary}" --root "${root}" \
  --min-miou 68.0 --min-weak-miou 51.0 --min-speedup 2.0 \
  --max-parameters-m 4.0 --output "${root}/summary.json"

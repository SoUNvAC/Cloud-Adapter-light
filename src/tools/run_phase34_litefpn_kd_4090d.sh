#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

root="work_dirs/phase34_litefpn_kd"
work_dir="${root}/seed42"
config="configs/protocol/phase34_litefpn_weak_boundary_kd_l1c.py"
inference_config="configs/protocol/phase31_mobilenetv2_litefpn_l1c.py"
phase31_summary="work_dirs/phase31_mobilenetv2_litefpn/summary.json"
phase33_summary="work_dirs/phase33_litefpn_3seed/summary.json"
student_dir="work_dirs/phase31_mobilenetv2_litefpn/seed42"
teacher_dir="work_dirs/phase22_clean_v8/seed42"
pretrained="work_dirs/phase29_mobilenetv2_pilot/pretrained/mobilenet_v2_backbone_only.pth"
resume_args=()

python - "${phase31_summary}" "${phase33_summary}" <<'PY'
import json
from pathlib import Path
import sys
p31 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
p33 = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if p31.get("phase") != 31 or p31.get("passed") is not True:
    raise SystemExit("Phase 34 requires passed Phase 31")
if p33.get("phase") != 33 or p33.get("passed") is not False:
    raise SystemExit("Phase 34 requires the failed Phase 33 stop-loss")
if p33.get("internal_test_evaluated") is not False:
    raise SystemExit("CloudSEN internal test is not sealed")
PY
mapfile -t student_checkpoints < <(find "${student_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
mapfile -t teacher_checkpoints < <(find "${teacher_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
if [[ "${#student_checkpoints[@]}" -ne 1 || "${#teacher_checkpoints[@]}" -ne 1 ]]; then
  echo "Expected one Phase 31 student and one Phase 22 teacher checkpoint" >&2
  exit 3
fi
export PHASE29_MOBILENETV2_PRETRAINED="${pretrained}"
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
    echo "Reusing completed Phase 34 training"
  elif [[ -f "${work_dir}/last_checkpoint" ]]; then
    resume_args=(--resume)
  elif ! find "${work_dir}" -maxdepth 2 -type f -name '*.pth' -print -quit | grep -q .; then
    echo "Restarting zero-checkpoint Phase 34 directory"
  else
    echo "Refusing checkpoint-bearing run without last_checkpoint" >&2
    exit 2
  fi
fi

if [[ ! -f "${work_dir}/VAL_EVAL_COMPLETE" ]]; then
  bash tools/train_light_4090d.sh "${config}" "${work_dir}" "${resume_args[@]}" \
    --cfg-options "load_from=${student_checkpoints[0]}" \
    "model.teacher_checkpoint=${teacher_checkpoints[0]}" \
    "randomness.seed=42" "randomness.deterministic=True"
  mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
  if [[ "${#checkpoints[@]}" -ne 1 ]]; then exit 4; fi
  python tools/test.py "${inference_config}" "${checkpoints[0]}" \
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
    --config "${inference_config}" --checkpoint "${checkpoints[0]}" \
    --batch-size 1 --input-size 512 --warmup 20 --iters 100 \
    --precision fp16 --weight-dtype fp16 --output-format json --quiet \
    | tee "${work_dir}/benchmark.json"
fi
python tools/summarize_phase34_litefpn_kd.py \
  --phase31-summary "${phase31_summary}" --root "${root}" \
  --min-miou 68.0 --min-improvement 1.5 --min-weak-miou 51.0 \
  --min-speedup 2.0 --max-parameters-m 3.0 --output "${root}/summary.json"

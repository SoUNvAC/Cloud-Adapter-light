#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

root="work_dirs/phase52_sparse_msre_1pct"
config="configs/protocol/phase52_sparse_msre_1pct_l1c.py"
work_dir="${root}/seed52"
baseline_checkpoint="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth"
mkdir -p "${root}"

python - <<'PY'
import json
from pathlib import Path
row = json.loads(Path("work_dirs/phase51_trust_audit/summary.json").read_text())
if row.get("decision") != "end_tgrs_main_method_route":
    raise SystemExit("Phase 51 terminal decision changed unexpectedly")
selection = json.loads(Path("work_dirs/phase50_active_1pct/selection.json").read_text())
if selection.get("selected_images") != 65:
    raise SystemExit("Frozen Phase 50 1% selection is unavailable")
PY

python tools/audit_phase52_sparse_msre.py

if [[ ! -f "${work_dir}/TRAIN_COMPLETE" ]]; then
  resume_args=()
  if [[ -f "${work_dir}/last_checkpoint" ]]; then
    resume_args=(--resume)
  fi
  bash tools/train_light_4090d.sh "${config}" "${work_dir}" "${resume_args[@]}"
  touch "${work_dir}/TRAIN_COMPLETE"
fi

mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "Expected one best Phase 52 checkpoint, found ${#checkpoints[@]}" >&2
  exit 3
fi

python tools/benchmark_deployment.py \
  --config configs/protocol/phase22_clean_v8_l1c.py \
  --checkpoint "${baseline_checkpoint}" --precision fp16 --weight-dtype fp16 \
  --batch-size 1 --input-size 512 --warmup 50 --iters 200 --output-format json \
  > "${root}/baseline_benchmark.json"
python tools/benchmark_deployment.py \
  --config "${config}" --checkpoint "${checkpoints[0]}" \
  --precision fp16 --weight-dtype fp16 --batch-size 1 --input-size 512 \
  --warmup 50 --iters 200 --output-format json \
  > "${root}/candidate_benchmark.json"

python tools/evaluate_phase52_sparse_msre.py --checkpoint "${checkpoints[0]}"

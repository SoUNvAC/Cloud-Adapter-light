#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

config="configs/protocol/phase22_clean_v8_l1c.py"
phase22_root="work_dirs/phase22_clean_v8"
phase22_summary="${phase22_root}/summary.json"
checkpoint_dir="${phase22_root}/seed42"
output_root="work_dirs/phase24_block_screen"

python - "${phase22_summary}" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("phase") != 22 or not summary.get("passed"):
    raise SystemExit("Phase 22 clean V8 anchor must pass before Phase 24")
if summary.get("test_evaluated") is not False:
    raise SystemExit("Phase 22 did not keep test sealed")
PY

mapfile -t checkpoints < <(find "${checkpoint_dir}" -maxdepth 1 -type f \
  -name 'best_mIoU_iter_*.pth' -print)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "Expected exactly one Phase 22 seed42 checkpoint" >&2
  exit 2
fi
checkpoint="${checkpoints[0]}"

names=(baseline12 blocks10 blocks8 blocks6 blocks4)
subsets=(
  "0,1,2,3,4,5,6,7,8,9,10,11"
  "0,2,3,4,5,6,8,9,10,11"
  "0,2,4,5,7,8,10,11"
  "0,2,5,8,10,11"
  "2,5,8,11"
)

for index in "${!names[@]}"; do
  name="${names[index]}"
  subset="${subsets[index]}"
  candidate_dir="${output_root}/${name}"
  if [[ -f "${candidate_dir}/SCREEN_COMPLETE" ]]; then
    echo "Reusing completed Phase 24 candidate: ${name}"
    continue
  fi
  mkdir -p "${candidate_dir}"

  python tools/test.py \
    "${config}" \
    "${checkpoint}" \
    --work-dir "${candidate_dir}/val_eval" \
    --cfg-options \
    "model.backbone.active_block_indices=[${subset}]" \
    "test_dataloader.dataset.data_prefix.img_path=img_dir/val" \
    "test_dataloader.dataset.data_prefix.seg_map_path=ann_dir/val" \
    "randomness.seed=42" \
    "randomness.deterministic=True" \
    2>&1 | tee "${candidate_dir}/val_eval.log"

  IFS=',' read -r -a block_indices <<< "${subset}"
  python tools/benchmark_deployment.py \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --active-block-indices "${block_indices[@]}" \
    --batch-size 1 \
    --input-size 512 \
    --warmup 20 \
    --iters 100 \
    --precision fp16 \
    --weight-dtype fp16 \
    --output-format json \
    --quiet \
    | tee "${candidate_dir}/benchmark.json"
  touch "${candidate_dir}/SCREEN_COMPLETE"
done

python tools/summarize_phase24_block_screen.py \
  --phase22-summary "${phase22_summary}" \
  --root "${output_root}" \
  --min-speedup 1.15 \
  --max-val-miou-drop 8.0 \
  --output "${output_root}/summary.json"

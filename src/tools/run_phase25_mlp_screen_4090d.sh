#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

config="configs/protocol/phase25_mlp_pruned_v8_l1c.py"
phase22_summary="work_dirs/phase22_clean_v8/summary.json"
source_checkpoint="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth"
output_root="work_dirs/phase25_mlp_screen"
names=(ratio4 ratio3 ratio2p5 ratio2)
ratios=(4.0 3.0 2.5 2.0)

python - "${phase22_summary}" <<'PY'
import json
from pathlib import Path
import sys
row = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if row.get("phase") != 22 or row.get("passed") is not True or row.get("test_evaluated") is not False:
    raise SystemExit("Phase 25 requires the passed sealed-test Phase 22 anchor")
PY

if [[ ! -f "${source_checkpoint}" ]]; then
  echo "Missing Phase 22 seed-42 checkpoint" >&2
  exit 2
fi

for index in "${!names[@]}"; do
  name="${names[index]}"
  ratio="${ratios[index]}"
  candidate_dir="${output_root}/${name}"
  checkpoint="${candidate_dir}/model.pth"
  if [[ -f "${candidate_dir}/SCREEN_COMPLETE" ]]; then
    echo "Reusing completed Phase 25 candidate: ${name}"
    continue
  fi
  mkdir -p "${candidate_dir}"
  python tools/prepare_phase25_mlp_checkpoint.py \
    "${source_checkpoint}" "${checkpoint}" --target-ratio "${ratio}"

  export PHASE25_MLP_RATIO="${ratio}"
  python tools/test.py "${config}" "${checkpoint}" \
    --work-dir "${candidate_dir}/val_eval" \
    --cfg-options \
    "test_dataloader.dataset.data_prefix.img_path=img_dir/val" \
    "test_dataloader.dataset.data_prefix.seg_map_path=ann_dir/val" \
    "randomness.seed=42" "randomness.deterministic=True" \
    2>&1 | tee "${candidate_dir}/val_eval.log"

  python tools/benchmark_deployment.py \
    --config "${config}" --checkpoint "${checkpoint}" \
    --batch-size 1 --input-size 512 --warmup 20 --iters 100 \
    --precision fp16 --weight-dtype fp16 --output-format json --quiet \
    | tee "${candidate_dir}/benchmark.json"
  touch "${candidate_dir}/SCREEN_COMPLETE"
done

python tools/summarize_phase25_mlp_screen.py \
  --phase22-summary "${phase22_summary}" --root "${output_root}" \
  --min-speedup 1.08 --max-val-miou-drop 5.0 \
  --output "${output_root}/summary.json"

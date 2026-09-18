#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

seed="${1:-42}"
config="configs/protocol/phase23_clean_v12_l1c.py"
phase22_root="work_dirs/phase22_clean_v8"
phase22_summary="${phase22_root}/summary.json"
work_dir="work_dirs/phase23_clean_v12/seed${seed}"

python - "${phase22_summary}" "${seed}" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
seed = int(sys.argv[2])
if not summary.get("passed") or summary.get("test_evaluated") is not False:
    raise SystemExit("Phase 22 has not passed the sealed-test validation gate")
if seed not in {int(run["seed"]) for run in summary.get("runs", [])}:
    raise SystemExit(f"Seed {seed} is absent from the Phase 22 summary")
PY

if [[ -e "${work_dir}" ]]; then
  echo "Refusing to reuse existing run directory: ${work_dir}" >&2
  exit 2
fi

mapfile -t source_checkpoints < <(find "${phase22_root}/seed${seed}" \
  -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
if [[ "${#source_checkpoints[@]}" -ne 1 ]]; then
  echo "Expected exactly one Phase 22 checkpoint, found ${#source_checkpoints[@]}" >&2
  exit 3
fi

bash tools/train_light_4090d.sh \
  "${config}" \
  "${work_dir}" \
  --cfg-options \
  "load_from=${source_checkpoints[0]}" \
  "randomness.seed=${seed}" \
  "randomness.deterministic=True"

mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f \
  -name 'best_mIoU_iter_*.pth' -print)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "Expected exactly one best checkpoint, found ${#checkpoints[@]}" >&2
  exit 4
fi

# Keep test sealed by redirecting the test loop to the validation partition.
python tools/test.py \
  "${config}" \
  "${checkpoints[0]}" \
  --work-dir "${work_dir}/val_eval" \
  --cfg-options \
  "test_dataloader.dataset.data_prefix.img_path=img_dir/val" \
  "test_dataloader.dataset.data_prefix.seg_map_path=ann_dir/val" \
  "randomness.seed=${seed}" \
  "randomness.deterministic=True" \
  2>&1 | tee "${work_dir}/val_eval.log"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

method="${1:?method must be frozen, rein, or cloud_adapter}"
seed="${2:-42}"
case "${method}" in
  frozen|rein|cloud_adapter) ;;
  *) echo "Unknown Phase 29 method: ${method}" >&2; exit 2 ;;
esac

audit="work_dirs/phase29_fair_baselines/audit.json"
python - "${audit}" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not result.get("passed") or result.get("test_evaluated") is not False:
    raise SystemExit("Phase 29 audit must pass with test sealed before training")
PY

config="configs/protocol/phase29_${method}_v12_l1c.py"
work_dir="work_dirs/phase29_fair_baselines/${method}/seed${seed}"
if [[ -e "${work_dir}" ]]; then
  echo "Refusing to reuse existing run directory: ${work_dir}" >&2
  exit 3
fi

bash tools/train_light_4090d.sh \
  "${config}" \
  "${work_dir}" \
  --cfg-options \
  "randomness.seed=${seed}" \
  "randomness.deterministic=True"

mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f \
  -name 'best_mIoU_iter_*.pth' -print)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "Expected exactly one best checkpoint, found ${#checkpoints[@]}" >&2
  exit 4
fi

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

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python tools/prepare_cloudsen12_l1c.py --data-dir data
mkdir -p work_dirs/phase21_protocol
python tools/audit_cloudsen12_protocol.py \
  --data-root data/cloudsen12_high_l1c \
  --output work_dirs/phase21_protocol/cloudsen12_high_l1c_manifest.json \
  --max-class-tv 0.10 \
  | tee work_dirs/phase21_protocol/audit.log

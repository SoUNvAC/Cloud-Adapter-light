#!/usr/bin/env bash
set -euo pipefail

cd /home/scv/Cloud-Adapter-light/src
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210

raw_root="${1:?usage: bash tools/run_phase61d2_4090d.sh /authorized/path/containing/l8biome}"
out="work_dirs/phase61d2_calibrated_review"
mkdir -p "${out}"

python -m unittest tools/test_phase61d2_protocol.py 2>&1 | tee "${out}/unit_test_console.log"
python -u tools/prepare_phase61d2_calibrated_review.py \
  --raw-root "${raw_root}" \
  2>&1 | tee "${out}/prepare_console.log"

python -m json.tool "${out}/packet_summary.json"
sha256sum \
  "${out}/sealed_manifest.json" \
  "${out}/packet_summary.json" \
  research_plans/PHASE61D2_REVIEW_MANUAL.md \
  > "${out}/SHA256SUMS.txt"
touch "${out}/PHASE61D2_PACKET_COMPLETE"

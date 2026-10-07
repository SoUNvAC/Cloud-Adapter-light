#!/usr/bin/env bash
set -uo pipefail
cd /home/scv/Cloud-Adapter-light || exit 1
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210 || exit 1
logs=/home/scv/shared/data/sentinel2_alcd_1460961/run_logs
mkdir -p "$logs"
echo $$ > "$logs/PID"
python -u src/tools/download_phase65_alcd.py --root /home/scv/shared/data/sentinel2_alcd_1460961 > "$logs/console.log" 2> "$logs/error.log"
code=$?
if [ "$code" -eq 0 ]; then
    python src/tools/audit_phase65_alcd.py --root /home/scv/shared/data/sentinel2_alcd_1460961 --output "$logs/catalogue_audit.json" >> "$logs/console.log" 2>> "$logs/error.log"
    code=$?
fi
echo "$code" > "$logs/EXIT_CODE"
exit "$code"

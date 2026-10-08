#!/usr/bin/env bash
set -uo pipefail
cd /home/scv/Cloud-Adapter-light || exit 1
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210 || exit 1
root=/home/scv/shared/data/sentinel2_alcd_1460961/imagery
logs="$root/run_logs"
mkdir -p "$logs"
echo $$ > "$logs/PID"
python -u src/tools/acquire_phase65_alcd_imagery.py download --root "$root" > "$logs/console.log" 2> "$logs/error.log"
code=$?
echo "$code" > "$logs/DOWNLOAD_EXIT_CODE"
if [ "$code" -eq 0 ]; then
    python src/tools/audit_phase65_alcd_imagery.py --archive /home/scv/shared/data/sentinel2_alcd_1460961/SENTINEL_2_reference_cloud_masks_Baetens_Hagolle.tgz --root "$root" --output "$root/geometry_audit.json" >> "$logs/console.log" 2>> "$logs/error.log"
    code=$?
fi
echo "$code" > "$logs/EXIT_CODE"
exit "$code"

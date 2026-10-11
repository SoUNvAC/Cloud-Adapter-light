#!/usr/bin/env bash
set -uo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
control=/home/scv/shared/phase79_control_20261011
echo $$ > "$control/PID"
git rev-parse HEAD > "$control/GIT_COMMIT"
python src/tools/test_phase79.py
code=$?
if [ "$code" -eq 0 ]; then
  python -u src/tools/run_phase79.py \
    --data /home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871 \
    --previous /home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase78_20261011_retry01 \
    --bindings src/research_plans/PHASE79_INPUT_BINDINGS_20261011.json \
    --output /home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase79_20261011
  code=$?
fi
echo "$code" > "$control/EXIT_CODE"
exit "$code"

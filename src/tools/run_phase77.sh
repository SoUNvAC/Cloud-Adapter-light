#!/usr/bin/env bash
set -uo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
control=/home/scv/shared/phase77_control_20261011
echo $$ > "$control/PID"
git rev-parse HEAD > "$control/GIT_COMMIT"
python src/tools/test_phase77.py
code=$?
if [ "$code" -eq 0 ]; then
  python -u src/tools/run_phase77.py \
    --data /home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871 \
    --previous /home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase76_20261010 \
    --bindings src/research_plans/PHASE77_INPUT_BINDINGS_20261011.json \
    --output /home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase77_20261011
  code=$?
fi
echo "$code" > "$control/EXIT_CODE"
exit "$code"

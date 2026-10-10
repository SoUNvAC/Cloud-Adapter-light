#!/usr/bin/env bash
set -uo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
control=/home/scv/shared/phase76_control_20261010
echo $$ > "$control/PID"
git rev-parse HEAD > "$control/GIT_COMMIT"
python src/tools/test_phase76.py || exit $?
python -u src/tools/run_phase76.py \
  --data /home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871 \
  --phase75 /home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase75_20261010 \
  --bindings src/research_plans/PHASE76_INPUT_BINDINGS_20261010.json \
  --candidates "$control/cube_mapping_candidates.csv" \
  --output /home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase76_20261010
code=$?
echo "$code" > "$control/EXIT_CODE"
exit "$code"

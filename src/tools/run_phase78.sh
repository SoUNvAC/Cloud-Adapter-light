#!/usr/bin/env bash
set -uo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
control=/home/scv/shared/phase78_control_20261011
export PYTHONPATH="$control/dependencies${PYTHONPATH:+:$PYTHONPATH}"
echo $$ > "$control/PID"
git rev-parse HEAD > "$control/GIT_COMMIT"
python src/tools/test_phase78.py
code=$?
if [ "$code" -eq 0 ]; then
  python -u src/tools/run_phase78.py \
    --data /home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871 \
    --bindings src/research_plans/PHASE78_INPUT_BINDINGS_20261011.json \
    --output /home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase78_20261011
  code=$?
fi
echo "$code" > "$control/EXIT_CODE"
exit "$code"

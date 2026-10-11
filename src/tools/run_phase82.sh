#!/usr/bin/env bash
set -uo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 XFORMERS_DISABLED=1
control=/home/scv/shared/phase82_control_20261011
echo $$ > "$control/PID"
git rev-parse HEAD > "$control/GIT_COMMIT"
python src/tools/test_phase82.py && python -u src/tools/run_phase82_analysis.py --baseline && python -u src/tools/run_phase82_training.py
code=$?
echo "$code" > "$control/TRAINING_EXIT_CODE"
if [ "$code" -eq 0 ]; then
  python -u src/tools/run_phase82_analysis.py && python src/tools/run_phase82_report.py
  code=$?
fi
echo "$code" > "$control/EXIT_CODE"
exit "$code"

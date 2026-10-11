#!/usr/bin/env bash
set -uo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
export XFORMERS_DISABLED=1 OMP_NUM_THREADS=4
control=/home/scv/shared/phase80_control_20261011
echo $$ > "$control/PID"
git rev-parse HEAD > "$control/GIT_COMMIT"
python src/tools/test_phase80.py
code=$?
if [ "$code" -eq 0 ]; then
  python -u src/tools/run_phase80_analysis.py --baseline
  code=$?
fi
if [ "$code" -eq 0 ]; then
  python -u src/tools/run_phase80_training.py
  code=$?
fi
echo "$code" > "$control/TRAINING_EXIT_CODE"
if [ "$code" -eq 0 ]; then
  python -u src/tools/run_phase80_analysis.py
  code=$?
fi
echo "$code" > "$control/EXIT_CODE"
exit "$code"

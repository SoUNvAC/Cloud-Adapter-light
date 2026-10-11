#!/usr/bin/env bash
set -uo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
export XFORMERS_DISABLED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
control=/home/scv/shared/phase89_control_20261011_run02
echo $$ > "$control/PID"
git rev-parse HEAD > "$control/GIT_COMMIT"
python src/tools/test_phase89.py
code=$?
for step in freeze prepare Source MsRE shallow analyze verify; do
  if [ "$code" -eq 0 ]; then
    python -u src/tools/run_phase89.py "$step"
    code=$?
    echo "$code" > "$control/${step}_EXIT_CODE"
  fi
done
echo "$code" > "$control/EXIT_CODE"
exit "$code"

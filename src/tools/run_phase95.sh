#!/usr/bin/env bash
set -uo pipefail
cd /home/scv/Cloud-Adapter-light
control=/home/scv/shared/phase95_control_20261012
echo $$ > "$control/PID"
git rev-parse HEAD > "$control/GIT_COMMIT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
/home/scv/shared/phase93_environment_20261012/venv/bin/python -u src/tools/run_phase95.py
code=$?
echo "$code" > "$control/EXIT_CODE"
exit "$code"

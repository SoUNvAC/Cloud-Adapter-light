#!/usr/bin/env bash
set -uo pipefail
cd /home/scv/Cloud-Adapter-light
control=/home/scv/shared/phase94_control_20261012
echo $$ > "$control/RESUME_PID"
git rev-parse HEAD > "$control/RESUME_GIT_COMMIT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
/home/scv/shared/phase93_environment_20261012/venv/bin/python -u src/tools/resume_phase94.py --authorization "$1"
code=$?
echo "$code" > "$control/RESUME_EXIT_CODE"
exit "$code"

#!/usr/bin/env bash
set -uo pipefail
cd /home/scv/Cloud-Adapter-light
control=/home/scv/shared/phase93_control_20261012
echo $$ > "$control/PID"
git rev-parse HEAD > "$control/GIT_COMMIT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
bash src/tools/setup_phase93_env.sh && /home/scv/shared/phase93_environment_20261012/venv/bin/python -u src/tools/run_phase93.py preflight && /home/scv/shared/phase93_environment_20261012/venv/bin/python -u src/tools/run_phase93.py fit && /home/scv/shared/phase93_environment_20261012/venv/bin/python -u src/tools/run_phase93.py evaluate && /home/scv/shared/phase93_environment_20261012/venv/bin/python src/tools/verify_phase93_delivery.py --root /home/scv/shared/phase93_20261012/delivery
code=$?
echo "$code" > "$control/EXIT_CODE"
exit "$code"

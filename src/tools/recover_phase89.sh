#!/usr/bin/env bash
set -uo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
export XFORMERS_DISABLED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
control=/home/scv/shared/phase89_control_recovered_20261011
echo $$ > "$control/PID"
git rev-parse HEAD > "$control/GIT_COMMIT"
python -u src/tools/recover_phase89.py
code=$?
echo "$code" > "$control/EXIT_CODE"
exit "$code"

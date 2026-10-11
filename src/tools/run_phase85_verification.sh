#!/usr/bin/env bash
set -uo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
control=/home/scv/shared/phase85_verification_20261011
echo $$ > "$control/PID"
git rev-parse HEAD > "$control/GIT_COMMIT"
python -u src/tools/verify_phase85_completion.py
code=$?
echo "$code" > "$control/EXIT_CODE"
exit "$code"

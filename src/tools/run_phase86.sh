#!/usr/bin/env bash
set -uo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
control=/home/scv/shared/phase86_control_20261011
echo $$ > "$control/PID"
git rev-parse HEAD > "$control/GIT_COMMIT"
python src/tools/test_phase86.py && python -u src/tools/audit_phase86.py
code=$?
echo "$code" > "$control/EXIT_CODE"
exit "$code"

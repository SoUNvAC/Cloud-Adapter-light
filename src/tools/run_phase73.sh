#!/usr/bin/env bash
set -u
control=/home/scv/shared/phase73_control_20261010
mkdir "$control" || exit 1
exec >"$control/console.log" 2>"$control/error.log"
echo $$ >"$control/PID"
cd /home/scv/Cloud-Adapter-light || exit 1
git rev-parse HEAD >"$control/GIT_COMMIT"
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
python src/tools/run_phase73.py
result=$?
echo "$result" >"$control/EXIT_CODE"
exit "$result"

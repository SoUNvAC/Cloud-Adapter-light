#!/usr/bin/env bash
set -euo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
logs=/home/scv/shared/phase107_20261013_logs
test -d "$logs"
test ! -e "$logs/PID"
echo $$ > "$logs/PID"
trap 'code=$?; printf "%s\n" "$code" > "$logs/EXIT_CODE"' EXIT
python -u src/tools/run_phase107.py

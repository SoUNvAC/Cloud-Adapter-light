#!/usr/bin/env bash
set -euo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
export GDAL_NUM_THREADS=1 OPJ_NUM_THREADS=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
root=/home/scv/shared/data/pixbox_s2_cmix_5036991/phase105_20261013
test -d "$root/run_logs"
test ! -e "$root/run_logs/PID"
echo $$ > "$root/run_logs/PID"
trap 'code=$?; printf "%s\n" "$code" > "$root/run_logs/EXIT_CODE"' EXIT
python -u src/tools/run_phase105_source.py
/home/scv/shared/phase93_environment_20261012/venv/bin/python -u src/tools/run_phase105_trees.py

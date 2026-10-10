#!/usr/bin/env bash
set -euo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
export XFORMERS_DISABLED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
control="${1:-/home/scv/shared/phase72_control_20261010}"
case "$control" in /home/scv/shared/phase72_control_20261010*) ;; *) exit 2 ;; esac
test ! -e "$control/PID"
test ! -d /home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase72_20261010
echo $$ > "$control/PID"
trap 'code=$?; echo "$code" > "$control/EXIT_CODE"' EXIT
git rev-parse HEAD > "$control/GIT_SHA"
python src/tools/test_phase72.py
python src/tools/run_phase72.py prepare
python src/tools/run_phase72.py extract --model source
python src/tools/run_phase72.py extract --model msre
python src/tools/run_phase72.py evaluate

#!/usr/bin/env bash
set -euo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
export XFORMERS_DISABLED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
control=/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase66_control_20261010
bundle="$control/frozen"
groups="$1"
test -d "$bundle"
test ! -e "$control/PID"
test ! -d /home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase66_20261010
# Run this launcher itself under nohup; do not detach untracked children or restart automatically.
echo $$ > "$control/PID"
trap 'code=$?; echo "$code" > "$control/EXIT_CODE"' EXIT
python src/tools/test_phase66.py
python src/tools/run_phase66.py prepare --bundle "$bundle" --groups "$groups"
python src/tools/run_phase66.py extract --model source
python src/tools/run_phase66.py extract --model msre
python src/tools/run_phase66.py evaluate

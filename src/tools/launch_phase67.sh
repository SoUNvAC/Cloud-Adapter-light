#!/usr/bin/env bash
set -euo pipefail
root=/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase67_20261010
test ! -e "$root"
mkdir "$root"
mkdir "$root/run_logs" "$root/delivery"
cd /home/scv/Cloud-Adapter-light
git rev-parse HEAD > "$root/run_logs/GIT_SHA"
export PHASE67_ROOT="$root"
nohup bash -c '
 source /home/scv/miniconda3/etc/profile.d/conda.sh || exit 8
 conda activate cloud-lite-pt210 || exit 8
 cd /home/scv/Cloud-Adapter-light || exit 8
 export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
 trap '\''code=$?; echo "$code" > "$PHASE67_ROOT/run_logs/EXIT_CODE"'\'' EXIT
 python -u src/tools/test_phase67.py || exit $?
 python -u src/tools/test_phase65d_recovery.py || exit $?
 python -u src/tools/run_phase67.py fit || exit $?
 python -u src/tools/run_phase67.py evaluate || exit $?
' > "$root/run_logs/console.log" 2> "$root/run_logs/error.log" < /dev/null &
echo "$!" > "$root/run_logs/PID"
echo "Phase67 PID=$!"

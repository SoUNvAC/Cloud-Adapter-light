#!/usr/bin/env bash
set -uo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
control=/home/scv/shared/phase89_control_20261011_run02
set -o noclobber
echo $$ > "$control/FINALIZER_PID" || exit 1
while [ ! -f "$control/EXIT_CODE" ]; do
  if ! kill -0 "$(cat "$control/PID")" 2>/dev/null; then
    echo 99 > "$control/FINALIZE_EXIT_CODE"
    exit 99
  fi
  sleep 10
done
code=$(cat "$control/EXIT_CODE")
if [ "$code" -eq 0 ]; then
  python -u src/tools/finalize_phase89.py
  code=$?
fi
echo "$code" > "$control/FINALIZE_EXIT_CODE"
exit "$code"

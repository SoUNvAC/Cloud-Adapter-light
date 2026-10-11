#!/usr/bin/env bash
set -uo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
export XFORMERS_DISABLED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
control=/home/scv/shared/phase85_control_20261011
echo $$ > "$control/PID"
git rev-parse HEAD > "$control/GIT_COMMIT"
python src/tools/test_phase85.py && python -u src/tools/run_phase85_analysis.py baseline
code=$?
for seed in 66 67; do
  if [ "$code" -eq 0 ]; then
    export PHASE85_SEED="$seed"
    python -u src/tools/run_phase85_training.py
    code=$?
    echo "$code" > "$control/TRAINING_SEED${seed}_EXIT_CODE"
    if [ "$code" -eq 0 ]; then
      python -u src/tools/run_phase85_analysis.py seed
      code=$?
    fi
    echo "$code" > "$control/SEED${seed}_EXIT_CODE"
  fi
done
if [ "$code" -eq 0 ]; then
  python -u src/tools/run_phase85_analysis.py report && python -u src/tools/verify_phase85.py
  code=$?
fi
echo "$code" > "$control/EXIT_CODE"
exit "$code"

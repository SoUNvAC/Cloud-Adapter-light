#!/usr/bin/env bash
set -euo pipefail
repo=/home/scv/Cloud-Adapter-light
root=/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase65d_recovery_20261008
test ! -e "$root" || { echo 'Preserve existing recovery experiment; inspect PID first'; exit 2; }
mkdir "$root"
mkdir "$root/run_logs"
git -C "$repo" rev-parse HEAD > "$root/run_logs/GIT_SHA"
sha256sum "$repo/src/tools/run_phase65d_recovery.py" "$repo/src/tools/phase65d_recovery_math.py" "$repo/src/research_plans/PHASE65D_RECOVERY_20261008.md" > "$root/run_logs/CODE_SHA256"
export PHASE65_RECOVERY_ROOT="$root"
nohup bash -c '
 source /home/scv/miniconda3/etc/profile.d/conda.sh || exit 8
 conda activate cloud-lite-pt210 || exit 8
 cd /home/scv/Cloud-Adapter-light/src || exit 8
 export XFORMERS_DISABLED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
 python -u tools/run_phase65d_recovery.py extract --root "$PHASE65_RECOVERY_ROOT" --model source
 code=$?
 if test "$code" = 0; then python -u tools/run_phase65d_recovery.py extract --root "$PHASE65_RECOVERY_ROOT" --model msre; code=$?; fi
 if test "$code" = 0; then python -u tools/run_phase65d_recovery.py fit --root "$PHASE65_RECOVERY_ROOT"; code=$?; fi
 if test "$code" = 0; then python -u tools/run_phase65d_recovery.py evaluate --root "$PHASE65_RECOVERY_ROOT"; code=$?; fi
 printf "%s\n" "$code" > "$PHASE65_RECOVERY_ROOT/run_logs/EXIT_CODE"
' > "$root/run_logs/console.log" 2> "$root/run_logs/error.log" < /dev/null &
printf '%s\n' "$!" > "$root/run_logs/PID"
echo "Started recovery PID=$!"

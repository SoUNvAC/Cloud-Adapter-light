#!/usr/bin/env bash
set -euo pipefail
root=/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase65d_followup_20261008_recovery01
test "$(cat "$root/run_logs/EXIT_CODE")" = 1
! kill -0 "$(cat "$root/run_logs/PID")" 2>/dev/null
test -f "$root/development_val_3500/index.json"
test -f "$root/development_val_4000/index.json"
test ! -e "$root/development_report.json"
logs="$root/run_logs/analysis_resume01"
mkdir "$logs"
export PHASE65_FOLLOWUP_ROOT="$root" PHASE65_FOLLOWUP_LOGS="$logs"
nohup bash -c '
 source /home/scv/miniconda3/etc/profile.d/conda.sh || exit 8
 conda activate cloud-lite-pt210 || exit 8
 cd /home/scv/Cloud-Adapter-light/src || exit 8
 export XFORMERS_DISABLED=1 OMP_NUM_THREADS=4
 python -u tools/trace_phase65d_followup.py development --root "$PHASE65_FOLLOWUP_ROOT"
 code=$?
 if test "$code" = 0; then python -u tools/trace_phase65d_followup.py spectral --root "$PHASE65_FOLLOWUP_ROOT"; code=$?; fi
 if test "$code" = 0; then python -u tools/trace_phase65d_followup.py extract --root "$PHASE65_FOLLOWUP_ROOT" --step 3500 --cohort phase65b_evidence; code=$?; fi
 if test "$code" = 0; then python -u tools/trace_phase65d_followup.py extract --root "$PHASE65_FOLLOWUP_ROOT" --step 4000 --cohort phase65b_evidence; code=$?; fi
 if test "$code" = 0; then python -u tools/trace_phase65d_followup.py t18 --root "$PHASE65_FOLLOWUP_ROOT"; code=$?; fi
 printf "%s\n" "$code" > "$PHASE65_FOLLOWUP_LOGS/EXIT_CODE"
' > "$logs/console.log" 2> "$logs/error.log" < /dev/null &
printf '%s\n' "$!" > "$logs/PID"
git -C /home/scv/Cloud-Adapter-light rev-parse HEAD > "$logs/GIT_SHA"
echo "Started analysis resume PID=$!"

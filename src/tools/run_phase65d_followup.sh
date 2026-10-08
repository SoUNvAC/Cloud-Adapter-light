#!/usr/bin/env bash
set -euo pipefail
repo=/home/scv/Cloud-Adapter-light
root=${1:-/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase65d_followup_20261008}
case "$root" in /home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase65d_followup_20261008*) ;; *) exit 2 ;; esac
test ! -e "$root" || { echo 'Existing followup; inspect without overwrite'; exit 2; }
mkdir "$root"
mkdir "$root/run_logs"
git -C "$repo" rev-parse HEAD > "$root/run_logs/GIT_SHA"
sha256sum "$repo/src/tools/trace_phase65d_followup.py" "$repo/src/tools/phase65d_followup_metrics.py" "$repo/src/research_plans/PHASE65D_FOLLOWUP_20261008.md" > "$root/run_logs/CODE_SHA256"
export PHASE65_FOLLOWUP_ROOT="$root"
nohup bash -c '
 source /home/scv/miniconda3/etc/profile.d/conda.sh || exit 8
 conda activate cloud-lite-pt210 || exit 8
 cd /home/scv/Cloud-Adapter-light/src || exit 8
 export XFORMERS_DISABLED=1 OMP_NUM_THREADS=4
 python -u tools/trace_phase65d_followup.py extract --root "$PHASE65_FOLLOWUP_ROOT" --step 3500 --cohort development_val
 code=$?
 if test "$code" = 0; then python -u tools/trace_phase65d_followup.py extract --root "$PHASE65_FOLLOWUP_ROOT" --step 4000 --cohort development_val; code=$?; fi
 if test "$code" = 0; then python -u tools/trace_phase65d_followup.py development --root "$PHASE65_FOLLOWUP_ROOT"; code=$?; fi
 if test "$code" = 0; then python -u tools/trace_phase65d_followup.py spectral --root "$PHASE65_FOLLOWUP_ROOT"; code=$?; fi
 if test "$code" = 0; then python -u tools/trace_phase65d_followup.py extract --root "$PHASE65_FOLLOWUP_ROOT" --step 3500 --cohort phase65b_evidence; code=$?; fi
 if test "$code" = 0; then python -u tools/trace_phase65d_followup.py extract --root "$PHASE65_FOLLOWUP_ROOT" --step 4000 --cohort phase65b_evidence; code=$?; fi
 if test "$code" = 0; then python -u tools/trace_phase65d_followup.py t18 --root "$PHASE65_FOLLOWUP_ROOT"; code=$?; fi
 printf "%s\n" "$code" > "$PHASE65_FOLLOWUP_ROOT/run_logs/EXIT_CODE"
' > "$root/run_logs/console.log" 2> "$root/run_logs/error.log" < /dev/null &
printf '%s\n' "$!" > "$root/run_logs/PID"
echo "Started followup PID=$!"

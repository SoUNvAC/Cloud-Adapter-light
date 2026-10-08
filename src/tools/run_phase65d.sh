#!/usr/bin/env bash
set -euo pipefail
repo=/home/scv/Cloud-Adapter-light
root=/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase65d_20261008
test ! -e "$root" || { echo 'Existing 65D: inspect, do not overwrite or refit'; exit 2; }
mkdir "$root"
mkdir "$root/run_logs"
git -C "$repo" rev-parse HEAD > "$root/run_logs/GIT_SHA"
sha256sum "$repo/src/research_plans/PHASE65D_20261008.md" "$repo/src/tools/extract_phase65d_scores.py" "$repo/src/tools/analyze_phase65d.py" "$repo/src/tools/phase65d_metrics.py" > "$root/run_logs/CODE_SHA256"
export PHASE65D_ROOT="$root"
nohup bash -c '
  source /home/scv/miniconda3/etc/profile.d/conda.sh || exit 8
  conda activate cloud-lite-pt210 || exit 8
  cd /home/scv/Cloud-Adapter-light/src || exit 8
  export XFORMERS_DISABLED=1 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
  python -u tools/extract_phase65d_scores.py --cohort development_val --model source --output "$PHASE65D_ROOT"
  code=$?
  if test "$code" = 0; then python -u tools/extract_phase65d_scores.py --cohort development_val --model msre --output "$PHASE65D_ROOT"; code=$?; fi
  if test "$code" = 0; then python -u tools/analyze_phase65d.py freeze --root "$PHASE65D_ROOT"; code=$?; fi
  if test "$code" = 0; then python -u tools/extract_phase65d_scores.py --cohort phase65b_evidence --model source --output "$PHASE65D_ROOT"; code=$?; fi
  if test "$code" = 0; then python -u tools/extract_phase65d_scores.py --cohort phase65b_evidence --model msre --output "$PHASE65D_ROOT"; code=$?; fi
  if test "$code" = 0; then python -u tools/analyze_phase65d.py summarize --root "$PHASE65D_ROOT"; code=$?; fi
  printf "%s\n" "$code" > "$PHASE65D_ROOT/run_logs/EXIT_CODE"
  exit "$code"
' > "$root/run_logs/console.log" 2> "$root/run_logs/error.log" < /dev/null &
printf '%s\n' "$!" > "$root/run_logs/PID"
printf 'Started 65D PID=%s\n' "$!"

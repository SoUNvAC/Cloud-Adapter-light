#!/usr/bin/env bash
set -euo pipefail
repo=/home/scv/Cloud-Adapter-light
run="$repo/src/work_dirs/phase65b_explore_20261008"
data=/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871
prepared="$data/phase65b_explore_20261008"
a_run="$repo/src/work_dirs/phase65a_explore_20261008"
test ! -e "$run" && test ! -e "$prepared" || { echo 'Existing run/data: inspect, do not overwrite'; exit 2; }
test "$(TZ=Asia/Shanghai date +%Y%m%d)" -lt 20261020 || { echo 'Budget deadline reached'; exit 3; }
mkdir "$run"
git -C "$repo" rev-parse HEAD > "$run/GIT_SHA"
sha256sum "$repo/src/research_plans/PHASE65B_EXPLORATORY_20261008.md" "$repo/src/tools/phase65b_evidence_probe.py" "$repo/src/tools/evaluate_phase65a_explore.py" "$repo/src/cloud_adapter/datasets/phase65_catalogue.py" "$repo/src/tools/summarize_phase65a_explore.py" > "$run/CODE_SHA256"
export PHASE65_PREPARED_MANIFEST="$data/phase65a_explore_20261008/manifest.json"
export PHASE65_SOURCE_CHECKPOINT="$repo/src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth"
export PHASE65B_RUN="$run" PHASE65B_DATA="$data" PHASE65B_PREPARED="$prepared" PHASE65A_RUN="$a_run"
nohup bash -c '
  source /home/scv/miniconda3/etc/profile.d/conda.sh || exit 8
  conda activate cloud-lite-pt210 || exit 8
  cd /home/scv/Cloud-Adapter-light/src || exit 8
  export XFORMERS_DISABLED=1 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
  python -u tools/phase65b_evidence_probe.py prepare --root "$PHASE65B_DATA" --split work_dirs/phase65a/split_lock.json --adapted-checkpoint "$PHASE65A_RUN/msre_seed65/best_mIoU_iter_2000.pth" --output "$PHASE65B_PREPARED"
  code=$?
  if test "$code" = 0; then
    python -u tools/evaluate_phase65a_explore.py --checkpoint "$PHASE65_SOURCE_CHECKPOINT" --source-only --phase65b-manifest "$PHASE65B_PREPARED/manifest.json" --output "$PHASE65B_RUN/source_evaluation.json"
    code=$?
  fi
  if test "$code" = 0; then
    python -u tools/evaluate_phase65a_explore.py --checkpoint "$PHASE65A_RUN/msre_seed65/best_mIoU_iter_2000.pth" --phase65b-manifest "$PHASE65B_PREPARED/manifest.json" --output "$PHASE65B_RUN/adapted_evaluation.json"
    code=$?
  fi
  if test "$code" = 0; then
    python -u tools/phase65b_evidence_probe.py probe --fit-source "$PHASE65A_RUN/source_evaluation.json" --fit-adapted "$PHASE65A_RUN/adapted_evaluation.json" --evidence-source "$PHASE65B_RUN/source_evaluation.json" --evidence-adapted "$PHASE65B_RUN/adapted_evaluation.json" --output "$PHASE65B_RUN/evidence_report.json"
    code=$?
  fi
  printf "%s\n" "$code" > "$PHASE65B_RUN/EXIT_CODE"
  exit "$code"
' > "$run/console.log" 2> "$run/error.log" < /dev/null &
printf '%s\n' "$!" > "$run/PID"
printf 'Started 65b evidence PID=%s\n' "$!"

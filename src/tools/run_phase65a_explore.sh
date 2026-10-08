#!/usr/bin/env bash
set -euo pipefail
repo=/home/scv/Cloud-Adapter-light
run="$repo/src/work_dirs/phase65a_explore_20261008"
data=/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871
prepared="$data/phase65a_explore_20261008"
test ! -e "$run" || { echo 'Existing exploratory run: inspect, do not overwrite'; exit 2; }
test ! -e "$prepared" || { echo 'Existing prepared data: inspect, do not overwrite'; exit 2; }
test "$(TZ=Asia/Shanghai date +%Y%m%d)" -le 20261020 || { echo 'Research deadline reached'; exit 3; }
mkdir "$run"
sha256sum "$repo/src/research_plans/PHASE65A_EXPLORATORY_20261008.md" "$repo/src/configs/protocol/phase65a_explore_rgb.py" "$repo/src/tools/prepare_phase65a_explore.py" "$repo/src/tools/evaluate_phase65a_explore.py" "$repo/src/tools/summarize_phase65a_explore.py" "$repo/src/cloud_adapter/datasets/phase65_catalogue.py" > "$run/CODE_SHA256"
export PHASE65_SOURCE_CHECKPOINT="$repo/src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth"
export PHASE65_PREPARED_MANIFEST="$prepared/manifest.json"
export PHASE65_EXPLORE_RUN="$run" PHASE65_EXPLORE_DATA="$data" PHASE65_EXPLORE_PREPARED="$prepared"
nohup bash -c '
  set +e
  source /home/scv/miniconda3/etc/profile.d/conda.sh
  conda activate cloud-lite-pt210
  cd /home/scv/Cloud-Adapter-light/src
  export XFORMERS_DISABLED=1 OMP_NUM_THREADS=4
  python -u tools/prepare_phase65a_explore.py --root "$PHASE65_EXPLORE_DATA" --split work_dirs/phase65a/split_lock.json --source "$PHASE65_SOURCE_CHECKPOINT" --output "$PHASE65_EXPLORE_PREPARED"
  code=$?
  if test "$code" = 0; then
    python -u tools/evaluate_phase65a_explore.py --checkpoint "$PHASE65_SOURCE_CHECKPOINT" --source-only --preflight --output "$PHASE65_EXPLORE_RUN/preflight.json"
    code=$?
  fi
  if test "$code" = 0; then
    python -u tools/evaluate_phase65a_explore.py --checkpoint "$PHASE65_SOURCE_CHECKPOINT" --source-only --output "$PHASE65_EXPLORE_RUN/source_evaluation.json"
    code=$?
  fi
  if test "$code" = 0; then
    if test "$(TZ=Asia/Shanghai date +%Y%m%d)" -ge 20261020; then code=3; else
      python -u tools/train.py configs/protocol/phase65a_explore_rgb.py --work-dir "$PHASE65_EXPLORE_RUN/msre_seed65"
      code=$?
    fi
  fi
  if test "$code" = 0; then
    best=$(find "$PHASE65_EXPLORE_RUN/msre_seed65" -maxdepth 1 -name "best_mIoU*.pth" -type f)
    if test -z "$best" || test "$(printf "%s\n" "$best" | wc -l)" != 1; then code=4; else
      python -u tools/evaluate_phase65a_explore.py --checkpoint "$best" --output "$PHASE65_EXPLORE_RUN/adapted_evaluation.json"
      code=$?
    fi
  fi
  if test "$code" = 0; then
    python -u tools/summarize_phase65a_explore.py --source "$PHASE65_EXPLORE_RUN/source_evaluation.json" --adapted "$PHASE65_EXPLORE_RUN/adapted_evaluation.json" --output "$PHASE65_EXPLORE_RUN/exploratory_summary.json"
    code=$?
  fi
  printf "%s\n" "$code" > "$PHASE65_EXPLORE_RUN/EXIT_CODE"
  exit "$code"
' > "$run/console.log" 2> "$run/error.log" < /dev/null &
printf '%s\n' "$!" > "$run/PID"
git -C "$repo" rev-parse HEAD > "$run/GIT_SHA"
printf 'Started exploratory orchestrator PID=%s\n' "$!"

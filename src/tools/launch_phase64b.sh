#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/scv/Cloud-Adapter-light"
src_root="${repo_root}/src"
conda_hook="/home/scv/miniconda3/etc/profile.d/conda.sh"

usage() {
  printf '%s\n' \
    "usage:" \
    "  $0 source SOURCE_INIT" \
    "  $0 target-eval {l8|sparcs} SOURCE_CHECKPOINT" \
    "  $0 target-train {l8|sparcs} SOURCE_CHECKPOINT" \
    "  $0 retention {l8|sparcs} SOURCE_CHECKPOINT TARGET_CHECKPOINT"
  exit 2
}

[[ $# -ge 1 ]] || usage
mode="$1"
target=""
method=""
checkpoint=""
case "$mode" in
  source)
    [[ $# -eq 2 ]] || usage
    export PHASE64B_SOURCE_INIT="$(realpath -e "$2")"
    config="configs/protocol/phase64b_source_parent_six.py"
    work_dir="work_dirs/phase64b_source_parent/seed64"
    action="train"
    ;;
  target-eval|target-train)
    [[ $# -eq 3 ]] || usage
    target="$2"
    [[ "$target" == "l8" || "$target" == "sparcs" ]] || usage
    checkpoint="$(realpath -e "$3")"
    method="source_only"
    action="test"
    if [[ "$mode" == "target-train" ]]; then
      method="msre"
      action="train"
    fi
    export PHASE64B_TARGET="$target" PHASE64B_METHOD="$method"
    export PHASE64B_SOURCE_CHECKPOINT="$checkpoint"
    config="configs/protocol/phase64b_target_parent_six.py"
    work_dir="work_dirs/phase64b_target_parent/${target}/${method}/seed64"
    ;;
  retention)
    [[ $# -eq 4 ]] || usage
    target="$2"
    [[ "$target" == "l8" || "$target" == "sparcs" ]] || usage
    export PHASE64B_TARGET="$target" PHASE64B_METHOD="msre"
    export PHASE64B_SOURCE_CHECKPOINT="$(realpath -e "$3")"
    checkpoint="$(realpath -e "$4")"
    config="configs/protocol/phase64b_source_retention_six.py"
    work_dir="work_dirs/phase64b_source_retention/${target}/msre/seed64"
    action="test"
    ;;
  *) usage ;;
esac

case "${checkpoint:-${PHASE64B_SOURCE_INIT:-}}" in
  "${repo_root}"/*) ;;
  *) printf 'checkpoint escapes repository\n' >&2; exit 3 ;;
esac
cd "$src_root"
[[ -f work_dirs/phase64b_input_protocol/summary.json ]] || {
  printf 'Phase64B-0 summary is missing\n' >&2; exit 4;
}
case "$(realpath -m "$work_dir")" in
  "${src_root}"/work_dirs/phase64b_*) ;;
  *) printf 'work directory escapes Phase64B scope\n' >&2; exit 5 ;;
esac

complete="TRAIN_COMPLETE"
[[ "$action" == "train" ]] || complete="EVALUATION_COMPLETE"
if [[ -f "$work_dir/$complete" ]]; then
  printf 'already complete: %s\n' "$work_dir"
  exit 0
fi
if [[ -f "$work_dir/PID" ]]; then
  old_pid="$(<"$work_dir/PID")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    printf 'already running pid=%s\n' "$old_pid"
    exit 0
  fi
  printf 'stale PID requires inspection: %s\n' "$work_dir/PID" >&2
  exit 6
fi
mkdir -p "$work_dir"
export PHASE64B_LAUNCH_CONFIG="$config"
export PHASE64B_LAUNCH_WORK_DIR="$work_dir"
export PHASE64B_LAUNCH_ACTION="$action"
export PHASE64B_LAUNCH_CHECKPOINT="$checkpoint"
export PHASE64B_LAUNCH_CONDA_HOOK="$conda_hook"

nohup bash -c '
  set +e
  source "$PHASE64B_LAUNCH_CONDA_HOOK"
  conda activate cloud-lite-pt210
  cd /home/scv/Cloud-Adapter-light/src
  if [[ "$PHASE64B_LAUNCH_ACTION" == "train" ]]; then
    python tools/train.py "$PHASE64B_LAUNCH_CONFIG" --work-dir "$PHASE64B_LAUNCH_WORK_DIR"
  else
    python tools/test.py "$PHASE64B_LAUNCH_CONFIG" "$PHASE64B_LAUNCH_CHECKPOINT" --work-dir "$PHASE64B_LAUNCH_WORK_DIR"
  fi
  status=$?
  printf "%s\n" "$status" > "$PHASE64B_LAUNCH_WORK_DIR/EXIT_CODE"
  if [[ $status -eq 0 ]]; then
    if [[ "$PHASE64B_LAUNCH_ACTION" == "train" ]]; then
      touch "$PHASE64B_LAUNCH_WORK_DIR/TRAIN_COMPLETE"
    else
      touch "$PHASE64B_LAUNCH_WORK_DIR/EVALUATION_COMPLETE"
    fi
  fi
  exit "$status"
' > "$work_dir/phase64b_console.log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "$pid" > "$work_dir/PID"
printf 'started pid=%s work_dir=%s\n' "$pid" "$work_dir"

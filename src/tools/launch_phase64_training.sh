#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/scv/Cloud-Adapter-light"
src_root="${repo_root}/src"
conda_hook="/home/scv/miniconda3/etc/profile.d/conda.sh"

usage() {
  printf '%s\n' \
    "usage:" \
    "  $0 source" \
    "  $0 target {l8|sparcs} {shared_parent|full|lora|msre} PARENT_CHECKPOINT"
  exit 2
}

if [[ $# -lt 1 ]]; then
  usage
fi

mode="$1"
case "${mode}" in
  source)
    [[ $# -eq 1 ]] || usage
    config="configs/protocol/phase64_source_parent_rgb.py"
    work_dir="work_dirs/phase64_source_parent/seed64"
    unset PHASE64_TARGET PHASE64_METHOD PHASE64_PARENT_CHECKPOINT || true
    ;;
  target)
    [[ $# -eq 4 ]] || usage
    target="$2"
    method="$3"
    checkpoint="$4"
    [[ "${target}" == "l8" || "${target}" == "sparcs" ]] || usage
    case "${method}" in
      shared_parent|full|lora|msre) ;;
      *) usage ;;
    esac
    checkpoint="$(realpath -e "${checkpoint}")"
    case "${checkpoint}" in
      "${repo_root}"/*) ;;
      *) printf 'checkpoint escapes repository: %s\n' "${checkpoint}" >&2; exit 3 ;;
    esac
    config="configs/protocol/phase64_target_parent_rgb.py"
    work_dir="work_dirs/phase64_target_parent/${target}/${method}/seed64"
    export PHASE64_TARGET="${target}"
    export PHASE64_METHOD="${method}"
    export PHASE64_PARENT_CHECKPOINT="${checkpoint}"
    ;;
  *) usage ;;
esac

cd "${src_root}"
case "$(realpath -m "${work_dir}")" in
  "${src_root}"/work_dirs/phase64_*) ;;
  *) printf 'work directory escapes Phase 64 scope: %s\n' "${work_dir}" >&2; exit 4 ;;
esac

if [[ -f "${work_dir}/TRAIN_COMPLETE" ]]; then
  printf 'already complete: %s\n' "${work_dir}"
  exit 0
fi
if [[ -f "${work_dir}/PID" ]]; then
  old_pid="$(<"${work_dir}/PID")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    printf 'already running pid=%s work_dir=%s\n' "${old_pid}" "${work_dir}"
    exit 0
  fi
  printf 'stale PID exists; inspect before relaunch: %s\n' "${work_dir}/PID" >&2
  exit 5
fi
if [[ -e "${work_dir}/phase64_console.log" || -e "${work_dir}/EXIT_CODE" ]]; then
  printf 'existing incomplete artifacts; inspect before relaunch: %s\n' "${work_dir}" >&2
  exit 6
fi

mkdir -p "${work_dir}"
export PHASE64_LAUNCH_CONFIG="${config}"
export PHASE64_LAUNCH_WORK_DIR="${work_dir}"
export PHASE64_LAUNCH_CONDA_HOOK="${conda_hook}"

nohup bash -c '
  set +e
  source "${PHASE64_LAUNCH_CONDA_HOOK}"
  conda activate cloud-lite-pt210
  cd /home/scv/Cloud-Adapter-light/src
  python tools/train.py "${PHASE64_LAUNCH_CONFIG}" --work-dir "${PHASE64_LAUNCH_WORK_DIR}"
  status=$?
  printf "%s\n" "${status}" > "${PHASE64_LAUNCH_WORK_DIR}/EXIT_CODE"
  if [[ ${status} -eq 0 ]]; then
    touch "${PHASE64_LAUNCH_WORK_DIR}/TRAIN_COMPLETE"
  fi
  exit "${status}"
' > "${work_dir}/phase64_console.log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "${pid}" > "${work_dir}/PID"
printf 'started pid=%s work_dir=%s\n' "${pid}" "${work_dir}"

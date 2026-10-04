#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/scv/Cloud-Adapter-light"
src_root="${repo_root}/src"
conda_hook="/home/scv/miniconda3/etc/profile.d/conda.sh"
source_parent="${src_root}/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth"

usage() {
  printf '%s\n' "usage: $0 {l8|sparcs} {shared_parent|full|lora|msre} TARGET_CHECKPOINT"
  exit 2
}

[[ $# -eq 3 ]] || usage
target="$1"
method="$2"
checkpoint="$(realpath -e "$3")"
[[ "${target}" == "l8" || "${target}" == "sparcs" ]] || usage
case "${method}" in
  shared_parent|full|lora|msre) ;;
  *) usage ;;
esac
case "${checkpoint}" in
  "${repo_root}"/*) ;;
  *) printf 'checkpoint escapes repository: %s\n' "${checkpoint}" >&2; exit 3 ;;
esac
[[ -f "${source_parent}" ]] || {
  printf 'source parent checkpoint missing: %s\n' "${source_parent}" >&2
  exit 4
}

cd "${src_root}"
work_dir="work_dirs/phase64_source_retention/${target}/${method}/seed64"
case "$(realpath -m "${work_dir}")" in
  "${src_root}"/work_dirs/phase64_source_retention/*) ;;
  *) printf 'work directory escapes Phase 64 retention scope: %s\n' "${work_dir}" >&2; exit 5 ;;
esac

if [[ -f "${work_dir}/EVALUATION_COMPLETE" ]]; then
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
  exit 6
fi
if [[ -e "${work_dir}/evaluation_console.log" || -e "${work_dir}/EXIT_CODE" ]]; then
  printf 'existing incomplete artifacts; inspect before relaunch: %s\n' "${work_dir}" >&2
  exit 7
fi

mkdir -p "${work_dir}"
export PHASE64_EVAL_TARGET="${target}"
export PHASE64_EVAL_METHOD="${method}"
export PHASE64_EVAL_CHECKPOINT="${checkpoint}"
export PHASE64_EVAL_SOURCE_PARENT="${source_parent}"
export PHASE64_EVAL_WORK_DIR="${work_dir}"
export PHASE64_EVAL_CONDA_HOOK="${conda_hook}"

nohup bash -c '
  set +e
  source "${PHASE64_EVAL_CONDA_HOOK}"
  conda activate cloud-lite-pt210
  cd /home/scv/Cloud-Adapter-light/src
  export PHASE64_TARGET="${PHASE64_EVAL_TARGET}"
  export PHASE64_METHOD="${PHASE64_EVAL_METHOD}"
  export PHASE64_PARENT_CHECKPOINT="${PHASE64_EVAL_SOURCE_PARENT}"
  python tools/test.py \
    configs/protocol/phase64_source_retention_rgb.py \
    "${PHASE64_EVAL_CHECKPOINT}" \
    --work-dir "${PHASE64_EVAL_WORK_DIR}"
  status=$?
  printf "%s\n" "${status}" > "${PHASE64_EVAL_WORK_DIR}/EXIT_CODE"
  if [[ ${status} -eq 0 ]]; then
    touch "${PHASE64_EVAL_WORK_DIR}/EVALUATION_COMPLETE"
  fi
  exit "${status}"
' > "${work_dir}/evaluation_console.log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "${pid}" > "${work_dir}/PID"
printf 'started pid=%s work_dir=%s\n' "${pid}" "${work_dir}"

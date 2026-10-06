#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/scv/Cloud-Adapter-light"
src_root="${repo_root}/src"
root="${src_root}/work_dirs/phase64_msre_multiseed"
conda_hook="/home/scv/miniconda3/etc/profile.d/conda.sh"
source_checkpoint="${src_root}/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth"

case "$(realpath -m "${root}")" in
  "${src_root}"/work_dirs/phase64_msre_multiseed) ;;
  *) printf 'orchestrator directory escapes Phase 64 scope: %s\n' "${root}" >&2; exit 2 ;;
esac
[[ -f "${source_checkpoint}" ]] || {
  printf 'source checkpoint missing: %s\n' "${source_checkpoint}" >&2
  exit 3
}
if [[ -f "${root}/PHASE64_MSRE_MULTISEED_COMPLETE" ]]; then
  printf 'already complete: %s\n' "${root}"
  exit 0
fi
if [[ -f "${root}/PID" ]]; then
  old_pid="$(<"${root}/PID")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    printf 'already running pid=%s\n' "${old_pid}"
    exit 0
  fi
  printf 'stale PID exists; inspect before relaunch: %s\n' "${root}/PID" >&2
  exit 4
fi
if [[ -e "${root}/console.log" || -e "${root}/EXIT_CODE" ]]; then
  printf 'existing incomplete artifacts; inspect before relaunch: %s\n' "${root}" >&2
  exit 5
fi

mkdir -p "${root}"
export PHASE64_MULTI_ROOT="${root}"
export PHASE64_MULTI_CONDA_HOOK="${conda_hook}"
export PHASE64_MULTI_SOURCE="${source_checkpoint}"

nohup bash -c '
  set +e
  source "${PHASE64_MULTI_CONDA_HOOK}"
  conda activate cloud-lite-pt210
  cd /home/scv/Cloud-Adapter-light/src
  status=0
  for target in l8 sparcs; do
    for seed in 65 66; do
      work_dir="work_dirs/phase64_target_parent/${target}/msre/seed${seed}"
      case "$(realpath -m "${work_dir}")" in
        /home/scv/Cloud-Adapter-light/src/work_dirs/phase64_target_parent/*/msre/seed*) ;;
        *) printf "unsafe work directory: %s\n" "${work_dir}" >&2; status=20; break 2 ;;
      esac
      if [[ -f "${work_dir}/TRAIN_COMPLETE" ]] && [[ "$(<"${work_dir}/EXIT_CODE")" == 0 ]]; then
        printf "skip completed target=%s seed=%s\n" "${target}" "${seed}"
        continue
      fi
      if [[ -e "${work_dir}/phase64_console.log" || -e "${work_dir}/EXIT_CODE" ]]; then
        printf "incomplete existing run requires audit: %s\n" "${work_dir}" >&2
        status=21
        break 2
      fi
      mkdir -p "${work_dir}"
      export PHASE64_TARGET="${target}"
      export PHASE64_METHOD="msre"
      export PHASE64_SEED="${seed}"
      export PHASE64_PARENT_CHECKPOINT="${PHASE64_MULTI_SOURCE}"
      printf "start target=%s seed=%s\n" "${target}" "${seed}"
      python tools/train.py \
        configs/protocol/phase64_target_parent_rgb_multiseed.py \
        --work-dir "${work_dir}" > "${work_dir}/phase64_console.log" 2>&1
      run_status=$?
      printf "%s\n" "${run_status}" > "${work_dir}/EXIT_CODE"
      if [[ ${run_status} -ne 0 ]]; then
        status=${run_status}
        printf "failed target=%s seed=%s status=%s\n" "${target}" "${seed}" "${status}" >&2
        break 2
      fi
      touch "${work_dir}/TRAIN_COMPLETE"
      printf "complete target=%s seed=%s\n" "${target}" "${seed}"
    done
  done
  printf "%s\n" "${status}" > "${PHASE64_MULTI_ROOT}/EXIT_CODE"
  if [[ ${status} -eq 0 ]]; then
    touch "${PHASE64_MULTI_ROOT}/PHASE64_MSRE_MULTISEED_COMPLETE"
  fi
  exit "${status}"
' > "${root}/console.log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "${pid}" > "${root}/PID"
printf 'started pid=%s root=%s\n' "${pid}" "${root}"

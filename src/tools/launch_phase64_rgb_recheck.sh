#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/scv/Cloud-Adapter-light"
src_root="${repo_root}/src"
work_dir="${src_root}/work_dirs/phase64_rgb_recheck"
conda_hook="/home/scv/miniconda3/etc/profile.d/conda.sh"

case "$(realpath -m "${work_dir}")" in
  "${src_root}"/work_dirs/phase64_rgb_recheck) ;;
  *) printf 'work directory escapes Phase 64 RGB recheck scope: %s\n' "${work_dir}" >&2; exit 2 ;;
esac

if [[ -f "${work_dir}/PHASE64_RGB_RECHECK_COMPLETE" ]]; then
  printf 'already complete: %s\n' "${work_dir}"
  exit 0
fi
if [[ -f "${work_dir}/PID" ]]; then
  old_pid="$(<"${work_dir}/PID")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    printf 'already running pid=%s\n' "${old_pid}"
    exit 0
  fi
  printf 'stale PID exists; inspect before relaunch: %s\n' "${work_dir}/PID" >&2
  exit 3
fi
if [[ -e "${work_dir}/console.log" || -e "${work_dir}/EXIT_CODE" ]]; then
  printf 'existing incomplete artifacts; inspect before relaunch: %s\n' "${work_dir}" >&2
  exit 4
fi

mkdir -p "${work_dir}"
export PHASE64_RECHECK_ROOT="${work_dir}"
export PHASE64_RECHECK_CONDA_HOOK="${conda_hook}"

nohup bash -c '
  set +e
  source "${PHASE64_RECHECK_CONDA_HOOK}"
  conda activate cloud-lite-pt210
  cd /home/scv/Cloud-Adapter-light/src
  python tools/evaluate_phase64_rgb_recheck.py source-switch \
    --output "${PHASE64_RECHECK_ROOT}/source_switch.json"
  status=$?
  if [[ ${status} -eq 0 ]]; then
    python tools/evaluate_phase64_rgb_recheck.py paired-scene \
      --output "${PHASE64_RECHECK_ROOT}/paired_scene.json"
    status=$?
  fi
  printf "%s\n" "${status}" > "${PHASE64_RECHECK_ROOT}/EXIT_CODE"
  if [[ ${status} -eq 0 ]]; then
    touch "${PHASE64_RECHECK_ROOT}/PHASE64_RGB_RECHECK_COMPLETE"
  fi
  exit "${status}"
' > "${work_dir}/console.log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "${pid}" > "${work_dir}/PID"
printf 'started pid=%s work_dir=%s\n' "${pid}" "${work_dir}"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
conda_executable="${CONDA_EXE:-/home/scv/miniconda3/bin/conda}"
if [[ ! -x "${conda_executable}" ]]; then
  echo "Conda executable not found: ${conda_executable}" >&2
  exit 127
fi
conda_base="$("${conda_executable}" info --base)"
source "${conda_base}/etc/profile.d/conda.sh"
conda activate cloud-lite-pt210

root="work_dirs/phase56_readout"
registry="work_dirs/phase56_protocol/run_registry.csv"
run_id="P56_readout_MSRE_CS2L8_cache_s56"
mkdir -p "${root}" "$(dirname "${registry}")"

finish_status="fail"
finish_note="launcher exited before completion"
finish_registry() {
  if [[ "${finish_status}" == "finish" ]]; then
    python tools/phase56_run_registry.py finish --registry "${registry}" \
      --run-id "${run_id}" --notes "${finish_note}"
  else
    python tools/phase56_run_registry.py fail --registry "${registry}" \
      --run-id "${run_id}" --notes "${finish_note}" || true
  fi
}
trap finish_registry EXIT

if [[ ! -f "${registry}" ]] || ! grep -q "^${run_id}," "${registry}"; then
  python tools/phase56_run_registry.py register --registry "${registry}" \
    --run-id "${run_id}" --phase 56 --route readout \
    --command "bash tools/run_phase56_readout_4090d.sh" \
    --notes "Shadows?=yes only; target-test sealed"
fi
python tools/phase56_run_registry.py start --registry "${registry}" --run-id "${run_id}"

python tools/audit_phase56_label_support.py 2>&1 | tee "${root}/label_support.log"

exec 9>"work_dirs/phase56_protocol/gpu.lock"
if ! flock -n 9; then
  finish_note="GPU registry lock is held by another Phase 56-59 run"
  exit 4
fi
python tools/cache_phase56_readout_features.py 2>&1 | tee "${root}/feature_cache.log"
flock -u 9

python tools/audit_phase56_readout.py 2>&1 | tee "${root}/readout.log"
if ! flock -n 9; then
  finish_note="GPU registry lock is held before dense readout validation"
  exit 4
fi
python tools/evaluate_phase56_readout_dense.py 2>&1 | tee "${root}/dense_readout.log"
flock -u 9
python tools/summarize_phase56_matrix.py 2>&1 | tee "${root}/summary.log"

manifest_sha="$(sha256sum "${root}/feature_cache_manifest.json" | awk '{print $1}')"
python tools/phase56_run_registry.py finish --registry "${registry}" \
  --run-id "${run_id}" --manifest-sha256 "${manifest_sha}" \
  --notes "Phase 56 completed; inspect summary.json for scientific gate"
trap - EXIT
finish_status="finish"
touch "${root}/RUN_COMPLETE"

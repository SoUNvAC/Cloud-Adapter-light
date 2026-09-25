#!/usr/bin/env bash
set -euo pipefail

cd /home/scv/Cloud-Adapter-light/src
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210

root="work_dirs/phase62_partial_labels"
mkdir -p "${root}"
date -Is > "${root}/STARTED_AT.txt"
python -c 'import torch; print({"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)})' \
  > "${root}/RUNTIME_PREFLIGHT.txt"

bash tools/run_phase62_4090d.sh

date -Is > "${root}/COMPLETED_AT.txt"


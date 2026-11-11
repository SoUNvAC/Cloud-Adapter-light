#!/usr/bin/env bash
set -euo pipefail
cd /home/scv/shared/phase466_20261107
export PYTHONPATH=/home/scv/shared/phase466_20261107:/home/scv/shared/phase450_20261106:/home/scv/Cloud-Adapter-light/src
for mode in token_read_off projector_off; do
 /home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -u dissect_msre466.py "$mode" >"$mode.log" 2>&1
done

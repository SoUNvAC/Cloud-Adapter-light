#!/usr/bin/env bash
set -euo pipefail
cd /home/scv/shared/phase466_20261107
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
export PYTHONPATH=/home/scv/shared/phase466_20261107:/home/scv/shared/phase450_20261106:/home/scv/Cloud-Adapter-light/src
for mode in msre_off read_off write_off read_write_off; do
 "$PY" -u ablate_components466.py "$mode" >"$mode.log" 2>&1
done

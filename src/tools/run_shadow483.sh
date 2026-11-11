#!/bin/bash
set -eu
cd /home/scv/shared/phase483_20261108
SEED="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "seed_${SEED}_EXIT_CODE"' EXIT
$PY train_shadow483.py B "$SEED"
$PY verify_shadow483.py B "$SEED"
$PY boundary_shadow483.py B "$SEED"
$PY ablate_shadow483.py B "$SEED"
$PY boundary_effect483.py B "$SEED"
touch "seed_${SEED}/B/DONE"

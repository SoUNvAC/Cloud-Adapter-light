#!/bin/bash
set -eu
cd /home/scv/shared/phase489_20261108
SEED="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "seed_${SEED}_EXIT_CODE"' EXIT
$PY train_shadow489.py B "$SEED"
$PY verify_shadow489.py B "$SEED"
$PY boundary_shadow489.py B "$SEED"
$PY ablate_shadow489.py B "$SEED"
$PY boundary_effect489.py B "$SEED"
$PY ablate_region489.py B "$SEED"
$PY boundary_region489.py B "$SEED"
touch "seed_${SEED}/B/DONE"

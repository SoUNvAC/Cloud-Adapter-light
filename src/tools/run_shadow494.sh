#!/bin/bash
set -eu
cd /home/scv/shared/phase494_20261109
SEED="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "seed_${SEED}_EXIT_CODE"' EXIT
$PY train_shadow494.py B "$SEED"
$PY verify_shadow494.py B "$SEED"
$PY boundary_shadow494.py B "$SEED"
$PY ablate_shadow494.py B "$SEED"
$PY boundary_effect494.py B "$SEED"
$PY ablate_region494.py B "$SEED"
$PY boundary_region494.py B "$SEED"
touch "seed_${SEED}/B/DONE"

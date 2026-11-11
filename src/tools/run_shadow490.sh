#!/bin/bash
set -eu
cd /home/scv/shared/phase490_20261108
ARM="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "${ARM}_EXIT_CODE"' EXIT
$PY train_shadow490.py "$ARM"
$PY verify_shadow490.py "$ARM"
$PY boundary_shadow490.py "$ARM"
$PY ablate_shadow490.py "$ARM"
$PY boundary_effect490.py "$ARM"
$PY ablate_region490.py "$ARM"
touch "${ARM}/DONE"

#!/bin/bash
set -eu
cd /home/scv/shared/phase486_20261108
ARM="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "${ARM}_EXIT_CODE"' EXIT
$PY train_shadow486.py "$ARM"
$PY verify_shadow486.py "$ARM"
$PY boundary_shadow486.py "$ARM"
$PY ablate_shadow486.py "$ARM"
$PY boundary_effect486.py "$ARM"
$PY ablate_region486.py "$ARM"
touch "${ARM}/DONE"

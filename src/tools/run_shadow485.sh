#!/bin/bash
set -eu
cd /home/scv/shared/phase485_20261108
ARM="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "${ARM}_EXIT_CODE"' EXIT
$PY train_shadow485.py "$ARM"
$PY verify_shadow485.py "$ARM"
$PY boundary_shadow485.py "$ARM"
$PY ablate_shadow485.py "$ARM"
$PY boundary_effect485.py "$ARM"
$PY ablate_region485.py "$ARM"
touch "${ARM}/DONE"

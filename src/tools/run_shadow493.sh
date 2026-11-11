#!/bin/bash
set -eu
cd /home/scv/shared/phase493_20261109
ARM="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "${ARM}_EXIT_CODE"' EXIT
$PY train_shadow493.py "$ARM"
$PY verify_shadow493.py "$ARM"
$PY boundary_shadow493.py "$ARM"
$PY ablate_shadow493.py "$ARM"
$PY boundary_effect493.py "$ARM"
$PY ablate_region493.py "$ARM"
touch "${ARM}/DONE"

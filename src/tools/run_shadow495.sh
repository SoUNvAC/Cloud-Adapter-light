#!/bin/bash
set -eu
cd /home/scv/shared/phase495_20261109
ARM="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "${ARM}_EXIT_CODE"' EXIT
$PY train_shadow495.py "$ARM"
$PY verify_shadow495.py "$ARM"
$PY boundary_shadow495.py "$ARM"
$PY ablate_shadow495.py "$ARM"
$PY boundary_effect495.py "$ARM"
$PY ablate_region495.py "$ARM"
touch "${ARM}/DONE"

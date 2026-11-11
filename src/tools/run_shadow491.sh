#!/bin/bash
set -eu
cd /home/scv/shared/phase491_20261108
ARM="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "${ARM}_EXIT_CODE"' EXIT
$PY train_shadow491.py "$ARM"
$PY verify_shadow491.py "$ARM"
$PY boundary_shadow491.py "$ARM"
$PY ablate_shadow491.py "$ARM"
$PY boundary_effect491.py "$ARM"
$PY ablate_region491.py "$ARM"
touch "${ARM}/DONE"

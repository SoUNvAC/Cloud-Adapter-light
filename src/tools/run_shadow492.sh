#!/bin/bash
set -eu
cd /home/scv/shared/phase492_20261109
ARM="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "${ARM}_EXIT_CODE"' EXIT
$PY train_shadow492.py "$ARM"
$PY verify_shadow492.py "$ARM"
$PY boundary_shadow492.py "$ARM"
$PY ablate_shadow492.py "$ARM"
$PY boundary_effect492.py "$ARM"
$PY ablate_region492.py "$ARM"
touch "${ARM}/DONE"

#!/bin/bash
set -eu
cd /home/scv/shared/phase482_20261108
ARM="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "${ARM}_EXIT_CODE"' EXIT
$PY train_shadow482.py "$ARM"
$PY verify_shadow482.py "$ARM"
$PY boundary_shadow482.py "$ARM"
$PY ablate_shadow482.py "$ARM"
$PY boundary_effect482.py "$ARM"
if [ "$ARM" = A ]; then $PY ablate_contrast482.py "$ARM"; fi
touch "${ARM}/DONE"

#!/bin/bash
set -eu
cd /home/scv/shared/phase488_20261108
ARM="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "${ARM}_EXIT_CODE"' EXIT
$PY train_shadow488.py "$ARM"
$PY verify_shadow488.py "$ARM"
$PY boundary_shadow488.py "$ARM"
$PY ablate_shadow488.py "$ARM"
$PY boundary_effect488.py "$ARM"
$PY ablate_region488.py "$ARM"
touch "${ARM}/DONE"

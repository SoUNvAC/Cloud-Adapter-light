#!/bin/bash
set -eu
cd /home/scv/shared/phase476_20261107
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY train_shadow476.py "$1"
$PY verify_shadow476.py "$1"
$PY boundary_shadow476.py "$1"
if [ "$1" = B ]; then
 $PY ablate_shadow476.py B
 $PY ablate_appearance476.py B
fi

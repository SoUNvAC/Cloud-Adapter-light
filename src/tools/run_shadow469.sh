#!/bin/bash
set -eu
cd /home/scv/shared/phase469_20261107
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY train_shadow469.py "$1"
$PY verify_shadow469.py "$1"
$PY boundary_shadow469.py "$1"
if [ "$1" = B ]; then
 $PY ablate_shadow469.py B
 $PY ablate_relative469.py B
fi

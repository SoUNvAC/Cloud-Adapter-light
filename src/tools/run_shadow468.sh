#!/bin/bash
set -eu
cd /home/scv/shared/phase468_20261107
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY train_shadow468.py "$1"
$PY verify_shadow468.py "$1"
$PY boundary_shadow468.py "$1"
if [ "$1" = B ]; then
 $PY ablate_shadow468.py B
 $PY ablate_frequency468.py B
fi

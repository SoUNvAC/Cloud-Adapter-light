#!/bin/bash
set -eu
cd /home/scv/shared/phase467_20261107
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY train_shadow467.py "$1"
$PY verify_shadow467.py "$1"
$PY boundary_shadow467.py "$1"
if [ "$1" = B ]; then
 $PY ablate_shadow467.py B
 $PY ablate_ring467.py B
fi

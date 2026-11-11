#!/bin/bash
set -eu
cd /home/scv/shared/phase470_20261107
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY train_shadow470.py "$1"
$PY verify_shadow470.py "$1"
$PY boundary_shadow470.py "$1"
if [ "$1" = B ]; then
 $PY ablate_shadow470.py B
 $PY ablate_routing470.py B
fi

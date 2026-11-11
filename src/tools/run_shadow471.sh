#!/bin/bash
set -eu
cd /home/scv/shared/phase471_20261107
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY train_shadow471.py "$1"
$PY verify_shadow471.py "$1"
$PY boundary_shadow471.py "$1"
if [ "$1" = B ]; then
 $PY ablate_shadow471.py B
 $PY ablate_normal471.py B
fi

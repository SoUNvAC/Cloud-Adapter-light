#!/bin/bash
set -eu
cd /home/scv/shared/phase472_20261107
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY train_shadow472.py "$1"
$PY verify_shadow472.py "$1"
$PY boundary_shadow472.py "$1"
if [ "$1" = B ]; then
 $PY ablate_shadow472.py B
 $PY ablate_appearance472.py B
fi

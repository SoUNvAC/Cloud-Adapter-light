#!/bin/bash
set -eu
cd /home/scv/shared/phase465_20261107
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY train_shadow465.py "$1"
$PY verify_shadow465.py "$1"
$PY boundary_shadow465.py "$1"
if [ "$1" = B ]; then
 $PY ablate_shadow465.py B
 $PY ablate_class465.py B
fi

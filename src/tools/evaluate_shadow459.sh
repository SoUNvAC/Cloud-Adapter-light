#!/bin/bash
set -eu
cd /home/scv/shared/phase459_20261106
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY verify_shadow459.py "$1"
$PY boundary_shadow459.py "$1"
if [ "$1" = "B" ]; then
 $PY ablate_shadow459.py B
fi

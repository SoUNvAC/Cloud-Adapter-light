#!/bin/bash
set -eu
cd /home/scv/shared/phase462_20261106
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
if [ -d "$1/network_run" ]; then exit 0; fi
$PY train_shadow462.py "$1"
$PY verify_shadow462.py "$1"
$PY boundary_shadow462.py "$1"

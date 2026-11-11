#!/bin/bash
set -eu
cd /home/scv/shared/phase460_20261106
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
if [ -d B/network_run ]; then exit 0; fi
$PY audit_shadow460.py
$PY train_shadow460.py B
$PY verify_shadow460.py B
$PY boundary_shadow460.py B
$PY ablate_shadow460.py B
$PY ablate_reference460.py B

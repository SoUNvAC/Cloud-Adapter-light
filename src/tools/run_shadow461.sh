#!/bin/bash
set -eu
cd /home/scv/shared/phase461_20261106
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
if [ -d B/network_run ]; then exit 0; fi
$PY audit_shadow461.py
$PY train_shadow461.py B
$PY verify_shadow461.py B
$PY boundary_shadow461.py B
$PY ablate_shadow461.py B
$PY ablate_reference461.py B
$PY ablate_shuffle461.py B

#!/bin/bash
set -eu
cd /home/scv/shared/phase463_20261106
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY train_shadow463.py B
$PY verify_shadow463.py B
$PY ablate_shadow463.py B
$PY ablate_reference463.py B
$PY boundary_shadow463.py B
$PY accept_shadow463.py

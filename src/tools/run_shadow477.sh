#!/bin/bash
set -eu
cd /home/scv/shared/phase477_20261107
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY train_shadow477.py A
$PY verify_shadow477.py A
$PY boundary_shadow477.py A

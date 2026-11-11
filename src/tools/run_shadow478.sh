#!/bin/bash
set -eu
cd /home/scv/shared/phase478_20261108
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY train_shadow478.py "$1"
$PY verify_shadow478.py "$1"
$PY boundary_shadow478.py "$1"

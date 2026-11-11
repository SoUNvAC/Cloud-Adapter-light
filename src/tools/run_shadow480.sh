#!/bin/bash
set -eu
cd /home/scv/shared/phase480_20261108
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY train_shadow480.py "$1"
$PY verify_shadow480.py "$1"
$PY boundary_shadow480.py "$1"
$PY ablate_shadow480.py "$1"

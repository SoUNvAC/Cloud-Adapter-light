#!/bin/bash
set -eu
cd /home/scv/shared/phase474_20261107
test -f A/network_run/BOUNDARY_DIAG.json
test -f B/network_run/ABLATION_CONDITIONAL_OFF.json
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY ablate_spatial474.py B
$PY compare_shadow474.py
$PY accept_shadow474.py
touch POSTPROCESS_COMPLETE

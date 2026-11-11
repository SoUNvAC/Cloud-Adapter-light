#!/bin/bash
set -eu
cd /home/scv/shared/phase476_20261107
test -f A/network_run/BOUNDARY_DIAG.json
test -f B/network_run/ABLATION_APPEARANCE_OFF.json
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY diagnose_material476.py
$PY compare_shadow476.py
$PY accept_shadow476.py
touch POSTPROCESS_COMPLETE

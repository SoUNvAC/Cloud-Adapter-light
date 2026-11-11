#!/bin/bash
set -eu
cd /home/scv/shared/phase479_20261108
test -f A/network_run/BOUNDARY_DIAG.json
test -f B/network_run/BOUNDARY_DIAG.json
test -f A/network_run/ABLATION_OFF.json
test -f B/network_run/ABLATION_OFF.json
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY diagnose_envelope479.py
$PY compare_shadow479.py
$PY accept_shadow479.py
touch POSTPROCESS_COMPLETE

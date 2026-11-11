#!/bin/bash
set -eu
cd /home/scv/shared/phase477_20261107
test -f A/network_run/BOUNDARY_DIAG.json
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY compare_shadow477.py
$PY accept_shadow477.py
touch POSTPROCESS_COMPLETE

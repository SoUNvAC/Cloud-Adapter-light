#!/bin/bash
set -eu
cd /home/scv/shared/phase478_20261108
test -f A/network_run/BOUNDARY_DIAG.json
test -f B/network_run/BOUNDARY_DIAG.json
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY compare_shadow478.py
$PY accept_shadow478.py
touch POSTPROCESS_COMPLETE

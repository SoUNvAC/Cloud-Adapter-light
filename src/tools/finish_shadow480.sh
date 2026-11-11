#!/bin/bash
set -eu
cd /home/scv/shared/phase480_20261108
test -f A/network_run/BOUNDARY_DIAG.json
test -f B/network_run/BOUNDARY_DIAG.json
test -f A/network_run/ABLATION_OFF.json
test -f B/network_run/ABLATION_OFF.json
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY diagnose_redistribution480.py
$PY boundary_effect480.py A
$PY boundary_effect480.py B
$PY compare_shadow480.py
$PY accept_shadow480.py
touch POSTPROCESS_COMPLETE

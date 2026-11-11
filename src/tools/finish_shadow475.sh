#!/bin/bash
set -eu
cd /home/scv/shared/phase475_20261107
test -f A/network_run/BOUNDARY_DIAG.json
test -f B/network_run/ABLATION_SPATIAL_MEAN.json
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY compare_shadow475.py
$PY accept_shadow475.py
touch POSTPROCESS_COMPLETE

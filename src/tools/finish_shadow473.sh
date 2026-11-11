#!/bin/bash
set -eu
cd /home/scv/shared/phase473_20261107
test -f B/network_run/ABLATION_APPEARANCE_OFF.json
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY ablate_uniform473.py B
$PY diagnose_contrast473.py
$PY compare_shadow473.py
$PY accept_shadow473.py
touch POSTPROCESS_COMPLETE

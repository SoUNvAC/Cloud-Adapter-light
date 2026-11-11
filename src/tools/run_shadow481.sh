#!/bin/bash
set -eu
cd /home/scv/shared/phase481_20261108
ARM="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "${ARM}_EXIT_CODE"' EXIT
for SEED in 45202 45203; do
 if [ "$ARM" = C ]; then
  $PY train_control481.py "$ARM" "$SEED"
 else
  $PY train_shadow481.py "$ARM" "$SEED"
 fi
 $PY verify_shadow481.py "$ARM" "$SEED"
 $PY boundary_shadow481.py "$ARM" "$SEED"
 if [ "$ARM" != C ]; then $PY ablate_shadow481.py "$ARM" "$SEED"; fi
 touch "seed_${SEED}/${ARM}/DONE"
done

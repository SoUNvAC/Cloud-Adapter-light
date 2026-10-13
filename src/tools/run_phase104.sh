#!/usr/bin/env bash
set -euo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
export GDAL_NUM_THREADS=1 OPJ_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
root=/home/scv/shared/data/pixbox_s2_cmix_5036991/phase104_20261013
coordinates=/home/scv/shared/phase103_positions_20261013/coordinate_metadata.csv
test -d "$root/run_logs"
test ! -e "$root/run_logs/EXIT_CODE"
echo $$ > "$root/run_logs/PID"
trap 'code=$?; printf "%s\n" "$code" > "$root/run_logs/EXIT_CODE"' EXIT
python src/tools/download_phase104.py --root "$root"
printf '0\n' > "$root/run_logs/DOWNLOAD_EXIT_CODE"
python src/tools/audit_phase104_inputs.py --root "$root" --coordinates "$coordinates"
python src/tools/verify_phase104_cpu.py --root "$root" --coordinates "$coordinates"

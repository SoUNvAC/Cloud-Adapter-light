#!/usr/bin/env bash
set -euo pipefail
cd /home/scv/Cloud-Adapter-light
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
root=/home/scv/shared/phase93_environment_20261012
# No original conda mutation, pip cache or installation outside the allowed root.
if [ -e "$root" ]; then
    test -f "$root/READY"
    "$root/venv/bin/python" -c 'import sklearn; assert sklearn.__version__ == "1.5.2"'
    exit 0
fi
mkdir "$root"
python -m venv "$root/venv"
mkdir "$root/wheels"
"$root/venv/bin/python" -m pip download --no-cache-dir --only-binary=:all: --dest "$root/wheels" -r src/tools/phase93_requirements.txt
"$root/venv/bin/python" -m pip install --no-cache-dir --no-index --find-links "$root/wheels" -r src/tools/phase93_requirements.txt
"$root/venv/bin/python" -m pip check
"$root/venv/bin/python" -c 'import sklearn; assert sklearn.__version__ == "1.5.2"'
touch "$root/READY"

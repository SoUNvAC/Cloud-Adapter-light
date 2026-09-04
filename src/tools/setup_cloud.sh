#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip setuptools wheel

# Use the exact PyTorch minor version against which the official MMCV wheel
# was compiled. Mixing (for example) torch 2.1.2 with a torch 2.1.0 MMCV
# extension can produce unresolved c10::SymInt symbols at import time.
python -m pip install \
  torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install xformers==0.0.22.post7
python -m pip install "numpy<2" opencv-python-headless

python -m pip install openmim
python -m mim install "mmengine==0.10.7"
python -m pip uninstall -y mmcv mmcv-lite mmcv-full || true
python -m pip install --no-cache-dir "mmcv==2.1.0" \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
python -m mim install "mmdet==3.3.0"
python -m mim install "mmsegmentation==1.2.2"
python -m pip install -r requirements-cloud.txt

python - <<'PY'
import torch
import mmcv
import mmengine
import mmdet
import mmseg
from mmcv.ops import point_sample  # noqa: F401

print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("mmcv:", mmcv.__version__)
print("mmengine:", mmengine.__version__)
print("mmdet:", mmdet.__version__)
print("mmseg:", mmseg.__version__)
print("MMCV compiled ops: OK")
PY

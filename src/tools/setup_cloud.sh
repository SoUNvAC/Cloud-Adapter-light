#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip setuptools wheel

# This conservative stack has pre-built CUDA 12.1 wheels and is compatible
# with the OpenMMLab APIs used by Cloud-Adapter.
python -m pip install \
  torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install xformers==0.0.23.post1

python -m pip install openmim
python -m mim install "mmengine==0.10.7"
python -m mim install "mmcv==2.1.0"
python -m mim install "mmdet==3.3.0"
python -m mim install "mmsegmentation==1.2.2"
python -m pip install -r requirements-cloud.txt

python - <<'PY'
import torch
import mmcv
import mmengine
import mmdet
import mmseg

print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("mmcv:", mmcv.__version__)
print("mmengine:", mmengine.__version__)
print("mmdet:", mmdet.__version__)
print("mmseg:", mmseg.__version__)
PY

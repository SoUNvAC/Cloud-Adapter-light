#!/usr/bin/env bash
set -euo pipefail

conda_hook="${PHASE64B_CONDA_HOOK:-/home/scv/miniconda3/etc/profile.d/conda.sh}"
source "$conda_hook"
conda activate cloud-lite-pt210

ROOT="${PHASE64B_CLOUDSEN_MEMMAP_ROOT:-../phase64b_data/cloudsen12_high_raw}"
OUT="${PHASE64B_DOWNLOAD_WORK_DIR:-work_dirs/phase64b_cloudsen_download}"
HF_ROOT="${PHASE64B_HF_HOME:-../phase64b_data/huggingface_home}"
export HF_HOME="$HF_ROOT"
export HF_HUB_CACHE="$HF_ROOT/hub"
export HF_XET_CACHE="$HF_ROOT/xet"
mkdir -p "$ROOT" "$OUT" "$HF_HUB_CACHE" "$HF_XET_CACHE"

python - "$ROOT" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

root = Path(sys.argv[1])
bands = ("B4", "B3", "B2", "B8", "B11", "B12")
for split in ("train", "val"):
    for band in bands:
        name = f"{split}/L1C_{band}.dat"
        print(f"downloading {name}", flush=True)
        hf_hub_download(
            repo_id="csaybar/CloudSEN12-high",
            filename=name,
            repo_type="dataset",
            local_dir=root,
        )

for forbidden in (root / "test",):
    if forbidden.exists():
        raise RuntimeError(f"target/source test material must remain absent: {forbidden}")
(root / "DEVELOPMENT_SPLITS_COMPLETE").write_text(
    "CloudSEN official train/val common-six bands only; test was not requested.\n",
    encoding="utf-8",
)
PY

touch "$OUT/DOWNLOAD_COMPLETE"

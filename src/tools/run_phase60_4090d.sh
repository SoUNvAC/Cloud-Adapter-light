#!/usr/bin/env bash
set -euo pipefail

cd /home/scv/Cloud-Adapter-light/src
OUT=work_dirs/phase60
mkdir -p "$OUT"

date -Is | tee "$OUT/STARTED_AT.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader | tee "$OUT/GPU_PREFLIGHT.txt"
else
    echo "nvidia-smi unavailable in PATH; CUDA validation delegated to torch" | tee "$OUT/GPU_PREFLIGHT.txt"
fi
python -c 'import torch; print({"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})' | tee "$OUT/RUNTIME_PREFLIGHT.txt"

python -u tools/run_phase60a_multispectral_audit.py 2>&1 | tee "$OUT/phase60a_console.log"
python -u tools/prepare_phase60b_support_audit.py 2>&1 | tee "$OUT/phase60b_prepare_console.log"
python -u tools/evaluate_phase60b_support_audit.py 2>&1 | tee "$OUT/phase60b_evaluate_console.log"

python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("work_dirs/phase60")
artifacts = {}
for name in ("phase60a_summary.json", "phase60b_selections.json", "phase60b_summary.json"):
    path = root / name
    if not path.is_file():
        raise SystemExit(f"missing required artifact: {path}")
    artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
summary = {
    "phase": 60,
    "status": "complete",
    "artifacts": artifacts,
    "phase60a_passed": json.loads((root / "phase60a_summary.json").read_text())["passed"],
    "phase60b_judgement": json.loads((root / "phase60b_summary.json").read_text())["key_judgement"],
}
(root / "phase60_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

date -Is | tee "$OUT/COMPLETED_AT.txt"
touch "$OUT/PHASE60_COMPLETE"

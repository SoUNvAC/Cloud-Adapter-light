#!/usr/bin/env bash
set -euo pipefail

cd /home/scv/Cloud-Adapter-light/src
source /home/scv/miniconda3/etc/profile.d/conda.sh
conda activate cloud-lite-pt210
OUT=work_dirs/phase61
mkdir -p "$OUT"
date -Is | tee "$OUT/STARTED_AT.txt"
python -c 'import torch; print({"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})' | tee "$OUT/RUNTIME_PREFLIGHT.txt"

python -u tools/run_phase61a_protocol_lineage.py 2>&1 | tee "$OUT/phase61a_console.log"
python -c 'import json; s=json.load(open("work_dirs/phase61/phase61a_lineage.json")); raise SystemExit(0 if s["passed"] else "Phase 61A lineage gate failed; 61B/61C/61D paused")'

python -u tools/run_phase61b_optimization_audit.py 2>&1 | tee "$OUT/phase61b_console.log"
python -u tools/run_phase61c_label_granularity_audit.py 2>&1 | tee "$OUT/phase61c_console.log"
python -u tools/prepare_phase61d_blind_review.py 2>&1 | tee "$OUT/phase61d_prepare_console.log"

python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("work_dirs/phase61")
paths = {
    "61A": root / "phase61a_lineage.json",
    "61B": root / "phase61b_optimization.json",
    "61C": root / "phase61c_granularity.json",
    "61D_packet": root / "blind_review/packet_summary.json",
}
artifacts = {}
for key, path in paths.items():
    if not path.is_file():
        raise SystemExit(f"missing {key}: {path}")
    artifacts[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
summary = {
    "phase": 61,
    "computational_audits_complete": True,
    "human_review_complete": False,
    "status": "awaiting_two_independent_phase61d_reviews",
    "artifacts": artifacts,
}
(root / "phase61_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

date -Is | tee "$OUT/COMPUTATIONAL_AUDITS_COMPLETED_AT.txt"
touch "$OUT/PHASE61_COMPUTATIONAL_COMPLETE"

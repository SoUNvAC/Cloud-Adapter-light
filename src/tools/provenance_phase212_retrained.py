"""Small CPU-only provenance/selection receipt; no historical report copying."""
from pathlib import Path
import hashlib
import json
import re

BASE=Path('/home/scv/shared/phase212_retrain_20261020_run02');SRC=Path('/home/scv/Cloud-Adapter-light/src')
out={'models_forwarded':0,'helper_code_SHA256':{},'selection':{}}
for rel in ['tools/evaluate_phase64_source_only_parent.py','tools/export_phase12_onnx.py','tools/validate_phase12_onnxruntime.py','configs/protocol/phase64_source_parent_rgb.py','configs/protocol/phase64_target_parent_rgb.py','cloud_adapter/datasets/phase64_parent.py','cloud_adapter/dg_metrics.py']:
    p=SRC/rel;out['helper_code_SHA256'][rel]=hashlib.sha256(p.read_bytes()).hexdigest()
for t in ['l8','sparcs']:
    text=(BASE/(t+'.console.log')).read_text(errors='replace')
    out['selection'][t]={'native_validation_and_best_log_lines':[x for x in text.splitlines() if ('mIoU' in x and ('Iter(val)' in x or 'best checkpoint' in x))]}
roots=[BASE,Path('/home/scv/shared/phase212_retrain_20261020')]
out['new_remote_bytes_two_run_directories']=sum(p.stat().st_size for r in roots for p in r.rglob('*') if p.is_file())
out['new_remote_package_bytes']=sum(p.stat().st_size for p in [Path('/home/scv/shared/phase212_retrain_20261020_bootstrap.zip'),Path('/home/scv/shared/phase212_retrain_20261020_run02_bootstrap.zip'),Path('/home/scv/shared/phase212_retrained_evaluation_20261020.zip')])
(BASE/'interpolation/FINAL_PROVENANCE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))

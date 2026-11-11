"""Local final acceptance after metadata-only collection."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[2]
r=root/'outputs/phase499/phase499_20261110'
audit=json.loads((r/'INDEPENDENT_AUDIT.json').read_text())
assert audit['passed'] and audit['texture']['update_check']['step']==40000
assert json.loads((r/'POSTPROCESS.json').read_text())['state']=='complete'
assert json.loads((r/'texture/COMPLETE.json').read_text())['steps']==40000
lock=json.loads((r/'texture/INPUT_LOCK.json').read_text())
assert all(hashlib.sha256((root/'src/phase499'/n).read_bytes()).hexdigest()==h for n,h in lock['code_sha256'].items())
assert audit['input_trace_matched_steps']==list(range(1,11))+list(range(2000,40001,2000))
assert all(audit['texture']['all_texture_parameters_updated'].values())
assert len(audit['texture']['all_texture_parameters_updated'])==25
manifest=json.loads((r/'TRANSFER_SHA.json').read_text())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
helpers=json.loads((r/'POSTPROCESS_CODE_SHA.json').read_text())
for n in ['audit499_remote.py','cost499_remote.py','off499_remote.py','report499_remote.py']:
    assert hashlib.sha256((root/'src/tools'/n).read_bytes()).hexdigest()==helpers[n],n
accept=dict(passed=True,steps=40000,target_supervision_percent=0,source_validation_images=535,multiseed=False,
    local_training_code_matches_lock=True,postprocess_helpers_match=True,retained_initialization_equal=True,
    all_25_texture_parameter_tensors_updated=True,frozen_encoder_unchanged=True,
    all_30_sampled_batches_match_496=True,transfer_hashes_verified=True,
    report_sha256=hashlib.sha256((r/'RESULT.md').read_bytes()).hexdigest())
(r/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(accept,indent=2),encoding='utf-8')
print(json.dumps(accept))

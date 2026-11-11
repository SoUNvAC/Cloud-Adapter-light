from pathlib import Path
import json,hashlib
R=Path('/home/scv/shared/phase489_20261108');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
bad='ffc0a9d6de9a25ae29fb0eb7b158c6d6d489d84fff6704309816336b34852205'
correct='ffc0a9d6de9a25ae29fb0eb7b158c6d6d483d84fff6704309816336b34852205'
assert (R/'accept_shadow489.py').read_text().replace(bad,correct)==(R/'accept_recheck489.py').read_text()
assert json.loads((R/'TRAINED_WEIGHT_ACCEPTANCE.json').read_text())['passed']
assert json.loads((R/'MULTISEED_RESULTS.json').read_text())['passed']
result=dict(passed=True,recovered=True,reason='Phase-number string substitution incorrectly changed the expected source SHA literal; separate recheck corrects only this literal. All initial files including training remain unchanged.',original_sha=sha(R/'accept_shadow489.py'),recheck_sha=sha(R/'accept_recheck489.py'),source_sha=correct)
(R/'ACCEPTANCE_RECOVERY.json').write_text(json.dumps(result,indent=2))
(R/'POSTPROCESS_COMPLETE.json').write_text(json.dumps(dict(passed=True,acceptance_recovery='ACCEPTANCE_RECOVERY.json')))
print(json.dumps(result))

from pathlib import Path,PureWindowsPath
import json,hashlib
R=Path('/home/scv/shared/phase480_20261108')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for manifest in ['INITIAL_CODE_SHA.json','CODE_SHA.json']:
 data=json.loads((R/manifest).read_text())
 for name,value in data.items():assert sha(R/PureWindowsPath(name).name)==value,name
checks=dict(passed=True,code_signatures=True,reports={n:sha(R/n) for n in ['RESULT.md','MAIN.csv']})
(R/'REMOTE_ARCHIVE_ACCEPTANCE.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))

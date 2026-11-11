"""Read-only compact217 status; zero model calls."""
from pathlib import Path
import json,os
r=Path('/home/scv/shared/phase217_20261023');s=json.loads((r/'CONTROLLER_STATUS.json').read_text()) if (r/'CONTROLLER_STATUS.json').exists() else {};child=r/s.get('child_root','_none');p=int((r/'PID').read_text())
try:os.kill(p,0);alive=True
except ProcessLookupError:alive=False
out={'PID':p,'alive':alive,'controller':s,'child':json.loads((child/'RUN_STATUS.json').read_text()) if (child/'RUN_STATUS.json').exists() else None,'EXIT_CODE':(r/'EXIT_CODE').read_text().strip() if (r/'EXIT_CODE').exists() else None}
if out['EXIT_CODE'] not in [None,'0']:
 out['failure_tail']=(r/'CONTROLLER_FAILURE.txt').read_text()[-2000:] if (r/'CONTROLLER_FAILURE.txt').exists() else ''
 if (child/'FAILURE.txt').exists():out['child_failure_tail']=(child/'FAILURE.txt').read_text()[-3500:]
print(json.dumps(out,indent=2))

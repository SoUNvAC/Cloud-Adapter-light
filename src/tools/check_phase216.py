"""Compact read-only Phase216 status. Never launches or retries work."""
from pathlib import Path
import json,os
r=Path('/home/scv/shared/phase216_20261023');p=int((r/'PID').read_text());alive=False
try:os.kill(p,0);alive=True
except ProcessLookupError:pass
result={'PID':p,'alive':alive,'status':json.loads((r/'RUN_STATUS.json').read_text()) if (r/'RUN_STATUS.json').exists() else None,'EXIT_CODE':(r/'EXIT_CODE').read_text().strip() if (r/'EXIT_CODE').exists() else None,'completed_arms':[a for a in ['C','D'] if (r/f'{a}_TRAIN_COMPLETE.json').exists()]}
if result['EXIT_CODE'] not in [None,'0']:result['failure_tail']=(r/'FAILURE.txt').read_text()[-4000:] if (r/'FAILURE.txt').exists() else (r/'error.log').read_text()[-4000:]
print(json.dumps(result,indent=2))

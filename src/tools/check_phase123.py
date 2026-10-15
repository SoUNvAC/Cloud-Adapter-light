"""One bounded, compact heartbeat snapshot; no model calls or log flooding."""
import json
from pathlib import Path
r=Path('/home/scv/shared/phase123_20261016')
def get(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None
out=dict(exists=r.exists(),controller=get(r/'CONTROLLER_STATUS.json'),pipeline_exit=(r/'PIPELINE_EXIT_CODE').read_text().strip() if (r/'PIPELINE_EXIT_CODE').exists() else None,delivery_ready=(r/'DELIVERY_RECEIPT.json').exists(),workers={})
if (r/'PID').exists():
    pid=int((r/'PID').read_text());q=Path('/proc')/str(pid)
    try:out['controller_alive']=(q/'stat').read_text().split()[2]!='Z' and 'phase123' in (q/'cmdline').read_text()
    except FileNotFoundError:out['controller_alive']=False
for name in ('train_C','C_Calibration','C_Catalogue','C_PixBox'):
    s=get(r/name/'RUN_STATUS.json')
    if s:out['workers'][name]={k:s[k] for k in ('status','step','all_calls','calls','main_calls','pool','product_id','patch_completed','elapsed_seconds','failure') if k in s}
f=get(r/'CONTROLLER_FAILURE.json')
if f:out['failure']=f
print(json.dumps(out,separators=(',',':')))

from collect_phase241 import ssh
print(ssh("""from pathlib import Path
import json
r=Path('/home/scv/shared/phase426_20261101');out={}
for n in ['STATUS.json','SELF_TEST.json','FAILURE.json','COMPLETE.json','REPORT_STATUS.json','REPORT_FAILURE.json','REPORT_COMPLETE.json']:
 if (r/n).exists():out[n]=json.loads((r/n).read_text())
for mode in ['baseline','competitive']:
 p=r/mode/'work_dirs/formal/M/STEP_TRACE.jsonl'
 if p.exists():
  lines=p.read_text().splitlines()
  if lines:
   row=json.loads(lines[-1]);out[mode]={k:row[k] for k in ['step','seconds','losses']}
 for p in sorted((r/mode/'evaluations/M').glob('step_*/METRICS.json')):
  v=json.loads(p.read_text());out[mode+'_'+p.parent.name]={k:v[k] for k in ['mIoU','IoU']}
 p=r/(mode+'.log')
 if p.exists() and (r/'FAILURE.json').exists():out[mode+'_log']=p.read_text()[-4000:]
print(json.dumps(out))
""").decode())

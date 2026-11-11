"""One compact read-only Phase208 snapshot."""
import json,os
from pathlib import Path
R=Path('/home/scv/shared/phase208_20261020')
def main():
 v={}
 for n in ('LAUNCH_DISPATCHED.json','CONTROLLER_STATUS.json','EXIT_CODE','FAILURE.json'):
  p=R/n
  if p.exists():v[n]=p.read_text().strip() if n=='EXIT_CODE' else json.loads(p.read_text())
 for stage in ('gate','train_M','train_B','infer','benchmark'):
  p=R/stage/'RUN_STATUS.json'
  if p.exists():
   s=json.loads(p.read_text());v[stage]={k:s[k] for k in ('status','PID','calls','cumulative_calls','stage_seconds','cumulative_stage_seconds','step','product_id','patch_completed','rss_peak_bytes','gpu_peak_reserved','host_min_free_bytes','new_bytes') if k in s}
   try:os.kill(s['PID'],0);v[stage]['alive']=True
   except ProcessLookupError:v[stage]['alive']=False
  p=R/stage/'FAILURE.json'
  if p.exists():v[stage+'_failure']=json.loads(p.read_text()).get('failure','')[-2300:]
 if (R/'CALL_LEDGER.jsonl').exists():v['cumulative_calls']=sum(1 for _ in (R/'CALL_LEDGER.jsonl').open())
 print(json.dumps(v))
if __name__=='__main__':main()

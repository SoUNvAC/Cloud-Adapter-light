"""One compact read-only remote snapshot; no model imports or inference."""
import json,os
from pathlib import Path
R=Path('/home/scv/shared/phase205_20261019')
def load(p):return json.loads(p.read_text())
def alive(pid):
 try:os.kill(int(pid),0);return True
 except (ProcessLookupError,ValueError):return False
def main():
 result={}
 for n in ('LAUNCH.json','CONTROLLER_STATUS.json','EXIT_CODE'):
  p=R/n
  if p.exists():result[n]=p.read_text().strip() if n=='EXIT_CODE' else load(p)
 for stage in ('gate','train_12201','train_12301','infer'):
  p=R/stage/'RUN_STATUS.json'
  if p.exists():
   s=load(p);result[stage]={k:s[k] for k in ('status','PID','step','calls','cumulative_calls','stage_seconds','cumulative_stage_seconds','pool','product_id','patch_completed','rss_peak_bytes','gpu_peak_reserved','host_min_free_bytes','new_bytes') if k in s};result[stage]['alive']=alive(s['PID'])
  p=R/stage/'FAILURE.json'
  if p.exists():result[stage+'_failure']=load(p).get('failure','')[-2600:]
 p=R/'CALL_LEDGER.jsonl'
 if p.exists():result['actual_cumulative_calls']=sum(1 for _ in p.open())
 if 'LAUNCH.json' in result:result['controller_alive']=alive(result['LAUNCH.json']['PID'])
 print(json.dumps(result))
if __name__=='__main__':main()

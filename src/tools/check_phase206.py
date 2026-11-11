"""Compact read-only Phase206 snapshot, no model imports."""
import json,os
from pathlib import Path
R=Path('/home/scv/shared/phase206_20261019')
def main():
 result={}
 for n in ('LAUNCH.json','CONTROLLER_STATUS.json','EXIT_CODE'):
  p=R/n
  if p.exists():result[n]=p.read_text().strip() if n=='EXIT_CODE' else json.loads(p.read_text())
 for stage in ('reconstruction','infer'):
  p=R/stage/'RUN_STATUS.json'
  if p.exists():
   s=json.loads(p.read_text());result[stage]={k:s[k] for k in ('status','PID','calls','cumulative_calls','cumulative_stage_seconds','pool','product_id','patch_completed','rss_peak_bytes','gpu_peak_reserved','host_min_free_bytes','new_bytes') if k in s}
   try:os.kill(int(s['PID']),0);result[stage]['alive']=True
   except ProcessLookupError:result[stage]['alive']=False
  p=R/stage/'FAILURE.json'
  if p.exists():result[stage+'_failure']=json.loads(p.read_text()).get('failure','')[-2200:]
 p=R/'CALL_LEDGER.jsonl'
 if p.exists():result['actual_calls']=sum(1 for _ in p.open())
 print(json.dumps(result))
if __name__=='__main__':main()

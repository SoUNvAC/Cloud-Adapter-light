"""One compact read-only remote status query, no model calls or polling loop."""
import json,os
from pathlib import Path
R=Path('/home/scv/shared/phase203_20261018')
def load(p):return json.loads(p.read_text())
def main():
 s=load(R/'RUN_STATUS.json') if (R/'RUN_STATUS.json').exists() else {};out=dict(launch=load(R/'LAUNCH.json') if (R/'LAUNCH.json').exists() else None,controller=s)
 for k in ('PID','worker_PID'):
  pid=s.get(k)
  if pid:
   try:os.kill(pid,0);out[k+'_exists']=True
   except ProcessLookupError:out[k+'_exists']=False
 if s.get('current'):
  d=R/s['current'];out['stage_counts']=[dict(stage=p.parent.name,**{k:v for k,v in load(p).items() if k in ('status','construction_calls','main_calls','warmup_calls','all_calls','patch_completed','elapsed_seconds','failure')}) for p in d.glob('R_*/RUN_STATUS.json')]
  if (d/'FAILURE.json').exists():out['failure']=load(d/'FAILURE.json')
  out['state_unchanged_complete']=(d/'COMPLETE.json').exists()
  if s.get('status')=='failed':
   if (d/'console.log').exists():out['console_tail']=(d/'console.log').read_text()[-5000:]
   if (d/'CONTROLLER_STATUS.json').exists():out['state_controller']=load(d/'CONTROLLER_STATUS.json')
   out['stage_evidence']={}
   for pool in ('Catalogue','PixBox'):
    stage=d/('R_'+pool);v={}
    for name in ('MODEL_INFO.json','SCORES_SEALED.json'):
     if (stage/name).exists():
      z=load(stage/name);v[name]={'file_count':len(z['scores_sha256'])} if name=='SCORES_SEALED.json' else z
    if (stage/'SCENE_RECEIPTS.json').exists():
     rows=load(stage/'SCENE_RECEIPTS.json');v['scene_receipts_count']=len(rows);v['scene_model_calls']=sum(z['model_calls'] for z in rows)
    if (stage/'EXIT_CODE').exists():v['exit_code']=(stage/'EXIT_CODE').read_text().strip()
    out['stage_evidence'][pool]=v
 if (R/'EXIT_CODE').exists():out['exit_code']=(R/'EXIT_CODE').read_text().strip()
 print(json.dumps(out,allow_nan=False))
if __name__=='__main__':main()

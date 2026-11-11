from pathlib import Path
import json,hashlib
from collect_phase241 import ssh
L=Path(__file__).resolve().parents[2]/'outputs/phase408/phase408_20261101';RR='/home/scv/shared/phase408_20261101'
def main():
 script=f"""from pathlib import Path
import json,os,csv
r=Path('{RR}');v={{}}
for n in ['STATUS.json','COMPLETE.json','FAILURE.json']:
 if (r/n).exists():v[n]=json.loads((r/n).read_text())
for seed in [40501,40502]:
 p=r/str(seed)/'RUN_STATUS.json'
 if p.exists():
  x=json.loads(p.read_text());v[str(seed)]={{k:x.get(k) for k in ['stage','current','calls_in_stage','updates_in_stage','cumulative_GPU_stage_seconds']}}
 points=sorted((r/str(seed)/'evaluations/H').glob('step_*/METRICS.json'),key=lambda p:int(p.parent.name.split('_')[1]))
 if points:
  p=points[-1];x=json.loads(p.read_text());v[str(seed)+'_last_validation']={{'step':int(p.parent.name.split('_')[1]),'mIoU':x['mIoU']}}
 p=r/str(seed)/'evaluation/STATUS.json'
 if p.exists():v[str(seed)+'_eval']=json.loads(p.read_text())
 p=r/str(seed)/'evaluation/summary.csv'
 if p.exists():
  with p.open() as f:v[str(seed)+'_metrics']=[{{k:x[k] for k in ['case','target_mIoU','source_mIoU']}} for x in csv.DictReader(f)]
 p=r/str(seed)/'work_dirs/formal/H/STEP_TRACE.jsonl'
 if p.exists():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-16384));lines=f.read().splitlines()
  if len(lines)>2:
   x=json.loads(lines[-2]);v[str(seed)+'_loss']={{'step':x['step'],'total':x['losses']['loss'],'penalty':x['losses'].get('loss_residual_energy')}}
if (r/'EXIT_CODE').exists():v['exit_code']=(r/'EXIT_CODE').read_text()
if v.get('exit_code')=='1':
 stage=v.get('STATUS.json',{{}}).get('stage');p=r/(str(stage)+'.log')
 if p.exists():v['log_tail']=p.read_text(errors='replace')[-4000:]
print(json.dumps(v))
"""
 data=ssh(script);(L/'LAST_STATUS.json').write_bytes(data);v=json.loads(data)
 compact={k:v[k] for k in ['STATUS.json','COMPLETE.json','FAILURE.json','exit_code','log_tail'] if k in v}
 for seed in ['40501','40502']:
  if seed in v:compact[seed]={'stage':v[seed]['stage'],'step':(v[seed].get('current') or {}).get('step')}
  for suffix in ['_last_validation','_eval','_loss']:
   if seed+suffix in v:compact[seed+suffix]=v[seed+suffix]
 print(json.dumps(compact))
 if v.get('exit_code')!='0' or (L/'TRANSFER_ACCEPTED.json').exists():return
 script=f"""from pathlib import Path
import json,hashlib
r=Path('{RR}');manifest=json.loads((r/'DELIVERY_MANIFEST.json').read_text());out={{}}
for n,h in manifest.items():
 b=(r/'delivery'/n).read_bytes();assert len(b)<32*1024**2 and hashlib.sha256(b).hexdigest()==h;out[n]={{'text':b.decode(),'SHA':h}}
print(json.dumps(out))
"""
 files=json.loads(ssh(script));manifest={}
 for n,v in files.items():
  p=L/n;assert p.resolve().is_relative_to(L.resolve());p.parent.mkdir(parents=True,exist_ok=True);b=v['text'].encode();assert hashlib.sha256(b).hexdigest()==v['SHA'];p.write_bytes(b);manifest[n]=v['SHA']
 (L/'TRANSFER_ACCEPTED.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()

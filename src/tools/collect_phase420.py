"""Read remote progress; transfer hashed text evidence after completion/failure."""
from pathlib import Path
import json,hashlib,sys
from collect_phase241 import ssh
PHASE=int(sys.argv[1]) if len(sys.argv)>1 else 412
L=Path(__file__).resolve().parents[2]/f'outputs/phase{PHASE}/phase{PHASE}_20261101'
RR=f'/home/scv/shared/phase{PHASE}_20261101'
def main():
 script=f"""from pathlib import Path
import json
r=Path('{RR}');v={{}}
for n in ['STATUS.json','SCREEN.json','PROMOTION.json','FAILURE.json','COMPLETE.json']:
 if (r/n).exists():v[n]=json.loads((r/n).read_text())
if (r/'EXIT_CODE').exists():v['exit_code']=(r/'EXIT_CODE').read_text()
mode=v.get('STATUS.json',{{}}).get('mode','')
p=r/mode/'RUN_STATUS.json'
if p.exists():
 x=json.loads(p.read_text());v['progress']={{'stage':x['stage'],'step':(x.get('current') or {{}}).get('step')}}
if v.get('exit_code')=='1':
 p=r/(mode+'.log') if mode else r/'controller.log'
 if p.exists():v['log_tail']=p.read_text(errors='replace')[-5000:]
print(json.dumps(v))
"""
 data=ssh(script);L.mkdir(parents=True,exist_ok=True);(L/'LAST_STATUS.json').write_bytes(data);v=json.loads(data)
 compact={k:v[k] for k in ['STATUS.json','progress','PROMOTION.json','FAILURE.json','exit_code','log_tail'] if k in v}
 compact['screen']={k:round(next((v for v in r['rows'] if v['step']==1000),r['rows'][-1])['mIoU'],4) for k,r in v.get('SCREEN.json',{}).items()}
 print(json.dumps(compact))
 if 'exit_code' not in v:return
 files=json.loads(ssh(f"""from pathlib import Path
import json,hashlib
r=Path('{RR}');out={{}}
for p in r.rglob('*'):
 if p.is_file() and p.suffix in ['.json','.jsonl','.py','.md','.csv'] and not any(s in p.parts for s in ['configs','scratch']):
  if p.stat().st_size>32*1024**2:continue
  b=p.read_bytes();out[p.relative_to(r).as_posix()]={{'text':b.decode(),'SHA':hashlib.sha256(b).hexdigest()}}
print(json.dumps(out))
"""))
 manifest={}
 for n,vv in files.items():
  p=L/n;assert p.resolve().is_relative_to(L.resolve());p.parent.mkdir(parents=True,exist_ok=True);b=vv['text'].encode();assert hashlib.sha256(b).hexdigest()==vv['SHA'];p.write_bytes(b);manifest[n]=vv['SHA']
 (L/'TRANSFER_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('text evidence transferred',len(manifest))
if __name__=='__main__':main()

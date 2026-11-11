"""Read retraining progress and download lightweight audit/results artifacts."""
from pathlib import Path
import json,io,tarfile,hashlib
from collect_phase241 import ssh
R=Path(__file__).resolve().parents[2]/'outputs/phase444/phase444_retrain_20261103'
if __name__=='__main__':
    R.mkdir(parents=True,exist_ok=True)
    status=json.loads(ssh("""from pathlib import Path
import json
r=Path('/home/scv/shared/phase444_retrain_20261103');out={}
for n in ['PRECHECK.json','LAUNCH.json','FAILURE.txt','EXIT_CODE']:
 p=r/n
 if p.exists():out[n]=json.loads(p.read_text()) if p.suffix=='.json' else p.read_text()
for arm in ['baseline429','source431']:
 out[arm]={}
 for n in ['INITIAL.json','UPDATE_CHECK.json','TRAIN_COMPLETE.json','EVAL_STATUS.json','EVAL_COMPLETE.json']:
  p=r/arm/n
  if p.exists():out[arm][n]=json.loads(p.read_text())
 p=r/(arm+'.log')
 if p.exists():out[arm]['log_tail']=p.read_text(errors='replace')[-1200:]
print(json.dumps(out))
"""));(R/'LAST_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status))
    raw=ssh("""from pathlib import Path
import json,hashlib,io,tarfile,sys
r=Path('/home/scv/shared/phase444_retrain_20261103');paths=[]
for p in r.rglob('*'):
 if p.is_file() and p.suffix in ['.json','.csv','.md','.py','.jsonl'] and p.stat().st_size<15000000 and 'predictions' not in p.parts:paths.append(p)
files={str(p.relative_to(r)):p.read_bytes() for p in paths};files['DELIVERY_SHA.json']=json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in files.items()}).encode();out=io.BytesIO()
with tarfile.open(fileobj=out,mode='w:gz') as t:
 for n,b in files.items():
  i=tarfile.TarInfo(n);i.size=len(b);t.addfile(i,io.BytesIO(b))
sys.stdout.buffer.write(out.getvalue())
""")
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(R,filter='data')
    for n,h in json.loads((R/'DELIVERY_SHA.json').read_text()).items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
    print('Snapshot transfer verified')

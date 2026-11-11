from pathlib import Path
import json,hashlib,io,tarfile,sys
from collect_phase241 import ssh
ARM=sys.argv[1];assert ARM in ['A','B']
R=Path(__file__).resolve().parents[2]/'outputs/phase456/phase456_20261106'/ARM/'network_run'
if __name__=='__main__':
    R.mkdir(parents=True,exist_ok=True)
    v=json.loads(ssh("""from pathlib import Path
import json
r=Path('/home/scv/shared/phase456_20261106/ARM_PLACEHOLDER/network_run');out={}
for n in ['STATUS.json','FAILURE.json','COMPLETE.json','BEST_SHADOW.json','BEST_PROTECTED.json','ZERO_INIT_CHECK.json','LOCK.json','HISTORY.json','UPDATE_CHECK.json']:
 p=r/n
 if p.exists():out[n]=json.loads(p.read_text())
print(json.dumps(out))
""".replace("ARM_PLACEHOLDER",ARM)));(R/'LAST_STATUS.json').write_text(json.dumps(v,indent=2))
    compact={k:v[k] for k in ['STATUS.json','FAILURE.json'] if k in v}
    compact['history']=[dict(step=x['step'],IoU=x['IoU'],mIoU=x['mIoU'],protected=x['protected'],success=x['success']) for x in v.get('HISTORY.json',[])]
    compact['complete']='COMPLETE.json' in v;print(json.dumps(compact))
    if not compact['complete']:raise SystemExit(0)
    b=ssh("""from pathlib import Path
import json,hashlib,tarfile,io,sys
r=Path('/home/scv/shared/phase456_20261106/ARM_PLACEHOLDER/network_run');files={str(p.relative_to(r)):p.read_bytes() for p in r.rglob('*') if p.is_file() and 'independent' not in p.parts and p.suffix in ['.json','.csv','.md','.npz','.jsonl']}
files['DELIVERY_SHA.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()}).encode();out=io.BytesIO()
with tarfile.open(fileobj=out,mode='w:gz') as t:
 for n,b in files.items():
  m=tarfile.TarInfo(n);m.size=len(b);t.addfile(m,io.BytesIO(b))
sys.stdout.buffer.write(out.getvalue())
""".replace("ARM_PLACEHOLDER",ARM))
    with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(R,filter='data')
    for n,h in json.loads((R/'DELIVERY_SHA.json').read_text()).items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
    (R/'TRANSFER_ACCEPTED.json').write_text(json.dumps(dict(passed=True)));print('TRANSFER_ACCEPTED')

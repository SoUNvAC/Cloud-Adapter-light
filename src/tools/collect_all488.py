from pathlib import Path
import json,hashlib,io,tarfile,subprocess
from collect_phase241 import ssh
R=Path(__file__).resolve().parents[2]/'outputs/phase488/phase488_20261108';R.mkdir(parents=True,exist_ok=True)
b=ssh("""from pathlib import Path
import json,hashlib,tarfile,io,sys
r=Path('/home/scv/shared/phase488_20261108');assert json.loads((r/'POSTPROCESS_COMPLETE.json').read_text())['passed']
files={str(p.relative_to(r)):p.read_bytes() for p in r.rglob('*') if p.is_file() and not {'independent','ablation_off','region_off','direction_off'}.intersection(p.parts) and p.suffix in ['.json','.csv','.md','.npz','.jsonl']}
files['DELIVERY_SHA.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()}).encode();out=io.BytesIO()
with tarfile.open(fileobj=out,mode='w:gz') as t:
 for n,b in files.items():
  m=tarfile.TarInfo(n);m.size=len(b);t.addfile(m,io.BytesIO(b))
sys.stdout.buffer.write(out.getvalue())
""")
with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(R,filter='data')
for n,h in json.loads((R/'DELIVERY_SHA.json').read_text()).items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
(R/'TRANSFER_ACCEPTED.json').write_text(json.dumps(dict(passed=True)))
print('TRANSFER_ACCEPTED',flush=True)
a=json.loads((R/'TRAINED_WEIGHT_ACCEPTANCE.json').read_text())
for arm,v in a['arms'].items():
 d=R/arm/'network_run'
 for name,w in v['weights'].items():
  if name=='branch_0.pth':continue
  p=d/name
  if not p.exists():subprocess.run(['scp',f'gzs:/home/scv/shared/phase488_20261108/{arm}/network_run/{name}',str(p)],check=True)
  assert hashlib.sha256(p.read_bytes()).hexdigest()==w['sha256']
 print(arm,'selected weights verified',flush=True)

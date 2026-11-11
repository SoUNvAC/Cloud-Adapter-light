from pathlib import Path
import json,hashlib,io,tarfile
from collect_phase241 import ssh
R=Path(__file__).resolve().parents[2]/'outputs/phase481/phase481_20261108'
R.mkdir(parents=True,exist_ok=True)
b=ssh("""from pathlib import Path
import json,hashlib,tarfile,io,sys
r=Path('/home/scv/shared/phase481_20261108')
assert json.loads((r/'MULTISEED_RESULTS.json').read_text())['passed']
files={str(p.relative_to(r)):p.read_bytes() for p in r.rglob('*') if p.is_file() and not {'independent','ablation_off','appearance_off','precision_strict'}.intersection(p.parts) and p.suffix in ['.json','.csv','.md','.npz','.jsonl']}
files['DELIVERY_SHA.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()}).encode();out=io.BytesIO()
with tarfile.open(fileobj=out,mode='w:gz') as t:
 for n,b in files.items():
  m=tarfile.TarInfo(n);m.size=len(b);t.addfile(m,io.BytesIO(b))
sys.stdout.buffer.write(out.getvalue())
""")
with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(R,filter='data')
for n,h in json.loads((R/'DELIVERY_SHA.json').read_text()).items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
(R/'TRANSFER_ACCEPTED.json').write_text(json.dumps(dict(passed=True,files=len(json.loads((R/'DELIVERY_SHA.json').read_text())))))
print('TRANSFER_ACCEPTED')

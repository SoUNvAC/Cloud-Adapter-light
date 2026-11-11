from pathlib import Path
import json,hashlib,io,tarfile,sys
from collect_phase241 import ssh
ARM=sys.argv[1];assert ARM in ['msre_off','read_off','write_off','read_write_off','token_read_off','projector_off']
R=Path(__file__).resolve().parents[2]/'outputs/phase466/phase466_20261107'/ARM
if __name__=='__main__':
    R.mkdir(parents=True,exist_ok=True)
    b=ssh("""from pathlib import Path
import json,hashlib,tarfile,io,sys
r=Path('/home/scv/shared/phase466_20261107/ARM_PLACEHOLDER');files={str(p.relative_to(r)):p.read_bytes() for p in r.rglob('*') if p.is_file() and 'ablation_off' not in p.parts and p.suffix in ['.json','.csv','.md','.npz','.jsonl']}
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

from pathlib import Path
import sys,io,json,hashlib,tarfile
from collect_phase241 import ssh
R=Path(__file__).resolve().parents[2]/'outputs/phase443/phase443_20261103'
if __name__=='__main__':
    status=json.loads(ssh("""from pathlib import Path
import json
r=Path('/home/scv/shared/phase443_20261103');out={}
for n in ['STATUS.json','COMPLETE.json','official/STATUS.json','official/COMPLETE.json','official/FAILURE.json']:
 p=r/n
 if p.exists():out[n]=json.loads(p.read_text())
out['log']=(r/'run.log').read_text()[-2000:];print(json.dumps(out))
"""));print(json.dumps(status));(R/'LAST_STATUS.json').write_text(json.dumps(status,indent=2))
    if not status.get('COMPLETE.json',{}).get('passed'):raise SystemExit(0)
    raw=ssh("""from pathlib import Path
import json,tarfile,io,hashlib,sys
r=Path('/home/scv/shared/phase443_20261103');paths=[p for p in r.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.md','.npy','.npz']]
paths += [r/'official'/n for n in ['LOCK.json','SIGNATURES.json','COMPLETE.json']]
files={str(p.relative_to(r)):p.read_bytes() for p in paths}
files['DELIVERY_SHA.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()},indent=2).encode()
out=io.BytesIO()
with tarfile.open(fileobj=out,mode='w:gz') as t:
 for n,b in files.items():
  i=tarfile.TarInfo(n);i.size=len(b);t.addfile(i,io.BytesIO(b))
sys.stdout.buffer.write(out.getvalue())
""");(R/'delivery.tar.gz').write_bytes(raw)
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(R,filter='data')
    for n,h in json.loads((R/'DELIVERY_SHA.json').read_text()).items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
    (R/'TRANSFER_ACCEPTED.json').write_text(json.dumps(dict(passed=True,archive_sha=hashlib.sha256(raw).hexdigest())))
    print('TRANSFER_ACCEPTED')

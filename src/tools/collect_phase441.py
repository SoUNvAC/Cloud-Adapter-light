"""Fetch compact analysis and reproducibility signatures; feature arrays stay on D."""
from pathlib import Path
import sys,json,hashlib,tarfile,io
from collect_phase241 import ssh
R=Path(__file__).resolve().parents[2]/'outputs/phase441/phase441_20261103'
if __name__=='__main__':
    R.mkdir(exist_ok=True,parents=True)
    status=json.loads(ssh("""from pathlib import Path
import json
r=Path('/home/scv/shared/phase441_20261103');out={}
for n in ['STATUS.json','FAILURE.json','COMPLETE.json','REPORT_COMPLETE.json','INDEPENDENT_ACCEPTANCE.json']+[a+'/analysis/'+n for a in ['source','adapted','official'] for n in ['STATUS.json','FAILURE.json','COMPLETE.json']]:
 p=r/n
 if p.exists():out[n]=json.loads(p.read_text())
if (r/'report.log').exists():out['report.log']=(r/'report.log').read_text()[-1500:]
print(json.dumps(out))
"""));print(json.dumps(status));(R/'LAST_STATUS.json').write_text(json.dumps(status,indent=2))
    if not status.get('COMPLETE.json',{}).get('passed'):raise SystemExit(0)
    if (R/'TRANSFER_ACCEPTED.json').exists() and '--refresh' not in sys.argv:raise SystemExit(0)
    raw=ssh("""from pathlib import Path
import json,hashlib,tarfile,io,sys
r=Path('/home/scv/shared/phase441_20261103');paths=[p for p in r.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.md','.png','.pdf','.svg']]
for arm in ['source','adapted','official']:
 paths += [r/arm/n for n in ['LOCK.json','SIGNATURES.json','COMPLETE.json']]
 paths += [p for p in (r/arm/'analysis').iterdir() if p.is_file() and p.suffix in ['.json','.npz','.npy'] and '_fold' not in p.name]
files={str(p.relative_to(r)):p.read_bytes() for p in paths};assert sum(map(len,files.values()))<150*1024**2
files['DELIVERY_SHA.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()},indent=2).encode()
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as t:
 for n,v in files.items():
  i=tarfile.TarInfo(n);i.size=len(v);t.addfile(i,io.BytesIO(v))
sys.stdout.buffer.write(b.getvalue())
""");(R/'delivery.tar.gz').write_bytes(raw)
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(R,filter='data')
    for n,h in json.loads((R/'DELIVERY_SHA.json').read_text()).items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
    (R/'TRANSFER_ACCEPTED.json').write_text(json.dumps(dict(passed=True,archive_sha=hashlib.sha256(raw).hexdigest())))
    print('TRANSFER_ACCEPTED')

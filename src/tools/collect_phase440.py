"""Compact progress; on completion fetch checksummed tables and figure artifacts."""
from pathlib import Path
import hashlib
import io
import json
import tarfile
import sys
from collect_phase241 import ssh

ROOT=Path(__file__).resolve().parents[2]/'outputs/phase440/phase440_20261103'
REMOTE="""from pathlib import Path
import json
r=Path('/home/scv/shared/phase440_20261103');out={}
for name in ['STATUS.json','COMPLETE.json','FAILURE.json']+[a+'/'+n for a in ['source','adapted'] for n in ['STATUS.json','COMPLETE.json','FAILURE.json','HOOK_PARITY.json']]:
 p=r/name
 if p.exists():
  try:out[name]=json.loads(p.read_text())
  except (OSError,ValueError):pass
for name in ['controller.log','source.log','adapted.log']:
 p=r/name
 if p.exists():out[name]=p.read_text(errors='replace')[-1200:]
print(json.dumps(out))
"""

if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    status=json.loads(ssh(REMOTE));(ROOT/'LAST_STATUS.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status))
    if not status.get('COMPLETE.json',{}).get('passed') or ((ROOT/'TRANSFER_ACCEPTED.json').exists() and '--refresh' not in sys.argv):raise SystemExit(0)
    script="""from pathlib import Path
import io,tarfile,json,hashlib,sys
r=Path('/home/scv/shared/phase440_20261103')
paths=[p for p in r.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.md','.png','.pdf','.svg']]
paths += [r/a/n for a in ['source','adapted'] for n in ['LOCK.json','COMPLETE.json','IDENTITY_CHECK.json','ENDPOINT_CHECK.json','PROTOCOL.json','TRANSFORM_TESTS.json','SUMMARY.json']]
paths += [p for a in ['source','adapted'] for p in (r/a/'paired').glob('*.json')]
files={str(p.relative_to(r)):p.read_bytes() for p in paths};assert sum(map(len,files.values()))<100*1024**2
files['DELIVERY_SHA.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()},indent=2).encode()
out=io.BytesIO()
with tarfile.open(fileobj=out,mode='w:gz') as t:
 for name,data in files.items():
  item=tarfile.TarInfo(name);item.size=len(data);t.addfile(item,io.BytesIO(data))
sys.stdout.buffer.write(out.getvalue())
"""
    raw=ssh(script);(ROOT/'delivery.tar.gz').write_bytes(raw)
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers())
        t.extractall(ROOT,filter='data')
    for name,digest in json.loads((ROOT/'DELIVERY_SHA.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    (ROOT/'TRANSFER_ACCEPTED.json').write_text(json.dumps(dict(passed=True,archive_sha256=hashlib.sha256(raw).hexdigest())))
    print('TRANSFER_ACCEPTED')

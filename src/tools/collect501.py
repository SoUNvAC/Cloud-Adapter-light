"""Download only Phase501 reports, metrics and reproducibility metadata."""
import io,json,tarfile,hashlib
from pathlib import Path
from collect_phase241 import ssh
R=Path(__file__).resolve().parents[2]/'outputs/phase501/phase501_20261111'
R.mkdir(parents=True,exist_ok=True)
raw=ssh('''
import io,sys,tarfile,json,hashlib
from pathlib import Path
r=Path('/home/scv/shared/phase501_20261111')
items=[]
for p in r.rglob('*'):
 if p.is_file() and p.suffix in ('.json','.jsonl','.md','.csv') and p.stat().st_size<10*1024**2:items.append(p)
manifest={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in items}
m=json.dumps(manifest).encode()
out=io.BytesIO()
with tarfile.open(fileobj=out,mode='w:gz') as t:
 for p in items:t.add(p,arcname=str(p.relative_to(r)))
 info=tarfile.TarInfo('TRANSFER_SHA.json');info.size=len(m);t.addfile(info,io.BytesIO(m))
sys.stdout.buffer.write(out.getvalue())
''')
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
    assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and not m.issym() and not m.islnk() for m in t.getmembers())
    t.extractall(R,filter='data')
manifest=json.loads((R/'TRANSFER_SHA.json').read_text())
assert all(hashlib.sha256((R/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
print(json.dumps({'metadata_files':len(manifest),'path':str(R),'hashes_verified':True}))

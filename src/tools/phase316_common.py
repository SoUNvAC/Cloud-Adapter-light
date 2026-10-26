"""Phase316 metadata-only helpers. Scientific work is Linux/gzs only."""
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
R=Path(__file__).resolve().parents[2]
ROOT=Path('/home/scv/shared/phase316_20261026')
S=R/'src/research_results/phase316_20261026'
E=S/'evidence'
OLD=R/'src/research_results/phase102_20261013'
LOCAL=R/'outputs/phase316/phase316_20261026'
SOURCE='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
START=time.monotonic()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def table(p,rows,fields=None):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:'unknown' if v is None else json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def resource_check():
    assert sys.platform=='linux'
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    disk=shutil.disk_usage(ROOT if ROOT.exists() else ROOT.parent).free
    assert mem>=4*1024**3 and disk>=4*1024**3
    assert time.time()-load(ROOT/'START.json')['unix']<3600
    return dict(available_memory_bytes=mem,free_disk_bytes=disk,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)

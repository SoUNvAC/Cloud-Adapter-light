"""Frozen Phase317 paths and byte evidence; no data/model side effects on import."""
import csv,hashlib,json,os,shutil,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[2]
ROOT=Path('/home/scv/shared/phase317_20261026')
S=R/'src/research_results/phase317_20261026'
LOCAL=R/'outputs/phase317/phase317_20261026'
OLD=Path('/home/scv/shared/phase304_20261024')
SOURCE='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
ARMS=[('CMC','CB'),('SPARCS','PL'),('SPARCS','CB'),('L8','PL'),('L8','CB')]
C=('Surface','Cloud','Shadow')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(p,rows,fields=None):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader()
  for r in rows:w.writerow({k:'NA' if v is None else json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def ah(a):return hashlib.sha256(a.tobytes()).hexdigest()
def resource_check():
 import resource
 assert sys.platform=='linux';resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
 mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
 disk=shutil.disk_usage(ROOT).free;assert mem>=4*1024**3 and disk>=8*1024**3
 return dict(available_memory=mem,free_disk=disk,peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
def budget_bytes():return 2*sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())+1000000000
def inputs(pool):return load(ROOT/pool/'IMAGES.json')
def schedule(pool):return load(ROOT/pool/'SCHEDULE.json')['rows']
def append(p,v):
 with Path(p).open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n')

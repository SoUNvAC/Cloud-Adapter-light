"""Phase320 immutable cache analysis. Local callers only use bytes and hashes."""
from phase317_common import R,SOURCE,sha,load,read,ah
from pathlib import Path
import os,sys,time,json,shutil,csv,hashlib
ROOT=Path('/home/scv/shared/phase320_20261027');P317=Path('/home/scv/shared/phase317_20261026');P319=Path('/home/scv/shared/phase319_20261027');OLD=Path('/home/scv/shared/phase304_20261024')
LOCAL=R/'outputs/phase320/phase320_20261027';S=R/'src/research_results/phase320_20261027'
POOLS=['CMC','SPARCS','L8'];ARMS=['PL','CB'];EXPECTED={'SPARCS':152450,'L8':426552}
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
def scalar(x):
 if type(x).__module__.startswith('numpy') and hasattr(x,'item'):return x.item()
 raise TypeError(type(x).__name__)
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.writing');temp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False,default=scalar)+'\n',encoding='utf-8');temp.replace(p)
def append(p,v):
 with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(v,ensure_ascii=False,allow_nan=False,default=scalar)+'\n')
def table(p,rows,fields=None):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader()
  for r in rows:w.writerow({k:'NA' if v is None else json.dumps(v,ensure_ascii=False,default=scalar) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def budget_bytes():return 2*sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_relative_to(ROOT/'tmp'))+100000000
def guard():
 import resource
 assert sys.platform=='linux' and os.environ.get('CONDA_DEFAULT_ENV')=='cloud-lite-pt210' and 'torch' not in sys.modules
 resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
 mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024;disk=shutil.disk_usage(ROOT if ROOT.exists() else ROOT.parent).free
 assert mem>=4*1024**3 and disk>=4*1024**3
 if ROOT.exists():
  assert budget_bytes()<=1024**3
  assert sum(p.stat().st_size for p in (ROOT/'tmp').glob('*') if p.is_file())<=6*1024**3
  if (ROOT/'STARTED.json').exists():assert time.time()-load(ROOT/'STARTED.json')['epoch']<10800
 return dict(available_memory=mem,free_disk=disk,peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
def lock():return load(ROOT/'ANALYSIS_LOCK.json')
def bound(p):
 p=Path(p).resolve();assert p.is_relative_to(R) or p.is_relative_to(Path('/home/scv/shared'));return p
def open_npz(entry):
 import numpy as np
 p=bound(entry['path']);assert sha(p)==entry['sha256'];return np.load(p,allow_pickle=False)
def mean(values):
 values=[x for x in values if x is not None];return sum(values)/len(values) if values else None

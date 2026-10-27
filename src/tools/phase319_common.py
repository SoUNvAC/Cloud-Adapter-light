"""Phase319 fixed CPU-only paths and byte evidence. No scientific imports."""
from phase317_common import R,SOURCE,sha,load,write,read,table,ah,append
from pathlib import Path
import os,sys,time,shutil,json,hashlib
ROOT=Path('/home/scv/shared/phase319_20261027')
PRIOR=Path('/home/scv/shared/phase317_20261026')
LOCAL=R/'outputs/phase319/phase319_20261027'
S=R/'src/research_results/phase319_20261027'
POOLS=['SPARCS','L8']
EXPECTED={'SPARCS':152450,'L8':426552}
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',GDAL_NUM_THREADS='4')
def inputs(pool):return load(PRIOR/pool/'IMAGES.json')['fit']
def budget_bytes():return 2*sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())+100000000
def guard():
 import resource
 assert sys.platform=='linux' and 'torch' not in sys.modules
 resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
 mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
 disk=shutil.disk_usage(ROOT if ROOT.exists() else ROOT.parent).free
 assert mem>=4*1024**3 and disk>=4*1024**3
 if ROOT.exists():assert budget_bytes()<=2000000000,'two-host persistence upper bound exceeded'
 return dict(available_memory=mem,free_disk=disk,peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
def pin(p,kind):
 p=Path(p);assert p.resolve().is_relative_to(R) or p.resolve().is_relative_to(Path('/home/scv/shared'))
 entry=dict(path=str(p),kind=kind,bytes=p.stat().st_size,sha256=sha(p));append(ROOT/'ACCESS.jsonl',entry);return entry

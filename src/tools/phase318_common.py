"""Phase318 immutable paths; local use is limited to files and hashes."""
from phase317_common import R,OLD,SOURCE,C,sha,load,write,read,table,ah,append
import os,sys,time,json,hashlib,shutil
from pathlib import Path
ROOT=Path('/home/scv/shared/phase318_20261027')
PRIOR=Path('/home/scv/shared/phase317_20261026')
LOCAL=R/'outputs/phase318/phase318_20261027'
S=R/'src/research_results/phase318_20261027'
ARMS=[(p,a) for p in ['CMC','SPARCS','L8'] for a in ['Flip-CB','Conf-CB']]
def inputs(pool):return load(ROOT/pool/'IMAGES.json')
def schedule(pool):return load(ROOT/pool/'SCHEDULE.json')['rows']
def name(row):return row.get('product') or row['name']
def budget_bytes():return 2*sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())+1000000000
def resource_check():
 import resource
 assert sys.platform=='linux';resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
 mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
 disk=shutil.disk_usage(ROOT).free;assert mem>=4*1024**3 and disk>=8*1024**3
 return dict(available_memory=mem,free_disk=disk,peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)

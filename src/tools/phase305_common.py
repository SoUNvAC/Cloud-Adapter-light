"""Phase305 fixed CPU-only paths and byte-exact delivery helpers."""
import csv
import hashlib
import json
from pathlib import Path
R=Path(__file__).resolve().parents[2]
ROOT=Path('/home/scv/shared/phase305_20261025')
LOCAL=R/'outputs/phase305/phase305_20261025'
S=R/'src/research_results/phase305_20261025'
A=Path('/home/scv/shared/phase302_20261020')
B=Path('/home/scv/shared/phase304_20261024')
C=('Surface','Cloud','Shadow')
FIELDS=('forest/jungle','snow/ice','agricultural','urban/developed','coastal','hills/mountains','desert/barren','shrublands/plains','wetland/bog/marsh','open_water','enclosed_water')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:('NA' if v is None else json.dumps(v) if isinstance(v,(list,dict)) else v) for k,v in r.items()})


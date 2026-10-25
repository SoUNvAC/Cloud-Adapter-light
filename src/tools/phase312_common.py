"""Phase312 fixed paths and byte-exact metadata helpers; no data access at import."""
import csv
import hashlib
import json
import os
from pathlib import Path
import time
R=Path(__file__).resolve().parents[2]
ROOT=Path('/home/scv/shared/phase312_20261025')
LOCAL=R/'outputs/phase312/phase312_20261025'
S=R/'src/research_results/phase312_20261025'
PL=Path('/home/scv/shared/phase304_20261024')
OLD=Path('/home/scv/shared/phase302_20261020')
MODELS=('RGB3','RGB6Q','MS6')
BANDS=('B04','B03','B02','B08','B11','B12')
INDICES=(3,2,1,7,11,12)
C=('Surface','Cloud','Shadow')
SOURCE='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def ah(a):return hashlib.sha256(a.tobytes(order='C')).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def append(n,v):
    with (ROOT/n).open('a',encoding='utf-8') as f:f.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n')
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(p,rows):
    assert rows,str(p)
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:'NA' if v is None else json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def average(values):
    v=[x for x in values if x is not None];return sum(v)/len(v) if v else None

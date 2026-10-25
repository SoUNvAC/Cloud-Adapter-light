"""Phase314 metadata-only paths and byte evidence helpers; no scientific imports."""
import csv
import hashlib
import json
from pathlib import Path
R=Path(__file__).resolve().parents[2]
ROOT=Path('/home/scv/shared/phase314_20261025')
LOCAL=R/'outputs/phase314/phase314_20261025'
S=R/'src/research_results/phase314_20261025'
PL=Path('/home/scv/shared/phase304_20261024')
BASE=Path('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871')
SOURCE='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def table(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:'NA' if v is None else json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

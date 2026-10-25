"""Fixed Q90 experiment paths, hashes and ledgers; no data access on import."""
import csv
import hashlib
import json
from pathlib import Path
import os
import time

R=Path(__file__).resolve().parents[2]
ROOT=Path('/home/scv/shared/phase311_20261025')
OLD=Path('/home/scv/shared/phase302_20261020')
PREV=Path('/home/scv/shared/phase303_20261024')
SOURCE='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
C=('Surface','Cloud','Shadow')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def array_sha(a):return hashlib.sha256(a.tobytes(order='C')).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def append(name,v):
    with (ROOT/name).open('a',encoding='utf-8') as f:f.write(json.dumps(v,allow_nan=False)+'\n')
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for row in rows:w.writerow({k:('NA' if v is None else json.dumps(v) if isinstance(v,(list,dict)) else v) for k,v in row.items()})
def counts(p):return [int((p==c).sum()) for c in range(3)]
def call(kind):
    n=sum(1 for _ in (ROOT/'CALLS.jsonl').open()) if (ROOT/'CALLS.jsonl').exists() else 0
    assert n<2400,'Forward cap2400'
    append('CALLS.jsonl',dict(context=CALL_CONTEXT,kind=kind,PID=os.getpid(),epoch=time.time()))
def score_path(row):return Path(row['scores_path'])


PL=Path("/home/scv/shared/phase304_20261024")
LOCAL=R/"outputs/phase311/phase311_20261025"
S=R/"src/research_results/phase311_20261025"
CALL_CONTEXT="startup"
def set_context(name):
    global CALL_CONTEXT
    CALL_CONTEXT=name

SP=Path("/home/scv/shared/phase310_20261025")

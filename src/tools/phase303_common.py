"""Image-only fixed confidence diagnostics and file admission."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

R=Path('/home/scv/Cloud-Adapter-light')
OLD=Path('/home/scv/shared/phase302_20261020')
ROOT=Path('/home/scv/shared/phase303_20261024')
THRESHOLDS=(.50,.70,.80,.90)
CLASSES=('Surface','Cloud','Shadow')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def table(name,rows):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:'NA' if v is None else json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def count_classes(pred,mask):return [int(((pred==c)&mask).sum()) for c in range(3)]
def confidence(s):
    assert s.dtype==np.float32 and np.isfinite(s).all() and (s>=0).all()
    shifted=s-s.max(0,keepdims=True);p=np.exp(shifted);p/=p.sum(0,keepdims=True)
    total=s.sum(0);q=np.divide(s,total[None],out=np.zeros_like(s),where=total[None]>1e-12)
    return p.max(0),q.max(0),total,s.argmax(0).astype(np.uint8)


def inputs():
    rows=load(OLD/'FIT_IMAGES.json')['rows'];assert len(rows)==100
    schedule=load(OLD/'CROP_SCHEDULE.json')['rows'];assert len(schedule)==1000
    lock=load(OLD/'PSEUDO_LOCK.json');assert len(lock['rows'])==100
    return rows,schedule,{r['product']:r for r in lock['rows']}


def guard(rows):
    from phase302_images import image_guard
    stream=image_guard(rows,ROOT/'IMAGE_ONLY_ACCESS.jsonl')
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        path=str(args[0]).replace('\\','/').lower()
        denied=any(x in path for x in ('/phase302_20261020/evaluation/','/phase302_20261020/analysis',
            '/phase302_20261020/probes/','final_lora.pth','initial_lora.pth','endpoint_lock.json',
            '/research_results/phase302_20261020/result','per_group_metrics','error_events.csv'))
        if denied:raise PermissionError('Phase303 forbids student/development/truth reads: '+path)
    sys.addaudithook(audit)
    return stream


def numerical_tests():
    s=np.array([[[1,0,1]],[[0,0,1]],[[0,0,0]]],np.float32)
    a,b,total,pred=confidence(s)
    assert abs(float(a[0,0])-.5761168848)<1e-6 and b[0,0]==1
    assert total[0,1]==0 and int((s[:,0,2]==s[:,0,2].max()).sum())==2
    for scale in (.5,1,2):
        aa,bb,_,pp=confidence(s*np.float32(scale))
        assert np.array_equal(pp,pred) and np.allclose(bb,b,atol=1e-6)
    masks=[a>=t for t in THRESHOLDS]
    assert all(not (masks[i+1]&~masks[i]).any() for i in range(3))
    return dict(status='passed',example_softmax=float(a[0,0]),zero_sum=True,ties=True,
        B_scale_invariance=True,threshold_monotonicity=True)

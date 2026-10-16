"""Small-file-only SCD helpers. No raster, network or model library imports."""
import collections
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import time
import zipfile

for _thread_key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_key]='4'

SETTINGS = ('O20','C20','H10')
LAYERS = ('all','subcell 00','subcell 01','subcell 10','subcell 11',
          'C20 coverage=1','C20 coverage=2','C20 coverage>=3','C20 margin<300m','C20 margin>=300m')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def write(p,v): Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def csvwrite(p,rows):
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def status_table(out,message,status='部分完成'):
    lines=['|分层|有效点数|正／负数|AP有定义组数|H10−C20平均AP差 pp|C20−O20平均AP差 pp|最坏组及AP差 pp|备注|',
           '|---|---|---|---|---|---|---|---|']
    for layer in LAYERS:
        lines.append('|'+layer.replace('all','全体').replace('>=','≥')+'|NA|NA|NA|NA|NA|NA|'+message+'|')
    lines.append('|运行状态|—|—|—|—|—|—|'+status+'；'+message+'；执行端'+str(out)+'|')
    (out/'phasescd-1-result.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def near(a,b):
    assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<=1e-10),(a,b)


def load_inputs(root):
    lock=read(root/'INPUT_MANIFEST.json')
    for name,h in lock['files'].items():
        p=(root/name).resolve()
        assert p.is_relative_to(root.resolve()) and sha(p)==h,name
    assert sha(root/'coordinate_metadata.csv')==read(root/'INPUT_PATCH_PLAN.json')['coordinate_source_sha256']
    assert read(root/'AUTHOR_DEDUPLICATION_CORRECTION.json')['new_plan_sha256']==sha(root/'INPUT_PATCH_PLAN.json')
    assert read(root/'SCORES_SEALED.json')['plan_sha256']==sha(root/'INPUT_PATCH_PLAN.json')
    assert read(root/'EXECUTION_PLAN_SEAL.json')['sha256']==sha(root/'INPUT_PATCH_PLAN.json')
    return lock


def score_points(root):
    support=read(root/'COMMON_SUPPORT_LOCK.json')
    seal=read(root/'SCORES_SEALED.json')['score_sha256']
    old=read(root/'O20_SCORES_SEALED.json')['score_sha256']
    pids=sorted(old)
    assert len(pids)==16
    by={s:{} for s in SETTINGS}
    with zipfile.ZipFile(root/'point_scores_private.zip') as z:
        for setting in SETTINGS:
            for pid in pids:
                if setting=='O20':
                    p=root/'O20'/pid/'UNLABELLED_POINT_SCORES.json'
                    raw=p.read_bytes();expected=old[pid]
                else:
                    raw=z.read(setting+'/'+pid+'/UNLABELLED_POINT_SCORES.json')
                    expected=seal[setting+'/'+pid]
                assert hashlib.sha256(raw).hexdigest()==expected==support['all_score_sha256'][setting+'/'+pid]
                points=json.loads(raw)['points']
                for q in points:
                    assert q['product_id']==pid and q['point_id'] not in by[setting]
                    by[setting][q['point_id']]=q
            assert len(by[setting])==9500
    assert set(by['O20'])==set(by['C20'])==set(by['H10'])
    return by


def dedup(windows):
    for w in windows:
        assert len(w)==4 and all(type(v) is int for v in w) and w[0]<w[1] and w[2]<w[3]
    return list(dict.fromkeys(tuple(w) for w in windows))


def geometry(row,col,windows,resolution):
    margins=[min(row-r0+.5,r1-row-.5,col-c0+.5,c1-col-.5)*resolution
             for r0,r1,c0,c1 in windows if r0<=row<r1 and c0<=col<c1]
    assert margins,'common valid point without patch coverage'
    return len(margins),max(margins)


def memberships(r):
    n=int(r['C20_coverage_count'])
    return ('all','subcell '+r['subcell'], 'C20 coverage='+str(n) if n<3 else 'C20 coverage>=3',
            'C20 margin<300m' if float(r['C20_best_margin_m'])<300 else 'C20 margin>=300m')


def f32(v):return struct.unpack('f',struct.pack('f',float(v)))[0]


def metrics(y,scores):
    assert len(y)==len(scores) and all(v in (0,1) for v in y)
    scores=[f32(s) for s in scores]
    assert all(math.isfinite(s) for s in scores)
    p=sum(y);n=len(y)-p
    tp=sum(v and s>=.5 for v,s in zip(y,scores));fp=sum(not v and s>=.5 for v,s in zip(y,scores))
    ap=None
    reason='no_positive_and_no_negative' if not y else ('no_positive' if not p else ('no_negative' if not n else None))
    if p and n:
        bins=collections.defaultdict(lambda:[0,0])
        for label,s in zip(y,scores):bins[s][0]+=label;bins[s][1]+=1
        seen=positive=0;ap=0.
        for s in sorted(bins,reverse=True):
            a,b=bins[s];seen+=b;positive+=a;ap+=a/p*(positive/seen)
    return dict(positive=p,negative=n,AP=ap,AP_NA_reason=reason,TP=tp,FP=fp,
                recall=tp/p if p else None,FPR=fp/n if n else None,
                precision=tp/(tp+fp) if tp+fp else None,sparse=p<5 or n<20)


class Budget:
    def __init__(self,paths,prior=0):
        self.start=time.monotonic();self.prior=prior;self.paths=paths;self.peak=0;self.free=None;self.bytes=0
        if os.name!='nt':
            import resource
            resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
            os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    def receipt(self):
        if os.name=='nt':
            from phase122_local_resources import peak_rss
            rss=peak_rss();free=shutil.disk_usage(self.paths[0]).free
        else:
            import resource
            from install_phase111_runtime import host_free
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;free=host_free()
        self.peak=max(self.peak,rss);self.free=min(self.free or free,free)
        self.bytes=max(self.bytes,sum(p.stat().st_size for root in self.paths for p in root.rglob('*') if p.is_file()))
        elapsed=time.monotonic()-self.start
        assert self.peak<=4*1024**3 and self.free>=4000000000 and self.bytes<=256*1024**2 and self.prior+elapsed<=3600
        assert datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))<datetime.datetime(2026,10,20,tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
        return dict(stage_seconds=elapsed,cumulative_CPU_stage_seconds=self.prior+elapsed,RSS_peak_bytes=self.peak,
                    host_min_free_bytes=self.free,new_file_bytes=self.bytes,CPU_threads_max=4,
                    GPU_calls=0,network_downloads=0,model_or_image_reads=0,training=0,threshold_search=0)

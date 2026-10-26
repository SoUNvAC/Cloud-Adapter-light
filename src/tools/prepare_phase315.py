"""Image-only cached-teacher identity audit. Historical initial is inspected, never used to initialize."""
import ast
import difflib
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from datetime import datetime,timezone,timedelta
import shutil
import subprocess
import sys
import resource
import time
import numpy as np
import torch
from phase315_common import *
def ah(a):return hashlib.sha256(a.tobytes()).hexdigest()
def summary(delta):
    h=hashlib.sha256();shapes={};bzero=True
    for n,v in sorted(delta.items()):
        shapes[n]=list(v.shape)
        if n.endswith('.lora_a'):h.update(n.encode());h.update(v.contiguous().numpy().tobytes())
        else:assert n.endswith('.lora_b');bzero &= bool(torch.count_nonzero(v)==0)
    return dict(A_sha256=h.hexdigest(),B_all_zero=bzero,shapes=shapes,parameters=sum(v.numel() for v in delta.values()))
def main():
    started=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    assert sys.platform=='linux' and not ROOT.exists();assert datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')=='20261026'
    ROOT.mkdir();(ROOT/'code').mkdir();(ROOT/'tmp').mkdir();(ROOT/'cache').mkdir()
    inputs={};pin=lambda p:inputs.setdefault(str(p),sha(p))
    old=obj=load(PL/'EXECUTION_LOCK.json')
    for n in ['worker_phase304.py','execute_phase302.py','phase302_images.py','prepare_phase302.py','phase304_common.py']:
        p=R/'src/tools'/n;assert sha(p)==old['code']['src/tools/'+n];pin(p)
    subprocess.run(['git','diff','--exit-code','0e40461e2defaf405b340c4c66c9d0892cc766d5','--','src/cloud_adapter/models','src/configs'],cwd=R,check=True,capture_output=True)
    source=R/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth';assert sha(source)==SOURCE;pin(source)
    for n in ['FIT_IMAGES.json','EVAL_IMAGES.json','CROP_SCHEDULE.json','CROP_REPRODUCTION.csv','PSEUDO_LOCK.json','EVALUATION_LOCK.json','ENDPOINT_LOCK.json','EXECUTION_LOCK.json']:
        pin(PL/n)
        if n in ['FIT_IMAGES.json','EVAL_IMAGES.json','CROP_SCHEDULE.json']:shutil.copyfile(PL/n,ROOT/n)
    lock=load(PL/'PSEUDO_LOCK.json');assert lock['accepted']==75469158 and lock['classes']==[40622307,33554491,1292360]
    rows=load(ROOT/'FIT_IMAGES.json')['rows'];steps=load(ROOT/'CROP_SCHEDULE.json')['rows'];prior={int(r['attempt']):r for r in read(PL/'CROP_REPRODUCTION.csv')};crop=[]
    for i,row in enumerate(rows):
        e=next(r for r in lock['rows'] if r['product']==row['product']);score=Path(e['scores_path']);teacher=PL/'teacher'/(row['product']+'.npz')
        assert sha(score)==e['scores_sha256'] and sha(teacher)==e['sha256'];pin(score);pin(teacher)
        s=np.load(score,allow_pickle=False);assert ah(s)==e['scores_array_sha256']
        with np.load(teacher) as z:pseudo=z['pseudo'];valid=z['valid'];pred=z['pred'];maximum=z['maximum']
        total=s.sum(0);q=np.divide(s,total[None],out=np.zeros_like(s),where=total[None]>1e-12)
        assert np.array_equal(pred,s.argmax(0)) and np.array_equal(maximum,q.max(0));expected=np.where(valid&(total>1e-12)&(maximum>=.9),pred,255).astype(np.uint8);assert np.array_equal(pseudo,expected)
        assert [int((pseudo==c).sum()) for c in range(3)]==e['counts']
        for step in (t for t in steps if t['fit_index']==i):
            t,l=step['top'],step['left'];p=pseudo[t:t+512,l:l+512].copy()
            if step['horizontal_flip']:p=p[:,::-1].copy()
            counts=[int((p==c).sum()) for c in range(3)];ex=prior[step['attempt']];assert counts==[int(ex[c]) for c in ['Surface','Cloud','Shadow']]
            crop.append(dict(**step,product=row['product'],mask_sha256=ah(p),Surface=counts[0],Cloud=counts[1],Shadow=counts[2],accepted=sum(counts)))
    crop.sort(key=lambda r:r['attempt']);assert len(crop)==1000 and sum(r['accepted']>0 for r in crop)==955 and sum(r['accepted'] for r in crop)==181307540
    assert [sum(r[c]>0 for r in crop) for c in ['Surface','Cloud','Shadow']]==[524,644,458];table(ROOT/'CROP_IDENTITY.csv',crop)
    endpoint=load(PL/'ENDPOINT_LOCK.json');assert sha(PL/'initial_lora.pth')==endpoint['initial_sha256'];pin(PL/'initial_lora.pth');pin(PL/'final_lora.pth');assert sha(PL/'final_lora.pth')==endpoint['final_sha256']
    # Inspection only, in a separate CPU preparer; none of these tensors are passed to a new model.
    initial=summary(torch.load(PL/'initial_lora.pth',map_location='cpu'));assert initial['B_all_zero'] and initial['parameters']==294912
    write(ROOT/'HISTORICAL_INITIAL_IDENTITY.json',dict(seed=30201,evidence='CPU inspection of sealed historical initial only; never loaded into new student',initial_file_sha256=endpoint['initial_sha256'],**initial))
    for row in load(PL/'EVALUATION_LOCK.json')['rows']:
        p=Path(row.get('cache_path',PL/'evaluation'/('LoRA_'+row['product']+'.npz')));assert sha(p)==row['sha256'];pin(p)
    files=list((R/'src/tools').glob('*phase315*.py'))+[R/'src/tools'/n for n in ['worker_phase304.py','execute_phase302.py','prepare_phase302.py','phase302_images.py','phase304_common.py','analyze_phase304.py']]
    for p in files:shutil.copyfile(p,ROOT/'code'/p.name)
    for name in ['worker_phase315.py','analyze_phase315.py']:
        oldfile=R/'src/tools'/('worker_phase304.py' if name.startswith('worker') else 'analyze_phase304.py');newfile=R/'src/tools'/name
        if newfile.exists():(ROOT/'code'/(name+'.diff')).write_text(''.join(difflib.unified_diff(oldfile.read_text().splitlines(True),newfile.read_text().splitlines(True),fromfile=oldfile.name,tofile=newfile.name)))
    write(ROOT/'EXECUTION_LOCK.json',dict(code={p.relative_to(R).as_posix():sha(p) for p in files},inputs=inputs,protocol_sha256=sha(R/'src/research_plans/phase315.md'),start_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),new_seeds=[31501,31502],historical_seed=30201,crop_identity_sha256=sha(ROOT/'CROP_IDENTITY.csv'),runtime=sys.executable,cap_forward=2800,cap_gpu_seconds=28800,cap_cpu_bytes=16*1024**3,min_free_bytes=4000000000,cap_persistent_all_copies=6000000000,truth_access='only independent CPU after both endpoints/evaluations sealed',authorization='User execute315 plus persistent code/results synchronization; no new methods/pool evaluation'))
    write(ROOT/'STAGES.json',{});write(ROOT/'STATUS.json',dict(status='prepared',new_seeds=[31501,31502],teacher_rebuild_forwards=0))
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024;assert available>=4000000000 and shutil.disk_usage(ROOT).free>=4000000000
    write(ROOT/'PREPARE_RESOURCE.json',dict(seconds=time.monotonic()-started,peak_cpu_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,host_available_end=available,neural_forwards=0))
    print('Phase315 prepared: 100 products; all1000crop masks and original counters exact')
if __name__=='__main__':main()

"""Post-seal image-only sample/support reproduction and independent CSV accounting; no fitting."""
import resource
import shutil
import sys
import traceback
from datetime import datetime
import numpy as np
import scipy
from phase312_common import *
from images_phase312 import guard,image

def limits(started,prior):
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    free=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    assert peak<16*1024**3 and free>=4000000000 and shutil.disk_usage(ROOT).free>=4000000000
    assert time.monotonic()-started+prior<=7200
    return peak,free

def main():
    assert R==Path('/home/scv/Cloud-Adapter-light') and sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
    assert load(ROOT/'STATUS.json')['status']=='completed' and not (ROOT/'AUDIT_ACCEPTANCE.json').exists()
    started=time.monotonic();prior=load(ROOT/'STATUS.json')['total_seconds'];minimum=2**63
    code=load(ROOT/'EXECUTION_LOCK.json')
    for n,h in code['code'].items():assert sha(R/n)==h,n
    for n,h in code['inputs'].items():assert sha(n)==h,n
    fit=load(ROOT/'FIT_LOCK.json');frozen={str(ROOT/'models'/(n+'.json')):h for n,h in fit['models'].items()};frozen[str(ROOT/'FIT_ATTEMPTS.jsonl')]=sha(ROOT/'FIT_ATTEMPTS.jsonl')
    numeric={p.name:sha(p) for p in ROOT.glob('*.csv')}
    with np.load(ROOT/'sampling.npz') as z:raw_saved=z['TOA'];label=z['y_Surface'];flat=z['flat_indices'];group=z['product_index'];w=z['weight']
    rows=sorted(load(ROOT/'FIT_IMAGES.json')['rows'],key=lambda r:r['product']);support={r['product']:r for r in read(ROOT/'FIT_SUPPORT.csv')};old={r['product']:r for r in load(PL/'PSEUDO_LOCK.json')['rows']}
    # Own process cannot read student outputs or any reference, including the earlier post-seal audit.
    def deny(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)) and '/evaluation/lora_' in str(args[0]).lower():raise PermissionError('No student inputs in fit-support audit')
    sys.addaudithook(deny);guard('audit',rows);rng=np.random.Generator(np.random.PCG64(31201));records=[]
    for i,row in enumerate(rows):
        entry=old[row['product']];assert sha(entry['scores_path'])==entry['scores_sha256'];s=np.load(entry['scores_path']);six,rgb,v6=image(row,'audit')
        total=s.sum(0);pred=s.argmax(0);maximum=np.divide(s.max(0),total,out=np.zeros_like(total),where=total>1e-12)
        candidate=rgb&v6&(total>1e-12)&(maximum>=np.float32(.9))&(pred<2);eligible=np.flatnonzero(candidate)
        chosen=np.sort(rng.choice(eligible,min(8192,len(eligible)),replace=False));ix=group==i
        assert np.array_equal(chosen,flat[ix]) and np.array_equal((pred.ravel()[chosen]==0).astype(np.uint8),label[ix])
        assert np.array_equal(six.reshape(-1,6)[chosen],raw_saved[ix])
        ex=support[row['product']];assert ah(chosen)==ex['sample_flat_SHA'] and ah(label[ix])==ex['sample_label_SHA']
        assert abs(float(w[ix].sum())-float(ex['product_weight']))<1e-12
        records.append(dict(product=row['product'],V6_mask_SHA=ah(v6),RGB_valid_SHA=ah(rgb),eligible_flat_SHA=ah(eligible),eligible=len(eligible),sampled=len(chosen),sample_flat_SHA=ah(chosen),sample_label_SHA=ah(label[ix]),raw_sample_features_identical=True,truth_reads=False))
        peak,free=limits(started,prior);minimum=min(minimum,free)
    # Independent weighted moments from the sealed selected raw values, no solver calls.
    for name in MODELS:
        model=load(ROOT/'models'/(name+'.json'))
        if model['status']!='passed':continue
        x=raw_saved.astype(np.float64)
        if name=='RGB3':x=x[:,:3]
        elif name=='RGB6Q':x=np.c_[x[:,:3],x[:,:3]**2]
        mean=np.average(x,axis=0,weights=w);variance=np.average((x-mean)**2,axis=0,weights=w);scale=np.sqrt(variance);constant=np.all(x==x[0],axis=0);mean[constant]=x[0,constant];scale[constant]=1
        assert np.allclose(mean,model['result']['mean'],rtol=1e-12,atol=1e-12) and np.allclose(scale,model['result']['scale'],rtol=1e-12,atol=1e-12)
    cells={}
    for r in read(ROOT/'TRANSITIONS27.csv'):
        key=r['product'],r['comparison'];idx=tuple(C.index(r[k]) for k in ('truth','Source_pred','student_pred'));t=cells.setdefault(key,{})
        assert idx not in t;t[idx]=int(r['count'])
    assert all(len(t)==27 for t in cells.values())
    for r in read(ROOT/'ERROR_FLOW.csv'):
        t=cells[r['product'],r['comparison']];a,b=r['error'].split('->');c=C.index(a);j=C.index(b)
        old=sum(t[c,j,k] for k in range(3));new=sum(t[c,k,j] for k in range(3));same=t[c,j,j]
        values=dict(Source_errors=old,student_errors=new,removed=old-same,added=new-same,net_reduction=old-new,corrected=t[c,j,c],changed_to_other_error=old-same-t[c,j,c],from_correct=t[c,c,j])
        assert all(int(r[k])==v for k,v in values.items())
    selection=read(ROOT/'SELECTION.csv')
    for row in load(ROOT/'READOUT_LOCK.json')['rows']:
        with np.load(row['readout_file']) as z:
            ids=z['flat_indices'];c=np.unpackbits(z['C_pack'],count=len(ids)).astype(bool);ms=None
            if 'MS6_90' in row['rules']:ms=int(np.unpackbits(z['MS6_90'],count=len(ids)).sum())
            for rule in row['rules']:
                choose=np.unpackbits(z[rule],count=len(ids)).astype(bool);assert not (choose&~c).any()
                ex=next(r for r in selection if r['product']==row['product'] and r['rule']==rule);assert int(choose.sum())==int(ex['selected_image'])
                if rule.endswith('equalMS6'):assert int(choose.sum())==ms
                assert sum(int(ex[k]) for k in ('selected_D','selected_K','selected_Z'))==int(ex['changed_reference'])
    assert all(sha(n)==h for n,h in frozen.items()) and all(sha(ROOT/n)==h for n,h in numeric.items())
    assert 'torch' not in sys.modules
    table(ROOT/'FIT_IMAGE_SUPPORT_AUDIT.csv',records);peak,free=limits(started,prior);minimum=min(minimum,free)
    script=R/'src/tools/audit_phase312.py';shutil.copyfile(script,ROOT/'code'/script.name)
    write(ROOT/'AUDIT_ACCEPTANCE.json',dict(status='passed',same_PCG64_sample_ids_labels_and_raw_features_all_100=True,weighted_scaler_independent_check=True,all_numeric_CSV_unchanged=numeric,models_and_fit_attempt_ledger_unchanged=frozen,fit_calls_added=0,neural_calls=0,independent_errorflow_and_equal_quantity_checks=True,reference_or_student_read=False,additional_seconds=time.monotonic()-started,total_science_plus_audit_seconds=prior+time.monotonic()-started,peak_cpu_bytes=peak,minimum_available_memory=minimum,audit_code_SHA=sha(script),runtime=dict(numpy=np.__version__,scipy=scipy.__version__,python=sys.version,blas_threads={n:os.environ.get(n) for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})))
    print('Phase312 independent image/support/table audit passed; no extra fit or forward')

if __name__=='__main__':
    try:main()
    except BaseException:append('AUDIT_FAILURES.jsonl',dict(traceback=traceback.format_exc(),action='Preserve state; no blind retry'));raise

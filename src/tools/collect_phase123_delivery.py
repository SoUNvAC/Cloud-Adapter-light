"""Independent full calibration check and shared-only evaluation delivery.

Full calibration score arrays stay in controlled shared with their sealed SHAs.
Only evaluation arrays must travel to the machine holding evaluation references.
"""
import resource,time,zipfile,io
import numpy as np
from phase123_common import *
def main():
    setup();assert (ROOT/'EXIT_CODE').read_text().strip()=='0'
    assert not (ROOT/'phase123_delivery.zip').exists()
    from install_phase111_runtime import host_free
    host_before=host_free();assert host_before>=4000000000
    start=time.time();lock=load(ROOT/'THRESHOLD_LOCK.json');groups=load(REPO/'src/research_results/phase122_20261015/MEMBER_INPUT_LOCK.json')['calibration']
    reports={};sources=[]
    from phase119_common import CAT
    with zipfile.ZipFile(CAT/'masks.zip') as masks:
        for arm in ('C',):
            t=np.float64(lock['thresholds'][arm]);point=lock['calibration'][arm];b=np.float64(point['unsafe_negative_boundary'])
            rates=[];low=[];ties=[];details=[];count=0
            seals=load(ROOT/(arm+'_Calibration')/'SCORES_SEALED.json')['scores_sha256']
            for g in sorted(groups,key=lambda g:g['group_id']):
                path=ROOT/(arm+'_Calibration')/'scores/Calibration'/(g['product']+'.npz');assert sha(path)==seals['Calibration/'+g['product']]
                with np.load(path,allow_pickle=False) as z:
                    assert z['valid'].all();idx=z['flat_indices'];s=z['scores'].astype(np.float64)
                raw=masks.read('masks/'+g['product']+'.npy');assert __import__('hashlib').sha256(raw).hexdigest()==g['mask_member_sha256']
                y=np.load(io.BytesIO(raw),allow_pickle=False).reshape(-1,3)[idx]
                assert (y.sum(1)==1).all();neg=~y[:,2];n=int(neg.sum());fp=int(((s>=t)&neg).sum());lower=int(((s>=b)&neg).sum())
                tie=int(((s==b)&neg).sum());count+=n;ties.append(tie)
                if n:rates.append(fp/n);low.append(lower/n)
                details.append(dict(group_id=g['group_id'],negative=n,FP=fp,lower_FP=lower,tie=tie))
                sources.append(dict(model=arm,group_id=g['group_id'],controlled_shared_path=str(path),sha256=sha(path),bytes=path.stat().st_size))
                assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=16*1024**3
            assert len(rates)==point['negative_groups'] and len(groups)==53
            assert sum(rates)/len(rates)<=.01 and sum(low)/len(low)>.01 and np.nextafter(t,-np.inf)==b
            assert abs(sum(rates)/len(rates)*100-point['macro_negative_FP_pp'])<1e-10 and ties==point['atomic_negative_tie_counts']
            reports[arm]=dict(negative_groups=len(rates),negative_pixels=count,macro_FP_pp=sum(rates)/len(rates)*100,
                adjacent_lower_macro_FP_pp=sum(low)/len(low)*100,atomic_ties=ties,rows=details,float64_domain=True)
    write(ROOT/'CALIBRATION_INDEPENDENT_CPU_ACCEPTANCE.json',dict(status='passed',execution='separate controlled-shared CPU process',
        models=reports,source_sha256=sha(Path(__file__)),threshold_lock_sha256=sha(ROOT/'THRESHOLD_LOCK.json'),
        full53_groups_no_sampling=True,uses_no_binary_threshold_search=True,model_calls=0,seconds=time.time()-start,
        CPU_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    write(ROOT/'CALIBRATION_SCORE_LOCATIONS.json',dict(files=sources,complete_full_arrays_retained_in_shared=True,
        not_transferred_with_evaluation='avoid redundant calibration arrays; complete independent direct CPU boundary proof included'))
    files=[p for p in ROOT.rglob('*') if p.is_file() and (p.suffix=='.json' or ('scores' in p.parts and p.suffix=='.npz' and 'Calibration' not in p.parts))]
    # Stage component is R_Calibration/C_Calibration; pool component is Calibration.
    assert all(p.suffix!='.npz' or 'Calibration' not in p.parts for p in files)
    write(ROOT/'DELIVERY_MANIFEST.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        no_reference_labels=True,shared_only=True,calibration_arrays_retained_in_controlled_shared=True))
    path=ROOT/'phase123_delivery.zip'
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files+[ROOT/'DELIVERY_MANIFEST.json']:z.write(p,p.relative_to(ROOT).as_posix())
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<=4000000000
    write(ROOT/'DELIVERY_RECEIPT.json',dict(bytes=path.stat().st_size,sha256=sha(path),files=len(files)+1,
        intermediate_bytes_after_package=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
        decoder=dict(bytes=(ROOT/'C_decoder_final.safetensors').stat().st_size,sha256=sha(ROOT/'C_decoder_final.safetensors'))))
    assert host_free()>=4000000000
    print(load(ROOT/'DELIVERY_RECEIPT.json'))
if __name__=='__main__':main()

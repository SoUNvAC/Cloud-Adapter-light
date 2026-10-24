"""Diagnosed evaluation-only recovery: verify image manifest before installing image guard.
Original sealed worker, training, endpoint, rules and failure artifacts remain immutable.
"""
from worker_phase304 import *

def evaluate():
    endpoint=load(ROOT/'ENDPOINT_LOCK.json');assert endpoint['status']=='sealed' and sha(ROOT/'final_lora.pth')==endpoint['final_sha256']
    assert sha(ROOT/'PSEUDO_LOCK.json')==endpoint['pseudo_lock_sha256']
    assert sha(ROOT/'EVAL_IMAGES.json')==sha(OLD/'EVAL_IMAGES.json')
    rows=load(ROOT/'EVAL_IMAGES.json')['rows'];image_guard(rows,ROOT/'EVAL_ACCESS.jsonl')
    old=load(OLD/'EVALUATION_LOCK.json');oldendpoint=load(OLD/'ENDPOINT_LOCK.json')
    assert oldendpoint['source_sha256']==SOURCE and load(OLD/'SOURCE_ADMISSION.json')['source_sha256']==SOURCE
    base.seeded();model,state,names,_=base.construct(True)
    assert base.frozen_digest(model,names)==endpoint['base_initial_final_sha256']==oldendpoint['base_initial_final_sha256']
    delta=torch.load(ROOT/'final_lora.pth',map_location='cpu');assert set(delta)==set(names)
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in delta:p.copy_(delta[n])
    results=[];reused=[]
    for i,row in enumerate(rows):
        pid=row['product'];ref=next(r for r in old['rows'] if r['arm']=='Source' and r['product']==pid)
        path=OLD/'evaluation'/('Source_'+pid+'.npz');assert sha(path)==ref['sha256'];assert [list(w) for w in WINDOWS]==ref['windows']
        image,valid=rgb(row,ROOT/'EVAL_IMAGE_ACCESS.jsonl')
        with np.load(path) as z:
            assert np.array_equal(z['valid'],valid) and np.array_equal(z['flat_indices'],np.flatnonzero(valid))
            s=z['scores'];assert np.array_equal(z['pred'],s.argmax(0))
            assert np.allclose(z['shadow'],torch.from_numpy(s).softmax(0)[2].numpy(),rtol=1e-6,atol=1e-7)
        reused.append(dict(product=pid,file=str(path),sha256=ref['sha256'],input_sha256=row['image_member_sha256'],
            valid=int(valid.sum()),support_sha256=array_sha(valid),source_sha256=SOURCE,windows=WINDOWS,
            source_same512_gate=True,source_full_score_raw_hashes_reproduced=True,score='mean semantic s then softmax(s)[2]'))
        s=base.stitched(model,image,'Q90_evaluation_window');shadow=torch.from_numpy(s).softmax(0)[2].numpy()
        target=ROOT/'evaluation'/('LoRA_'+pid+'.npz')
        np.savez_compressed(target,scores=s,shadow=shadow,pred=s.argmax(0).astype(np.uint8),valid=valid,flat_indices=np.flatnonzero(valid).astype(np.int64))
        results.extend([dict(ref,cache_path=str(path)),dict(arm='LoRA',product=pid,sha256=sha(target),valid=int(valid.sum()),windows=WINDOWS)])
        print('evaluate',i+1,pid,flush=True)
    check_base(model,state,names)
    write(ROOT/'SOURCE_CACHE_ADMISSION.json',dict(status='passed',rows=reused,all32_same_support=True,additional_Source_calls=0))
    write(ROOT/'EVALUATION_LOCK.json',dict(status='completed',rows=results,endpoint_sha256=sha(ROOT/'ENDPOINT_LOCK.json')))


if __name__=='__main__':
    try:
        started=time.monotonic();startup()
        recovery=load(ROOT/'RECOVERY01_LOCK.json')
        for name,h in recovery['code'].items():assert sha(R/name)==h,name
        assert sha(ROOT/'ENDPOINT_LOCK.json')==recovery['endpoint_sha256']
        evaluate()
        write(ROOT/'evaluate_GPU_RECEIPT.json',dict(status='completed',PID=os.getpid(),seconds=time.monotonic()-started,
            runtime=sys.executable,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
            gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved(),recovery='01 guard order only'))
    except BaseException:
        append('FAILURES.jsonl',dict(stage='evaluate_recovery01',traceback=traceback.format_exc(),epoch=time.time()));raise

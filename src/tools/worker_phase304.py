"""One fresh fixed Q90 preparation/training/evaluation; never reads target labels."""
import gc
import os
import sys
import time
import traceback
import numpy as np
import torch
from phase304_common import R,ROOT,OLD,PREV,SOURCE,C,sha,array_sha,load,write,append,read,table,counts,call,score_path
from phase302_images import rgb,image_guard,WINDOWS
import execute_phase302 as base

def startup():
    assert sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
    assert torch.__version__=='2.1.0+cu121' and torch.cuda.is_available()
    import mmcv,mmseg
    assert mmcv.__version__=='2.1.0' and mmseg.__version__=='1.2.2'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True);torch.cuda.reset_peak_memory_stats()
    from mmseg.utils import register_all_modules
    register_all_modules()
    import cloud_adapter.models
    base.ROOT=ROOT;base.count=call
    for name,h in load(ROOT/'EXECUTION_LOCK.json')['code'].items():assert sha(R/name)==h,name
    def deny(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=str(args[0]).replace('\\','/').lower()
            assert not any(s in p for s in ('masks.zip','alt_masks','pixbox','classification_tags','ground_truth','cloudsen12_high_l1c/img','cloudsen12_high_l1c/ann')),'Forbidden truth/source-data access: '+p
            assert not ('phase302_20261020' in p and 'final_lora.pth' in p),'Old student endpoint forbidden'
    sys.addaudithook(deny)

def check_base(model,state,names):
    assert all(torch.equal(model.state_dict()[n].detach().cpu(),v) for n,v in state.items())
    assert all(not m.training for m in model.modules())
    assert {n for n,p in model.named_parameters() if p.requires_grad}==set(names)

def prepare():
    rows=load(ROOT/'FIT_IMAGES.json')['rows'];schedule=load(ROOT/'CROP_SCHEDULE.json')['rows']
    image_guard(rows,ROOT/'PREPARE_ACCESS.jsonl')
    index={r['product']:r for r in load(PREV/'SCORE_INDEX.json')}
    deleted={r['file']:r for r in load(PREV/'TRANSIENT_SCORE_CLEANUP.json')['deleted']}
    expected={r['product']:r for r in read(PREV/'PRODUCT_THRESHOLDS.csv') if r['definition']=='B' and float(r['threshold'])==.9}
    replay={int(r['attempt']):r for r in read(PREV/'CROP_REPLAY_8000.csv') if r['definition']=='B' and float(r['threshold'])==.9}
    assert len(index)==len(expected)==100 and len(replay)==1000
    retained=sorted(p for p,r in index.items() if r['retained']);assert len(retained)==4 and len(deleted)==96
    base.seeded();model,state,names,_=base.construct(False)
    results=[];crop_results=[];validation=[]
    for i,row in enumerate(rows):
        pid=row['product'];entry=index[pid];prior=PREV/'scores'/entry['file'];image,valid=rgb(row,ROOT/'IMAGE_ACCESS.jsonl')
        if entry['retained']:
            assert sha(prior)==entry['sha256'];s=np.load(prior);assert array_sha(s)==entry['array_sha256']
            path=prior
            if pid==retained[0]:
                reconstructed=base.stitched(model,image,'retained_gate_window')
                err=float(np.abs(reconstructed-s).max());assert err<=1e-6
                def mask(z):
                    total=z.sum(0);q=np.divide(z,total[None],out=np.zeros_like(z),where=total[None]>1e-12)
                    return valid&(total>1e-12)&(q.max(0)>=.9)
                assert np.array_equal(reconstructed.argmax(0),s.argmax(0)) and np.array_equal(mask(reconstructed),mask(s))
                write(ROOT/'RECONSTRUCTION_GATE.json',dict(product=pid,maximum_error=err,argmax_identical=True,Q90_mask_identical=True,windows=9))
                del reconstructed
            mode='retained file and raw-array SHA exact'
        else:
            assert not prior.exists() and deleted[entry['file']]['sha256']==entry['sha256']
            s=base.stitched(model,image,'teacher_rebuild_window');path=ROOT/'scores'/entry['file'];np.save(path,s)
            # Strictly require the prior raw-array hash; never substitute a different mask.
            assert array_sha(s)==entry['array_sha256'],'Phase303 raw score mismatch: '+pid
            mode='rebuilt raw-array SHA exact'
        assert s.dtype==np.float32 and np.isfinite(s).all() and (s>=0).all()
        total=s.sum(0);q=np.divide(s,total[None],out=np.zeros_like(s),where=total[None]>1e-12)
        maximum=q.max(0);pred=s.argmax(0).astype(np.uint8)
        pseudo=np.where(valid&(total>1e-12)&(maximum>=.9),pred,255).astype(np.uint8)
        cc=counts(pseudo);ex=expected[pid]
        assert int(valid.sum())==int(ex['valid']) and cc==[int(ex[c]) for c in C] and sum(cc)==int(ex['accepted'])
        target=ROOT/'teacher'/(pid+'.npz');np.savez_compressed(target,pred=pred,pseudo=pseudo,maximum=maximum,valid=valid)
        results.append(dict(product=pid,valid=int(valid.sum()),accepted=sum(cc),counts=cc,sha256=sha(target),
            scores_path=str(path),scores_sha256=sha(path),scores_array_sha256=array_sha(s),matching=mode,
            predicted_counts=[int(((pred==c)&valid).sum()) for c in range(3)]))
        validation.append(dict(product=pid,method=mode,expected=ex['accepted'],observed=sum(cc),classes=cc))
        for step in (r for r in schedule if r['fit_index']==i):
            t,l=step['top'],step['left'];v=pseudo[t:t+512,l:l+512];cs=counts(v);a=step['attempt'];e=replay[a]
            assert cs==[int(e[c]) for c in C] and sum(cs)==int(e['accepted'])
            crop_results.append(dict(**step,product=pid,Surface=cs[0],Cloud=cs[1],Shadow=cs[2],accepted=sum(cs),would_update=bool(sum(cs))))
        print('prepare',i+1,pid,cc,flush=True)
    check_base(model,state,names)
    assert sum(r['valid'] for r in results)==104448397
    assert np.sum([r['counts'] for r in results],axis=0).tolist()==[40622307,33554491,1292360]
    assert sum(r['accepted']>0 for r in results)==97
    assert len(crop_results)==1000 and sum(r['would_update'] for r in crop_results)==955
    assert [sum(r[c]>0 for r in crop_results) for c in C]==[524,644,458]
    table(ROOT/'PRODUCT_REPRODUCTION.csv',validation);table(ROOT/'CROP_REPRODUCTION.csv',sorted(crop_results,key=lambda r:r['attempt']))
    write(ROOT/'PSEUDO_LOCK.json',dict(status='locked',definition='Q90-PL=B@0.90',threshold=.9,
        scoring='mean nine-window s; q=s/sum(s) with sum>1e-12; fixed argmax(s)',rows=results,
        crop_table_sha256=sha(ROOT/'CROP_REPRODUCTION.csv'),schedule_sha256=sha(ROOT/'CROP_SCHEDULE.json'),
        valid=104448397,accepted=75469158,classes=[40622307,33554491,1292360],nonempty_products=97,
        expected_updates=955,class_present_steps=[524,644,458],teacher_frozen=True,truth_reads=False))

def train():
    from torch.nn import functional as F
    rows=load(ROOT/'FIT_IMAGES.json')['rows'];schedule=load(ROOT/'CROP_SCHEDULE.json')['rows']
    lock=load(ROOT/'PSEUDO_LOCK.json');assert lock['status']=='locked'
    assert sha(ROOT/'CROP_REPRODUCTION.csv')==lock['crop_table_sha256']
    coverage={r['product']:r for r in lock['rows']};replay={int(r['attempt']):r for r in read(ROOT/'CROP_REPRODUCTION.csv')}
    image_guard(rows,ROOT/'TRAIN_ACCESS.jsonl')
    probes=sorted(rows,key=lambda r:r['product'])[:4]
    image,_=rgb(probes[0],ROOT/'IMAGE_ACCESS.jsonl');crop=image[:,255:767,255:767]
    base.seeded();source,state,_,_=base.construct(False)
    with torch.no_grad():
        expected=base.forward(source,crop,'gate_source_original',original=True,record=True).cpu()
        custom=base.forward(source,crop,'gate_source_custom').cpu();error=float((expected-custom).abs().max());assert error<=1e-6
        for row in probes:
            if row==probes[0]:s=expected[0].numpy()
            else:
                a,_=rgb(row,ROOT/'IMAGE_ACCESS.jsonl');s=base.forward(source,a[:,255:767,255:767],'gate_source_probe',original=True)[0].cpu().numpy()
            # Existing same512 scores also bind cache context to this Source and input.
            with np.load(OLD/'acceptance'/(row['product']+'.npz')) as z:assert float(np.abs(s-z['scores']).max())<=1e-6
            np.savez_compressed(ROOT/'acceptance'/(row['product']+'.npz'),scores=s)
    del source;gc.collect();torch.cuda.empty_cache()
    base.seeded();model,state,names,cfg=base.construct(True)
    frozen=base.frozen_digest(model,names);params=[p for n,p in model.named_parameters() if n in names]
    assert sum(p.numel() for p in params)==294912
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in names}
    prior=torch.load(OLD/'initial_lora.pth',map_location='cpu');assert set(initial)==set(prior)
    assert sha(OLD/'initial_lora.pth')==load(OLD/'ENDPOINT_LOCK.json')['initial_sha256']
    assert all(torch.equal(initial[n],prior[n]) for n in initial),'Fresh initialization differs from Phase302'
    torch.save(initial,ROOT/'initial_lora.pth');errors={}
    with torch.no_grad():
        for enabled in (False,True):
            model.backbone.set_lora_enabled(enabled);observed=base.forward(model,crop,'gate_lora_'+str(enabled)).cpu()
            errors[str(enabled)]=float((observed-expected).abs().max());assert errors[str(enabled)]<=1e-6
    model.backbone.set_lora_enabled(True)
    output=base.forward(model,crop,'gate_gradient');loss=F.cross_entropy(output,output.detach().argmax(1));loss.backward()
    gradients={n:dict(connected=p.grad is not None,finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else False,
        norm=float(p.grad.norm()) if p.grad is not None else None) for n,p in model.named_parameters() if n in names}
    assert all(v['connected'] and v['finite'] for v in gradients.values()) and any(v['norm']>0 for v in gradients.values())
    assert all(p.grad is None for n,p in model.named_parameters() if n not in names)
    check_base(model,state,names);assert base.frozen_digest(model,names)==frozen
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in initial:p.copy_(initial[n])
    model.zero_grad(set_to_none=True)
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in params}
    write(ROOT/'MODEL_ACCEPTANCE.json',dict(status='passed',trainable_parameters=294912,active_blocks=list(model.backbone.active_block_indices),
        trainable_names=names,gradient_receipt=gradients,original_predict_vs_semantic_max_abs_error=error,
        zero_lora_max_abs_error=errors,initial_equals_phase302_tensorwise=True,prior_initial_sha256=sha(OLD/'initial_lora.pth'),
        base_digest=frozen,all_eval=True,optimizer_only_lora=True,restored_initial_cleared_gradient_rng_reset=True))
    cfg.dump(str(ROOT/'resolved_lora_config.py'))
    def probe(attempt):
        for row in probes:
            a,v=rgb(row,ROOT/'IMAGE_ACCESS.jsonl')
            with torch.no_grad():s=base.forward(model,a[:,255:767,255:767],'probe_'+str(attempt))[0].cpu().numpy()
            if attempt==0:
                with np.load(ROOT/'acceptance'/(row['product']+'.npz')) as z:assert float(np.abs(s-z['scores']).max())<=1e-6
            np.savez_compressed(ROOT/'probes'/(row['product']+'_'+str(attempt)+'.npz'),scores=s,pred=s.argmax(0).astype(np.uint8),valid=v[255:767,255:767])
    probe(0);base.seeded();updates=0;skipped=0
    for step in schedule:
        row=rows[step['fit_index']];pid=row['product'];path=ROOT/'teacher'/(pid+'.npz')
        assert sha(path)==coverage[pid]['sha256']
        with np.load(path) as z:p=z['pseudo']
        t,l=step['top'],step['left'];p=p[t:t+512,l:l+512].copy();cs=counts(p);accepted=sum(cs)
        assert cs==[int(replay[step['attempt']][c]) for c in C]
        grad_norm=None;loss_value=None
        if accepted:
            a,_=rgb(row,ROOT/'IMAGE_ACCESS.jsonl');a=a[:,t:t+512,l:l+512].copy()
            if step['horizontal_flip']:a=a[:,:,::-1].copy();p=p[:,::-1].copy()
            optimizer.zero_grad(set_to_none=True);s=base.forward(model,a,'train')
            loss=F.cross_entropy(s,torch.from_numpy(p.astype(np.int64))[None].cuda(),ignore_index=255)
            assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            assert all(p.grad is None for n,p in model.named_parameters() if n not in names)
            grad_norm=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in params)))
            loss_value=float(loss.detach());optimizer.step();updates+=1
        else:skipped+=1
        append('TRAIN_CURVE.jsonl',dict(attempt=step['attempt'],product=pid,accepted=accepted,Surface=cs[0],Cloud=cs[1],Shadow=cs[2],
            updated=bool(accepted),loss=loss_value,lora_grad_norm=grad_norm,lr=1e-4))
        if step['attempt'] in (100,500,1000):probe(step['attempt'])
        if step['attempt']%100==0:print('train',step['attempt'],updates,loss_value,flush=True)
    assert updates==955 and skipped==45;check_base(model,state,names);assert base.frozen_digest(model,names)==frozen
    torch.save({n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in names},ROOT/'final_lora.pth')
    write(ROOT/'ENDPOINT_LOCK.json',dict(status='sealed',attempt=1000,updates=updates,skipped=skipped,
        initial_sha256=sha(ROOT/'initial_lora.pth'),final_sha256=sha(ROOT/'final_lora.pth'),source_sha256=SOURCE,
        base_initial_final_sha256=frozen,all_base_tensors_unchanged=True,
        pseudo_lock_sha256=sha(ROOT/'PSEUDO_LOCK.json'),curve_sha256=sha(ROOT/'TRAIN_CURVE.jsonl'),
        execution_lock_sha256=sha(ROOT/'EXECUTION_LOCK.json'),sealed_epoch=time.time(),
        probes={p.name:sha(p) for p in (ROOT/'probes').glob('*.npz')},
        training_access='fit images only; no source pixels or target truth',training_rule='CE(s,pseudo) accepted mean',
        GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved()))

def evaluate():
    endpoint=load(ROOT/'ENDPOINT_LOCK.json');assert endpoint['status']=='sealed' and sha(ROOT/'final_lora.pth')==endpoint['final_sha256']
    assert sha(ROOT/'PSEUDO_LOCK.json')==endpoint['pseudo_lock_sha256']
    rows=load(ROOT/'EVAL_IMAGES.json')['rows'];image_guard(rows,ROOT/'EVAL_ACCESS.jsonl')
    assert sha(ROOT/'EVAL_IMAGES.json')==sha(OLD/'EVAL_IMAGES.json')
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
        started=time.monotonic();startup();{'prepare':prepare,'train':train,'evaluate':evaluate}[sys.argv[1]]()
        write(ROOT/(sys.argv[1]+'_GPU_RECEIPT.json'),dict(status='completed',PID=os.getpid(),seconds=time.monotonic()-started,
            runtime=sys.executable,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
            gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved()))
    except BaseException:
        append('FAILURES.jsonl',dict(stage=sys.argv[1],traceback=traceback.format_exc(),epoch=time.time()));raise

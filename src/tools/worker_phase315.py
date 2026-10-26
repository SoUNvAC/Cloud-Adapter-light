"""Two fresh initialization repeats; exact Phase304 accepted-step AST and no truth access."""
import ast
import gc
import os
import random
import sys
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from phase315_common import *
from phase302_images import rgb,image_guard,WINDOWS
from prepare_phase315 import summary
import execute_phase302 as base
import worker_phase304 as oldworker
SEED=int(sys.argv[2]);D=ROOT/str(SEED);CONTEXT='startup'
def context(s):
    global CONTEXT
    CONTEXT=s
def append(name,row):
    with (D/name).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
def call(kind):
    file=ROOT/'CALLS.jsonl';n=sum(1 for _ in file.open()) if file.exists() else 0;assert n<2800
    with file.open('a') as f:f.write(json.dumps(dict(seed=SEED,context=CONTEXT,kind=kind,epoch=time.time(),PID=os.getpid()))+'\n')
def seeded():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
def opt(params):return torch.optim.AdamW(params,lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
def restore(model,initial):
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in initial:p.copy_(initial[n])
    model.zero_grad(set_to_none=True);seeded()
def startup():
    assert sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python' and torch.__version__=='2.1.0+cu121' and torch.cuda.is_available()
    import mmcv,mmseg
    assert mmcv.__version__=='2.1.0' and mmseg.__version__=='1.2.2'
    for n,h in load(ROOT/'EXECUTION_LOCK.json')['code'].items():assert sha(R/n)==h,n
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True);torch.cuda.reset_peak_memory_stats()
    from mmseg.utils import register_all_modules
    register_all_modules();import cloud_adapter.models
    base.ROOT=D;base.count=call;base.seeded=seeded
    oldworker.ROOT=D;oldworker.base=base
    def deny(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=str(args[0]).replace('\\','/').lower();assert not any(k in p for k in ['masks.zip','alt_masks','classification_tags','ground_truth','pixbox','phase305','phase306','phase307','cloudsen12_high_l1c/img','cloudsen12_high_l1c/ann']),p
            if sys.argv[1]=='train':assert not (any(k in p for k in ['phase302_','phase304_']) and p.endswith(('initial_lora.pth','final_lora.pth'))),'Old LoRA cannot initialize new seed'
    sys.addaudithook(deny)
def cropmask(row,step):
    e=next(r for r in load(PL/'PSEUDO_LOCK.json')['rows'] if r['product']==row['product']);path=PL/'teacher'/(row['product']+'.npz');assert sha(path)==e['sha256']
    with np.load(path) as z:p=z['pseudo'][step['top']:step['top']+512,step['left']:step['left']+512].copy()
    return p
def reference():
    tree=ast.parse((R/'src/tools/worker_phase304.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train');block=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='accepted');module=ast.fix_missing_locations(ast.Module(body=block.body,type_ignores=[]))
    (D/'PHASE304_ACCEPTED_STEP.py').write_text(ast.unparse(module)+'\n');return compile(module,'original_Phase304_accepted_step','exec')
def train():
    assert SEED in [31501,31502] and not (D/'initial_lora.pth').exists()
    rows=load(ROOT/'FIT_IMAGES.json')['rows'];steps=load(ROOT/'CROP_SCHEDULE.json')['rows'];replay={int(r['attempt']):r for r in read(ROOT/'CROP_IDENTITY.csv')};image_guard(rows,D/'TRAIN_ACCESS.jsonl')
    first=next(t for t in steps if int(replay[t['attempt']]['accepted']));row=rows[first['fit_index']];t,l=first['top'],first['left'];unflipped=cropmask(row,first)
    image,_=rgb(row,D/'IMAGE_ACCESS.jsonl');a=image[:,t:t+512,l:l+512].copy();p=unflipped.copy()
    if first['horizontal_flip']:a=a[:,:,::-1].copy();p=p[:,::-1].copy()
    assert hashlib.sha256(p.tobytes()).hexdigest()==replay[first['attempt']]['mask_sha256']
    context('gates');seeded();teacher,state,_,_=base.construct(False)
    with torch.no_grad():expected=base.forward(teacher,a,'Source_original',original=True,record=True);custom=base.forward(teacher,a,'Source_custom');assert float((expected-custom).abs().max())<=1e-6
    expected=expected.detach();del teacher;gc.collect();torch.cuda.empty_cache()
    seeded();model,state,names,cfg=base.construct(True);params=[v for n,v in model.named_parameters() if n in names];assert sum(v.numel() for v in params)==294912
    initial={n:v.detach().cpu().clone() for n,v in model.named_parameters() if n in names};ident=summary(initial);hist=load(ROOT/'HISTORICAL_INITIAL_IDENTITY.json');assert ident['A_sha256']!=hist['A_sha256'] and ident['B_all_zero'] and ident['shapes']==hist['shapes']
    if SEED==31502:assert ident['A_sha256']!=load(ROOT/'31501/MODEL_ACCEPTANCE.json')['A_sha256']
    torch.save(initial,D/'initial_lora.pth');frozen=base.frozen_digest(model,names);assert frozen==load(PL/'ENDPOINT_LOCK.json')['base_initial_final_sha256']
    errors={}
    with torch.no_grad():
        for enabled in [False,True]:
            model.backbone.set_lora_enabled(enabled);z=base.forward(model,a,'zero_LoRA_'+str(enabled));errors[str(enabled)]=float((z-expected).abs().max());assert errors[str(enabled)]<=1e-6
    model.backbone.set_lora_enabled(True);code=reference();optimizer=opt(params)
    scope=dict(np=np,torch=torch,F=F,rgb=rgb,row=row,ROOT=D,step=first,t=t,l=l,p=unflipped.copy(),optimizer=optimizer,model=model,base=base,params=params,names=names,updates=0)
    exec(code,scope);refgrads={n:v.grad.detach().cpu().clone() for n,v in model.named_parameters() if n in names};refvalues={n:v.detach().cpu().clone() for n,v in model.named_parameters() if n in names};ref_loss=scope['loss_value']
    restore(model,initial);optimizer=opt(params);z=base.forward(model,a,'current_PL_gate');target=torch.from_numpy(p.astype(np.int64))[None].cuda();loss=F.cross_entropy(z,target,ignore_index=255);manual=-z.log_softmax(1).permute(0,2,3,1)[target!=255].gather(1,target[target!=255,None]).mean();assert abs(float(loss-manual))<=1e-6
    loss.backward();graderr=max(float((v.grad.detach().cpu()-refgrads[n]).abs().max()) for n,v in model.named_parameters() if n in names);optimizer.step();uperr=max(float((v.detach().cpu()-refvalues[n]).abs().max()) for n,v in model.named_parameters() if n in names);losserr=abs(float(loss)-ref_loss);assert max(graderr,uperr,losserr)<=1e-6
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params) and all(v.grad is None for n,v in model.named_parameters() if n not in names)
    oldworker.check_base(model,state,names);restore(model,initial);optimizer=opt(params);assert not optimizer.state and all(v.grad is None for v in params)
    write(D/'MODEL_ACCEPTANCE.json',dict(seed=SEED,**ident,zero_lora_errors=errors,original_custom_max_error=float((expected-custom).abs().max()),same_seed_original304_AST_loss_error=losserr,same_seed_gradient_max_error=graderr,same_seed_update_max_error=uperr,only_lora_gradients=True,frozen_head_allows_backprop=True,optimizer_gradient_initial_RNG_restored=True,old_initial_not_loaded_into_new_model=True,base_digest=frozen,seed_options=dict(python=SEED,numpy=SEED,torch=SEED,cuda=SEED,deterministic=True,warn_only=True,tf32=False,cudnn_benchmark=False,dropout=0,schedule_seed=30001),initial_SHA=sha(D/'initial_lora.pth')))
    cfg.dump(str(D/'resolved_config.py'));probes=sorted(rows,key=lambda r:r['product'])[:4]
    del refgrads,refvalues,z,expected,custom,scope;gc.collect();torch.cuda.empty_cache()
    def probe(attempt):
        context('probe_'+str(attempt))
        for row in probes:
            image,v=rgb(row,D/'IMAGE_ACCESS.jsonl')
            with torch.no_grad():s=base.forward(model,image[:,255:767,255:767],'probe')[0].cpu().numpy()
            if attempt==0:
                with np.load(PL/'acceptance'/(row['product']+'.npz')) as old:assert float(np.abs(old['scores']-s).max())<=1e-6
            np.savez_compressed(D/'probes'/(row['product']+'_'+str(attempt)+'.npz'),scores=s,pred=s.argmax(0).astype(np.uint8),valid=v[255:767,255:767])
    probe(0);seeded();updates=skipped=0
    for step in steps:
        row=rows[step['fit_index']];p=cropmask(row,step);t,l=step['top'],step['left'];pm=p[:,::-1].copy() if step['horizontal_flip'] else p;ex=replay[step['attempt']];assert hashlib.sha256(pm.tobytes()).hexdigest()==ex['mask_sha256'];cs=[int((pm==c).sum()) for c in range(3)];assert cs==[int(ex[c]) for c in ['Surface','Cloud','Shadow']]
        loss_value=grad_norm=None;context('train')
        if sum(cs):
            scope=dict(np=np,torch=torch,F=F,rgb=rgb,row=row,ROOT=D,step=step,t=t,l=l,p=p,optimizer=optimizer,model=model,base=base,params=params,names=names,updates=updates);exec(code,scope);updates=scope['updates'];loss_value=scope['loss_value'];grad_norm=scope['grad_norm']
        else:skipped+=1
        append('TRAIN_CURVE.jsonl',dict(attempt=step['attempt'],product=row['product'],mask_sha256=ex['mask_sha256'],Surface=cs[0],Cloud=cs[1],Shadow=cs[2],accepted=sum(cs),updated=bool(sum(cs)),loss=loss_value,lora_grad_norm=grad_norm,lr=1e-4))
        if step['attempt'] in [100,500,1000]:probe(step['attempt'])
        if step['attempt']%100==0:print(SEED,step['attempt'],updates,loss_value,flush=True)
    assert updates==955 and skipped==45;oldworker.check_base(model,state,names);assert base.frozen_digest(model,names)==frozen
    torch.save({n:v.detach().cpu().clone() for n,v in model.named_parameters() if n in names},D/'final_lora.pth')
    write(D/'ENDPOINT_LOCK.json',dict(status='sealed',seed=SEED,updates=updates,skipped=skipped,attempts=1000,initial_sha256=sha(D/'initial_lora.pth'),final_sha256=sha(D/'final_lora.pth'),A_initial_sha256=ident['A_sha256'],base_digest=frozen,all_base_tensors_unchanged=True,source_sha256=SOURCE,curve_sha256=sha(D/'TRAIN_CURVE.jsonl'),crop_identity_sha256=sha(ROOT/'CROP_IDENTITY.csv'),sealed_epoch=time.time(),probes={p.name:sha(p) for p in (D/'probes').glob('*.npz')}))
def replay():
    assert SEED==30201;context('historical_replay');rows=load(ROOT/'EVAL_IMAGES.json')['rows'];row=sorted(rows,key=lambda r:r['product'])[0];image_guard(rows,D/'IMAGE_GUARD.jsonl');seeded();model,state,names,_=base.construct(True)
    delta=torch.load(PL/'final_lora.pth',map_location='cpu');assert sha(PL/'final_lora.pth')==load(PL/'ENDPOINT_LOCK.json')['final_sha256']
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in delta:p.copy_(delta[n])
    image,_=rgb(row,D/'IMAGE_ACCESS.jsonl');scores=base.stitched(model,image,'old30201_replay_window')
    with np.load(PL/'evaluation'/('LoRA_'+row['product']+'.npz')) as z:err=float(np.abs(scores-z['scores']).max());assert err<=1e-6 and np.array_equal(scores.argmax(0),z['pred'])
    oldworker.check_base(model,state,names);write(ROOT/'HISTORICAL_REPLAY.json',dict(product=row['product'],seed=30201,windows=9,max_abs_error=err,pred_identical=True,old_weights_only_for_identity_not_new_initialization=True))
def evaluate():
    assert all(load(ROOT/str(s)/'ENDPOINT_LOCK.json')['status']=='sealed' for s in [31501,31502]);context('evaluate');rows=load(ROOT/'EVAL_IMAGES.json')['rows'];image_guard(rows,D/'EVAL_ACCESS.jsonl');endpoint=load(D/'ENDPOINT_LOCK.json');assert sha(D/'final_lora.pth')==endpoint['final_sha256']
    seeded();model,state,names,_=base.construct(True);delta=torch.load(D/'final_lora.pth',map_location='cpu')
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in delta:p.copy_(delta[n])
    results=[]
    for i,row in enumerate(rows):
        image,valid=rgb(row,D/'EVAL_IMAGE_ACCESS.jsonl');s=base.stitched(model,image,'evaluation_window');target=D/'evaluation'/('LoRA_'+row['product']+'.npz');np.savez_compressed(target,scores=s,shadow=torch.from_numpy(s).softmax(0)[2].numpy(),pred=s.argmax(0).astype(np.uint8),valid=valid,flat_indices=np.flatnonzero(valid).astype(np.int64));results.append(dict(product=row['product'],group_id=row['group_id'],file=str(target),sha256=sha(target),valid=int(valid.sum())));print('evaluate',SEED,i+1,flush=True)
    oldworker.check_base(model,state,names);assert base.frozen_digest(model,names)==endpoint['base_digest'];write(D/'EVALUATION_LOCK.json',dict(status='sealed',seed=SEED,endpoint_sha256=sha(D/'ENDPOINT_LOCK.json'),rows=results,sealed_epoch=time.time()))
if __name__=='__main__':
    try:startup();{'train':train,'replay':replay,'evaluate':evaluate}[sys.argv[1]]()
    except BaseException:
        with (ROOT/'FAILURES.jsonl').open('a') as f:f.write(json.dumps(dict(stage=sys.argv[1],seed=SEED,traceback=traceback.format_exc(),epoch=time.time()))+'\n')
        raise

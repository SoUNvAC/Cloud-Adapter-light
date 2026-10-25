"""One fresh Q90-SP run; teacher is a separate frozen Source, all training image-only."""
import ast
import gc
import os
import sys
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from phase310_common import *
from phase302_images import rgb,image_guard,WINDOWS
import execute_phase302 as base
from loss_phase310 import losses,tests

def startup():
    assert sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
    assert torch.__version__=='2.1.0+cu121' and torch.cuda.is_available()
    import mmcv,mmseg
    assert mmcv.__version__=='2.1.0' and mmseg.__version__=='1.2.2'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True);torch.cuda.reset_peak_memory_stats()
    from mmseg.utils import register_all_modules
    register_all_modules();import cloud_adapter.models
    base.ROOT=ROOT;base.count=call
    for n,h in load(ROOT/'EXECUTION_LOCK.json')['code'].items():assert sha(R/n)==h,n
    def deny(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=str(args[0]).replace('\\','/').lower()
            assert not any(s in p for s in ('masks.zip','alt_masks','classification_tags','pixbox','phase306','phase307','cloudsen12_high_l1c/img','cloudsen12_high_l1c/ann')),'Forbidden training/evaluation truth access: '+p
            if sys.argv[1]=='train':assert not ('final_lora.pth' in p and ('phase304_' in p or 'phase302_' in p))
    sys.addaudithook(deny)

def check(model,state,names):
    assert all(torch.equal(model.state_dict()[n].detach().cpu(),v) for n,v in state.items())
    assert all(not m.training for m in model.modules())
    assert {n for n,p in model.named_parameters() if p.requires_grad}==set(names)
def parameters(model,names):return [p for n,p in model.named_parameters() if n in names]
def optimizer(params):return torch.optim.AdamW(params,lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
def restore(model,initial):
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in initial:p.copy_(initial[n])
    model.zero_grad(set_to_none=True);base.seeded()
def arrays(row,step,entry):
    path=PL/'teacher'/(row['product']+'.npz');assert sha(path)==entry['teacher_sha256']
    mp=ROOT/entry['mask_file'];assert sha(mp)==entry['mask_sha256']
    with np.load(path) as z:pseudo=z['pseudo']
    with np.load(mp) as z:P=z['P'];A=z['A']
    t,l=step['top'],step['left'];p=pseudo[t:t+512,l:l+512].copy();protect=P[t:t+512,l:l+512].copy();accept=A[t:t+512,l:l+512].copy()
    if step['horizontal_flip']:p=p[:,::-1].copy();protect=protect[:,::-1].copy();accept=accept[:,::-1].copy()
    return p,protect,accept
def input_crop(row,step):
    a,_=rgb(row,ROOT/'IMAGE_ACCESS.jsonl');t,l=step['top'],step['left'];a=a[:,t:t+512,l:l+512].copy()
    return a[:,:,::-1].copy() if step['horizontal_flip'] else a
def norm(grads):return float(torch.sqrt(sum(g.detach().square().sum() for g in grads if g is not None)))

def train():
    rows=load(ROOT/'FIT_IMAGES.json')['rows'];steps=load(ROOT/'CROP_SCHEDULE.json')['rows'];lock=load(ROOT/'MASK_LOCK.json')
    assert sha(ROOT/'CROP_SUPPORT.csv')==lock['crops_sha256'];coverage={r['product']:r for r in lock['rows']};croplock={int(r['attempt']):r for r in read(ROOT/'CROP_SUPPORT.csv')}
    image_guard(rows,ROOT/'TRAIN_ACCESS.jsonl');set_context('gates')
    numeric=tests();base.seeded();teacher,teacher_state,teacher_names,_=base.construct(False);assert not teacher_names
    teacher_digest=base.frozen_digest(teacher,[])
    base.seeded();model,state,names,cfg=base.construct(True);params=parameters(model,names);assert sum(p.numel() for p in params)==294912
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in names}
    oldinitial=torch.load(PL/'initial_lora.pth',map_location='cpu');assert set(initial)==set(oldinitial) and all(torch.equal(v,oldinitial[n]) for n,v in initial.items())
    assert sha(PL/'initial_lora.pth')==load(PL/'ENDPOINT_LOCK.json')['initial_sha256']
    torch.save(initial,ROOT/'initial_lora.pth');frozen=base.frozen_digest(model,names)
    assert frozen==load(PL/'ENDPOINT_LOCK.json')['base_initial_final_sha256']
    first=next(s for s in steps if int(croplock[s['attempt']]['A'])>0);row=rows[first['fit_index']]
    a=input_crop(row,first);p,P,A=arrays(row,first,coverage[row['product']]);target=torch.from_numpy(p.astype(np.int64))[None].cuda();protect=torch.from_numpy(P)[None].cuda()
    with torch.no_grad():
        original=base.forward(teacher,a,'gate_Source_original',original=True,record=True)
        t=base.forward(teacher,a,'gate_Source_same512');z=base.forward(model,a,'gate_zero_LoRA')
        errors=dict(source_original_custom=float((original-t).abs().max()),zero_lora_source=float((z-t).abs().max()))
        assert max(errors.values())<=1e-6
        _,initial_sp=losses(z,t,target,protect);assert abs(float(initial_sp))<=1e-6
        np.savez_compressed(ROOT/'acceptance/gate512.npz',Source=t[0].cpu().numpy(),zero_LoRA=z[0].cpu().numpy(),P=P,A=A,pseudo=p)
        protection_gate_attempt=first['attempt']
        if not P.any():
            gate=next(s for s in steps if int(croplock[s['attempt']]['A'])>0 and int(croplock[s['attempt']]['P'])>0)
            gate_row=rows[gate['fit_index']];gate_image=input_crop(gate_row,gate);gp,gP,gA=arrays(gate_row,gate,coverage[gate_row['product']])
            gt=base.forward(teacher,gate_image,'gate_first_nonempty_P_teacher');gs=base.forward(model,gate_image,'gate_first_nonempty_P_student')
            _,initial_sp=losses(gs,gt,torch.from_numpy(gp.astype(np.int64))[None].cuda(),torch.from_numpy(gP)[None].cuda())
            assert abs(float(initial_sp))<=1e-6 and float((gt-gs).abs().max())<=1e-6
            protection_gate_attempt=gate['attempt']
            np.savez_compressed(ROOT/'acceptance/nonempty_P_gate.npz',Source=gt[0].cpu().numpy(),zero_LoRA=gs[0].cpu().numpy(),P=gP,A=gA,pseudo=gp)
    # Execute the exact original Phase304 accepted-step AST, not a rewritten reference loss.
    tree=ast.parse((R/'src/tools/worker_phase304.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    block=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='accepted')
    reference=ast.fix_missing_locations(ast.Module(body=block.body,type_ignores=[]))
    (ROOT/'acceptance/phase304_reference_step.py').write_text(ast.unparse(reference)+'\n')
    t0,l0=first['top'],first['left']
    with np.load(PL/'teacher'/(row['product']+'.npz')) as cache:unflipped=cache['pseudo'][t0:t0+512,l0:l0+512].copy()
    restore(model,initial);opt=optimizer(params)
    scope=dict(np=np,torch=torch,F=F,rgb=rgb,row=row,ROOT=ROOT,step=first,p=unflipped,optimizer=opt,model=model,base=base,params=params,names=names,updates=0)
    exec(compile(reference,'original_phase304_accepted_step','exec'),scope)
    reference_grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if n in names}
    reference_update={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in names}
    restore(model,initial);opt=optimizer(params)
    with torch.no_grad():tt=base.forward(teacher,a,'gate_lambda0_teacher')
    ss=base.forward(model,a,'gate_lambda0_student');pl,sp=losses(ss,tt,target,protect);loss=pl+0.*sp;loss.backward()
    graderror=max(float((p.grad.detach().cpu()-reference_grads[n]).abs().max()) for n,p in model.named_parameters() if n in names)
    opt.step();updateerror=max(float((p.detach().cpu()-reference_update[n]).abs().max()) for n,p in model.named_parameters() if n in names)
    losserror=abs(float(pl)-scope['loss_value']);assert max(losserror,graderror,updateerror)<=1e-6
    assert all(p.grad is None for p in teacher.parameters()) and all(p.grad is None for n,p in model.named_parameters() if n not in names)
    restore(model,initial);opt=optimizer(params);assert not opt.state
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in params}
    write(ROOT/'MODEL_ACCEPTANCE.json',dict(status='passed',trainable_parameters=294912,active_blocks=list(model.backbone.active_block_indices),initial_equals_Phase304_tensorwise=True,base_initial_digest=frozen,teacher_is_separate_Source=True,teacher_has_no_lora=True,teacher_frozen_digest=teacher_digest,all_eval=True,optimizer_only_lora=True,numerical_tests=numeric,zero_lora_errors=errors,initial_L_SP=float(initial_sp),lambda0_loss_error=losserror,lambda0_gradient_max_error=graderror,lambda0_update_max_error=updateerror,lambda0_reference='exact AST of original worker_phase304.py accepted-step block',gate_attempt=first['attempt'],protection_gate_attempt=protection_gate_attempt,initial_optimizer_gradient_RNG_restored=True))
    cfg.dump(str(ROOT/'resolved_lora_config.py'));probes=sorted(rows,key=lambda r:r['product'])[:4]
    del original,t,z,ss,tt,reference_grads,reference_update;gc.collect();torch.cuda.empty_cache()
    def probe(attempt):
        set_context('probe_'+str(attempt))
        for row in probes:
            image,v=rgb(row,ROOT/'IMAGE_ACCESS.jsonl')
            with torch.no_grad():s=base.forward(model,image[:,255:767,255:767],'probe')[0].cpu().numpy()
            if attempt==0:
                with np.load(PL/'acceptance'/(row['product']+'.npz')) as old:assert float(np.abs(old['scores']-s).max())<=1e-6
            np.savez_compressed(ROOT/'probes'/(row['product']+'_'+str(attempt)+'.npz'),scores=s,pred=s.argmax(0).astype(np.uint8),valid=v[255:767,255:767])
    probe(0);base.seeded();updates=skipped=0
    for step in steps:
        set_context('formal_train');row=rows[step['fit_index']];pid=row['product'];p,P,A=arrays(row,step,coverage[pid]);cs=counts(p);ex=croplock[step['attempt']]
        assert array_sha(P)==ex['P_sha256'] and array_sha(A)==ex['A_sha256'] and array_sha(p)==ex['pseudo_sha256'] and not (P&A).any()
        accepted=sum(cs);values=dict(PL=None,SP=None,weighted_SP=None,total=None,lora_grad_norm=None,PL_grad_norm=None,SP_grad_norm=None)
        if accepted:
            image=input_crop(row,step);target=torch.from_numpy(p.astype(np.int64))[None].cuda();protect=torch.from_numpy(P)[None].cuda();opt.zero_grad(set_to_none=True)
            with torch.no_grad():st=base.forward(teacher,image,'teacher_same512')
            ss=base.forward(model,image,'student');pl,sp=losses(ss,st,target,protect);total=pl+1.*sp
            assert torch.isfinite(total) and torch.isfinite(pl) and torch.isfinite(sp)
            if step['attempt'] in lock['gradient_audit_attempts']:
                values['PL_grad_norm']=norm(torch.autograd.grad(pl,params,retain_graph=True,allow_unused=True))
                values['SP_grad_norm']=norm(torch.autograd.grad(sp,params,retain_graph=True,allow_unused=True))
            total.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            assert all(p.grad is None for p in teacher.parameters()) and all(p.grad is None for n,p in model.named_parameters() if n not in names)
            values.update(PL=float(pl.detach()),SP=float(sp.detach()),weighted_SP=float(sp.detach()),total=float(total.detach()),lora_grad_norm=norm([p.grad for p in params]))
            opt.step();updates+=1
        else:skipped+=1
        append('TRAIN_CURVE.jsonl',dict(attempt=step['attempt'],product=pid,A=accepted,P=int(P.sum()),Surface=cs[0],Cloud=cs[1],Shadow=cs[2],A_sha256=array_sha(A),P_sha256=array_sha(P),updated=bool(accepted),teacher_calls=int(bool(accepted)),student_calls=int(bool(accepted)),lr=1e-4,lambda_SP=1.,**values))
        if step['attempt'] in (100,500,1000):probe(step['attempt'])
        if step['attempt']%100==0:print('train',step['attempt'],updates,values,flush=True)
    assert updates==955 and skipped==45;check(model,state,names);check(teacher,teacher_state,[])
    assert base.frozen_digest(model,names)==frozen and base.frozen_digest(teacher,[])==teacher_digest
    torch.save({n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in names},ROOT/'final_lora.pth')
    write(ROOT/'ENDPOINT_LOCK.json',dict(status='sealed',attempt=1000,updates=updates,skipped=skipped,initial_sha256=sha(ROOT/'initial_lora.pth'),final_sha256=sha(ROOT/'final_lora.pth'),source_sha256=SOURCE,base_initial_final_sha256=frozen,teacher_frozen_sha256=teacher_digest,all_base_tensors_unchanged=True,teacher_all_tensors_unchanged=True,mask_lock_sha256=sha(ROOT/'MASK_LOCK.json'),curve_sha256=sha(ROOT/'TRAIN_CURVE.jsonl'),execution_lock_sha256=sha(ROOT/'EXECUTION_LOCK.json'),sealed_epoch=time.time(),probes={p.name:sha(p) for p in (ROOT/'probes').glob('*.npz')},training_rule='CE(s,pseudo) mean_A + forward_KL(softmax(Source_same512_s)||softmax(student_s)) mean_P; lambda1 T1',truth_reads=False))

def evaluate():
    end=load(ROOT/'ENDPOINT_LOCK.json');assert end['status']=='sealed' and sha(ROOT/'final_lora.pth')==end['final_sha256']
    assert sha(ROOT/'EVAL_IMAGES.json')==sha(PL/'EVAL_IMAGES.json')
    rows=load(ROOT/'EVAL_IMAGES.json')['rows'];image_guard(rows,ROOT/'EVAL_ACCESS.jsonl');set_context('evaluation')
    base.seeded();model,state,names,_=base.construct(True);assert base.frozen_digest(model,names)==end['base_initial_final_sha256']
    prior=load(PL/'ENDPOINT_LOCK.json');assert sha(PL/'final_lora.pth')==prior['final_sha256']
    olddelta=torch.load(PL/'final_lora.pth',map_location='cpu');restore(model,olddelta)
    first=sorted(rows,key=lambda x:x['product'])[0];image,valid=rgb(first,ROOT/'EVAL_IMAGE_ACCESS.jsonl')
    replay=base.stitched(model,image,'old_Q90_identity_replay')
    with np.load(PL/'evaluation'/('LoRA_'+first['product']+'.npz')) as z:
        err=float(np.abs(z['scores']-replay).max());assert err<=1e-6 and np.array_equal(z['pred'],replay.argmax(0)) and np.array_equal(z['valid'],valid)
    write(ROOT/'BASELINE_REPLAY_ACCEPTANCE.json',dict(product=first['product'],windows=9,maximum_error=err,pred_identical=True,original_final_sha256=prior['final_sha256'],old_model_used_only_after_new_endpoint_seal=True))
    restore(model,torch.load(ROOT/'final_lora.pth',map_location='cpu'));out=[]
    oldsha={(r['arm'],r['product']):r['sha256'] for r in load(PL/'EVALUATION_LOCK.json')['rows']}
    for i,row in enumerate(rows):
        pid=row['product'];image,valid=rgb(row,ROOT/'EVAL_IMAGE_ACCESS.jsonl')
        for arm,folder,prefix in [('Source',OLD,'Source_'),('Q90-PL',PL,'LoRA_')]:
            path=folder/'evaluation'/(prefix+pid+'.npz');expected=oldsha['Source' if arm=='Source' else 'LoRA',pid];assert sha(path)==expected
            with np.load(path) as z:assert np.array_equal(z['valid'],valid) and np.array_equal(z['flat_indices'],np.flatnonzero(valid))
            out.append(dict(arm=arm,product=pid,path=str(path),sha256=expected,valid=int(valid.sum()),windows=WINDOWS))
        s=base.stitched(model,image,'Q90_SP_evaluation_window');shadow=torch.from_numpy(s).softmax(0)[2].numpy();path=ROOT/'evaluation'/('SP_'+pid+'.npz')
        np.savez_compressed(path,scores=s,shadow=shadow,pred=s.argmax(0).astype(np.uint8),valid=valid,flat_indices=np.flatnonzero(valid))
        out.append(dict(arm='Q90-SP',product=pid,path=str(path),sha256=sha(path),valid=int(valid.sum()),windows=WINDOWS));print('evaluation',i+1,flush=True)
    check(model,state,names);write(ROOT/'EVALUATION_LOCK.json',dict(status='completed',rows=out,endpoint_sha256=sha(ROOT/'ENDPOINT_LOCK.json'),old_baselines_read_only=True))
if __name__=='__main__':
    try:
        started=time.monotonic();startup();{'train':train,'evaluate':evaluate}[sys.argv[1]]()
        write(ROOT/(sys.argv[1]+'_GPU_RECEIPT.json'),dict(status='completed',seconds=time.monotonic()-started,torch=torch.__version__,GPU=torch.cuda.get_device_name(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
    except BaseException:append('FAILURES.jsonl',dict(stage=sys.argv[1],traceback=traceback.format_exc(),epoch=time.time()));raise

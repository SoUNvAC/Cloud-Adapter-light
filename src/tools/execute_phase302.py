"""Fixed FP32 Source->LoRA pseudo-label experiment; isolated train/eval workers."""
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import threading
import time
import traceback

sys.dont_write_bytecode = True
R = Path('/home/scv/Cloud-Adapter-light')
ROOT = Path(os.environ.get('PHASE302_ROOT','/home/scv/shared/phase302_20261020'))
sys.path.insert(0,str(R/'src'))
from prepare_phase302 import sha, load, write, EXPECTED
START = time.monotonic()
STATE = dict(stage=sys.argv[-1],status='starting',calls=0)


def append(name, value):
    with (ROOT/name).open('a') as f: f.write(json.dumps(value,allow_nan=False)+'\n')


def receipt(status):
    import torch
    return dict(STATE,status=status,seconds=time.monotonic()-START,
        cpu_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        gpu_peak_allocated=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved=torch.cuda.max_memory_reserved(),PID=os.getpid())


def check():
    import torch
    prior=load(ROOT/'CONTROLLER.json')['completed_seconds']
    assert time.monotonic()-START+prior < 14400, 'GPU four-hour budget'
    count=sum(1 for _ in (ROOT/'CALLS.jsonl').open()) if (ROOT/'CALLS.jsonl').exists() else 0
    assert count <= 4000, 'Forward budget'
    assert torch.cuda.max_memory_allocated() <= 20*1024**3
    assert torch.cuda.max_memory_reserved() <= 20*1024**3
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 16*1024**3
    memory={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
    assert memory['MemAvailable'] >= 4000000000, 'Host available memory'
    import shutil
    assert shutil.disk_usage(ROOT).free >= 4000000000
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    assert size <= 6000000000, 'Persistent disk budget'


def count(kind):
    check()
    append('CALLS.jsonl',dict(stage=STATE['stage'],kind=kind,step=STATE.get('step'),epoch=time.time()))
    STATE['calls']+=1


def startup():
    for name,digest in load(ROOT/'EXECUTION_LOCK.json')['code'].items(): assert sha(R/name)==digest,name
    lock=load(ROOT/'PROTOCOL_LOCK.json')
    for name,digest in lock['input_files'].items(): assert sha(ROOT/name)==digest,name
    os.environ.update(XFORMERS_DISABLED='1',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',TMPDIR=str(ROOT/'tmp'),
        XDG_CACHE_HOME=str(ROOT/'cache'),MPLCONFIGDIR=str(ROOT/'cache'),PYTHONDONTWRITEBYTECODE='1')
    import torch, numpy as np
    assert Path(sys.executable).resolve()==Path('/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python').resolve()
    assert torch.__version__=='2.1.0+cu121' and torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.cuda.reset_peak_memory_stats()
    def monitor():
        while True:
            try: check();write(ROOT/(STATE['stage']+'_LIVE.json'),receipt('running'))
            except BaseException:
                write(ROOT/(STATE['stage']+'_GUARD_FAILURE.json'),dict(traceback=traceback.format_exc()))
                os._exit(97)
            time.sleep(2)
    threading.Thread(target=monitor,daemon=True).start()
    from mmseg.utils import register_all_modules
    register_all_modules()
    import cloud_adapter.models


def seeded():
    import torch,numpy as np
    random.seed(30201);np.random.seed(30201);torch.manual_seed(30201);torch.cuda.manual_seed_all(30201)


def construct(lora):
    import torch
    from mmengine.config import Config
    from mmseg.registry import MODELS
    cfg=Config.fromfile(str(R/'src/configs/protocol/phase64_source_parent_rgb.py'))
    cfg.model.backbone.init_cfg=None
    if lora:
        cfg.model.type='FrozenHeadEncoderDecoder'
        cfg.model.backbone.update(type='Phase64LoRACloudAdapterDinoVisionTransformer',
            lora_rank=4,lora_alpha=4,lora_dropout=0,lora_targets=('qkv','proj','fc1','fc2'))
    # Capture any unexpected model/backbone/decode forwards inside constructors.
    def constructed_forward(module,args):
        if module.__class__.__name__ in ('Mask2FormerHead','CloudAdapterDinoVisionTransformer',
                'Phase64LoRACloudAdapterDinoVisionTransformer'):
            count('constructor_'+module.__class__.__name__)
    handle=torch.nn.modules.module.register_module_forward_pre_hook(constructed_forward)
    try: model=MODELS.build(cfg.model)
    finally: handle.remove()
    source=R/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'
    assert sha(source)==EXPECTED
    state=torch.load(source,map_location='cpu')['state_dict']
    missing,unexpected=model.load_state_dict(state,strict=not lora)
    extra=set(model.state_dict())-set(state)
    assert not unexpected and set(missing)==extra
    if lora:
        assert len(extra)>0 and all(n.endswith(('.lora_a','.lora_b')) for n in extra)
        assert all(torch.count_nonzero(v)==0 for n,v in model.state_dict().items() if n.endswith('.lora_b'))
    assert all(torch.equal(model.state_dict()[n],v) for n,v in state.items())
    model.eval().cuda()
    for p in model.parameters(): p.requires_grad_(False)
    active=set(model.backbone.active_block_indices)
    names=[]
    if lora:
        for n,p in model.named_parameters():
            if n.endswith(('.lora_a','.lora_b')) and int(n.split('blocks.')[1].split('.')[0]) in active:
                p.requires_grad_(True);names.append(n)
    assert all(not m.training for m in model.modules())
    return model,state,names,cfg


def frozen_digest(model,names):
    h=hashlib.sha256()
    for n,v in model.state_dict().items():
        if n in names: continue
        h.update(n.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def preprocess(model,array,record=False):
    import torch
    from mmseg.structures import SegDataSample
    tensor=torch.from_numpy(array.copy()).cuda()
    sample=SegDataSample(metainfo=dict(img_shape=(512,512),ori_shape=(512,512),pad_shape=(512,512),
        padding_size=[0,0,0,0],img_padding_size=[0,0,0,0]))
    batch=model.data_preprocessor(dict(inputs=[tensor],data_samples=[sample]),training=False)
    assert batch['inputs'].shape==(1,3,512,512)
    if record:
        expected=(tensor[[2,1,0]]-tensor.new_tensor([123.675,116.28,103.53])[:,None,None])/tensor.new_tensor([58.395,57.12,57.375])[:,None,None]
        assert torch.equal(expected,batch['inputs'][0])
        write(ROOT/'PREPROCESS_ACCEPTANCE.json',dict(status='passed',shape=list(batch['inputs'].shape),
            input_bgr_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
            output_rgb_sha256=hashlib.sha256(batch['inputs'].detach().cpu().numpy().tobytes()).hexdigest(),
            exact_source_mean_std=True,bgr_to_rgb_once=True,no_clipping=True,padding=[0,0,0,0]))
    return batch


def forward(model,array,kind,original=False,record=False):
    import torch
    from torch.nn import functional as F
    batch=preprocess(model,array,record)
    count(kind)
    if original:
        result=model.decode_head.predict(model.extract_feat(batch['inputs']),
            [s.metainfo for s in batch['data_samples']],model.test_cfg)
    else:
        # Never use FrozenBackbone.extract_feat or test_step/no_grad for student gradients.
        features=model.backbone(batch['inputs'])
        if model.with_neck: features=model.neck(features)
        cls,masks=model.decode_head(features,batch['data_samples'])
        masks=F.interpolate(masks[-1],size=(512,512),mode='bilinear',align_corners=False)
        result=torch.einsum('bqc,bqhw->bchw',cls[-1].softmax(-1)[...,:-1],masks.sigmoid())
    assert result.shape==(1,3,512,512) and torch.isfinite(result).all()
    return result


def stitched(model,array,kind):
    import numpy as np,torch
    from phase302_images import WINDOWS
    scores=np.zeros((3,1022,1022),np.float32);weights=np.zeros((1022,1022),np.uint8)
    with torch.no_grad():
        for top,left in WINDOWS:
            s=forward(model,array[:,top:top+512,left:left+512],kind)[0].cpu().numpy()
            scores[:,top:top+512,left:left+512]+=s;weights[top:top+512,left:left+512]+=1
    assert (weights>0).all()
    return scores/weights[None]


def train():
    import numpy as np,torch
    from torch.nn import functional as F
    from phase302_images import rgb,image_guard,WINDOWS
    rows=load(ROOT/'FIT_IMAGES.json')['rows'];schedule=load(ROOT/'CROP_SCHEDULE.json')['rows']
    image_guard(rows,ROOT/'TRAIN_ACCESS.jsonl')
    probes=sorted(rows,key=lambda r:r['product'])[:4]
    array,valid=rgb(probes[0],ROOT/'IMAGE_ACCESS.jsonl');crop=array[:,255:767,255:767]
    seeded();source,base,_,_=construct(False)
    acceptance=ROOT/'acceptance';acceptance.mkdir()
    gate_start=time.monotonic()
    with torch.no_grad():
        expected=forward(source,crop,'acceptance_source_original',original=True,record=True).cpu()
        custom=forward(source,crop,'acceptance_source_semantic').cpu()
        np.savez_compressed(acceptance/(probes[0]['product']+'.npz'),scores=expected[0].numpy())
        for row in probes[1:]:
            image,_=rgb(row,ROOT/'IMAGE_ACCESS.jsonl')
            score=forward(source,image[:,255:767,255:767],'acceptance_source_probe',original=True)[0].cpu().numpy()
            np.savez_compressed(acceptance/(row['product']+'.npz'),scores=score)
    inference_upper=(time.monotonic()-gate_start)/5*4000*2
    assert inference_upper<14400, 'Measured Source gate projects over GPU time budget: '+str(inference_upper)
    error=float((expected-custom).abs().max());assert error<=1e-6,error
    del source;gc.collect();torch.cuda.empty_cache()
    seeded();model,base,names,cfg=construct(True)
    frozen=frozen_digest(model,names)
    from cloud_adapter.models.backbones.phase64_lora_layers import LoRALinear
    layers=[dict(name=n,rank=m.rank,alpha=m.alpha,scaling=m.scaling,shape=list(m.weight.shape))
        for n,m in model.named_modules() if isinstance(m,LoRALinear)]
    errors={}
    with torch.no_grad():
        for enabled in (False,True):
            model.backbone.set_lora_enabled(enabled)
            observed=forward(model,crop,'acceptance_lora_'+str(enabled)).cpu()
            errors[str(enabled)]=float((observed-expected).abs().max());assert errors[str(enabled)]<=1e-6
    # Exercise a real differentiable path without optimizer update; no ground truth.
    model.backbone.set_lora_enabled(True)
    output=forward(model,crop,'acceptance_gradient')
    target=output.detach().argmax(1)
    loss=F.cross_entropy(output,target);loss.backward()
    gradients={n:dict(connected=p.grad is not None,finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else False,
        norm=float(p.grad.norm()) if p.grad is not None else None) for n,p in model.named_parameters() if n in names}
    assert all(v['connected'] and v['finite'] for v in gradients.values())
    assert any(v['norm']>0 for v in gradients.values())
    assert frozen_digest(model,names)==frozen
    model.zero_grad(set_to_none=True)
    params=[p for n,p in model.named_parameters() if n in names]
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in params}
    write(ROOT/'MODEL_ACCEPTANCE.json',dict(status='passed',zero_lora_max_abs_error=errors,
        original_predict_vs_semantic_max_abs_error=error,gradient_receipt=gradients,
        base_digest=frozen,active_blocks=list(model.backbone.active_block_indices),lora_layers=layers,
        trainable_names=names,trainable_parameters=sum(p.numel() for p in params),
        base_parameters=sum(p.numel() for p in model.parameters())-sum(p.numel() for p in params),
        all_eval=True,optimizer_only_lora=True,source_sha256=EXPECTED,
        conservative_inference_projection_seconds=inference_upper,
        source_probe_sha256={p.name:sha(p) for p in acceptance.glob('*.npz')}))
    cfg.dump(str(ROOT/'resolved_lora_config.py'))
    def save_delta(path):
        torch.save({n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in names},path)
    save_delta(ROOT/'initial_lora.pth')
    teacher=ROOT/'teacher';teacher.mkdir();probe_dir=ROOT/'probes';probe_dir.mkdir()
    coverage=[]
    for index,row in enumerate(rows):
        STATE.update(step=index,kind='teacher')
        image,valid=rgb(row,ROOT/'IMAGE_ACCESS.jsonl')
        scores=stitched(model,image,'teacher_window')
        p=torch.from_numpy(scores).softmax(0).numpy();maximum=p.max(0);pred=scores.argmax(0).astype(np.uint8)
        pseudo=np.where(valid & (maximum>=.90),pred,255).astype(np.uint8)
        values=dict(pseudo=pseudo,maximum=maximum,valid=valid)
        if row['product'] in {r['product'] for r in probes}: values.update(scores=scores,pred=pred)
        path=teacher/(row['product']+'.npz');np.savez_compressed(path,**values)
        coverage.append(dict(product=row['product'],valid=int(valid.sum()),
            accepted=int((pseudo!=255).sum()),counts=[int((pseudo==c).sum()) for c in range(3)],sha256=sha(path)))
        if index%10==0: print('teacher',index+1,flush=True)
    write(ROOT/'PSEUDO_LOCK.json',dict(rows=coverage,threshold=.90,windows=WINDOWS,
        scoring='mean window sum_query softmax(class,exclude no-object)*sigmoid(mask); then softmax class'))
    def probe(attempt):
        assert all(not m.training for m in model.modules())
        assert {n for n,p in model.named_parameters() if p.requires_grad}==set(names)
        for row in probes:
            image,valid=rgb(row,ROOT/'IMAGE_ACCESS.jsonl')
            with torch.no_grad(): score=forward(model,image[:,255:767,255:767],'probe_'+str(attempt))[0].cpu().numpy()
            if attempt==0:
                with np.load(acceptance/(row['product']+'.npz')) as z:
                    assert float(np.abs(score-z['scores']).max())<=1e-6
            np.savez_compressed(probe_dir/(row['product']+'_'+str(attempt)+'.npz'),
                scores=score,pred=score.argmax(0).astype(np.uint8),valid=valid[255:767,255:767])
    probe(0);seeded();updates=0;skipped=0
    for row in schedule:
        attempt=row['attempt'];STATE.update(step=attempt,kind='training')
        check()
        item=rows[row['fit_index']];image,valid=rgb(item,ROOT/'IMAGE_ACCESS.jsonl')
        with np.load(teacher/(item['product']+'.npz')) as z: pseudo=z['pseudo']
        top,left=row['top'],row['left']
        image=image[:,top:top+512,left:left+512].copy();pseudo=pseudo[top:top+512,left:left+512].copy()
        if row['horizontal_flip']: image=image[:,:,::-1].copy();pseudo=pseudo[:,::-1].copy()
        accepted=int((pseudo!=255).sum());grad_norm=None;loss_value=None
        if accepted:
            optimizer.zero_grad(set_to_none=True)
            score=forward(model,image,'train')
            loss=F.cross_entropy(score,torch.from_numpy(pseudo.astype(np.int64))[None].cuda(),ignore_index=255)
            assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            grad_norm=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in params)))
            loss_value=float(loss.detach());optimizer.step();updates+=1
        else: skipped+=1
        append('TRAIN_CURVE.jsonl',dict(attempt=attempt,product=item['product'],accepted=accepted,
            updated=bool(accepted),loss=loss_value,lora_grad_norm=grad_norm,lr=1e-4))
        if attempt in (100,500,1000): probe(attempt)
        if attempt%100==0: print('train',attempt,'updates',updates,'loss',loss_value,flush=True)
    assert updates+skipped==1000
    assert frozen_digest(model,names)==frozen
    assert all(torch.equal(model.state_dict()[n].detach().cpu(),v) for n,v in base.items())
    save_delta(ROOT/'final_lora.pth')
    write(ROOT/'ENDPOINT_LOCK.json',dict(status='sealed',attempt=1000,updates=updates,skipped=skipped,
        initial_sha256=sha(ROOT/'initial_lora.pth'),final_sha256=sha(ROOT/'final_lora.pth'),
        source_sha256=EXPECTED,base_initial_final_sha256=frozen,all_base_tensors_unchanged=True,
        probes={p.name:sha(p) for p in probe_dir.glob('*.npz')},training_access='fit images only; no source pixels or target truth',
        receipt=receipt('completed')))


def evaluate():
    import numpy as np,torch
    from phase302_images import rgb,WINDOWS
    endpoint=load(ROOT/'ENDPOINT_LOCK.json');assert endpoint['status']=='sealed'
    assert sha(ROOT/'final_lora.pth')==endpoint['final_sha256']
    rows=load(ROOT/'EVAL_IMAGES.json')['rows'];out=ROOT/'evaluation';out.mkdir()
    results=[]
    for arm in ('Source','LoRA'):
        seeded();model,base,names,_=construct(arm=='LoRA')
        if arm=='LoRA':
            delta=torch.load(ROOT/'final_lora.pth',map_location='cpu')
            assert set(delta)==set(names)
            with torch.no_grad():
                for n,p in model.named_parameters():
                    if n in delta: p.copy_(delta[n])
        assert frozen_digest(model,names)==endpoint['base_initial_final_sha256']
        for index,row in enumerate(rows):
            STATE.update(step=index,kind=arm+'_eval')
            image,valid=rgb(row,ROOT/'EVAL_IMAGE_ACCESS.jsonl')
            scores=stitched(model,image,arm+'_evaluation_window')
            shadow=torch.from_numpy(scores).softmax(0)[2].numpy()
            path=out/(arm+'_'+row['product']+'.npz')
            np.savez_compressed(path,scores=scores,shadow=shadow,pred=scores.argmax(0).astype(np.uint8),
                valid=valid,flat_indices=np.flatnonzero(valid).astype(np.int64))
            results.append(dict(arm=arm,product=row['product'],sha256=sha(path),valid=int(valid.sum()),windows=WINDOWS))
            if index%8==0:print('evaluation',arm,index+1,flush=True)
        assert all(torch.equal(model.state_dict()[n].detach().cpu(),v) for n,v in base.items())
        del model,base;gc.collect();torch.cuda.empty_cache()
    write(ROOT/'EVALUATION_LOCK.json',dict(status='completed',rows=results,endpoint_sha256=sha(ROOT/'ENDPOINT_LOCK.json'),receipt=receipt('completed')))


def controller():
    lock=load(ROOT/'EXECUTION_LOCK.json')
    for name,digest in lock['code'].items(): assert sha(R/name)==digest,name
    assert not (ROOT/'CONTROLLER.json').exists(), 'No automatic retry'
    completed=0
    for stage in ('train','evaluate'):
        write(ROOT/'CONTROLLER.json',dict(status='running',stage=stage,completed_seconds=completed,PID=os.getpid()))
        t=time.monotonic()
        with (ROOT/(stage+'.log')).open('xb') as stream:
            result=subprocess.run([sys.executable,'-B',__file__,stage],cwd=R,stdout=stream,stderr=subprocess.STDOUT,
                timeout=max(1,14400-completed))
        completed+=time.monotonic()-t
        append('STAGE_LEDGER.jsonl',dict(stage=stage,exit_code=result.returncode,seconds=completed))
        if result.returncode:
            write(ROOT/'CONTROLLER.json',dict(status='failed',stage=stage,completed_seconds=completed))
            (ROOT/'EXIT_CODE').write_text(str(result.returncode));return
    write(ROOT/'CONTROLLER.json',dict(status='completed',completed_seconds=completed))
    (ROOT/'EXIT_CODE').write_text('0')


if __name__=='__main__':
    try:
        if sys.argv[-1]=='controller':controller()
        else:
            startup()
            {'train':train,'evaluate':evaluate}[sys.argv[-1]]()
            write(ROOT/(sys.argv[-1]+'_RECEIPT.json'),receipt('completed'))
    except BaseException:
        append('FAILURES.jsonl',dict(stage=STATE['stage'],traceback=traceback.format_exc(),epoch=time.time()))
        raise

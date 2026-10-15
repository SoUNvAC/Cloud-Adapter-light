"""Two isolated, identically scheduled decoder-only fits, no development selection."""
import csv,gc,inspect,json,os,resource,sys,threading,time,traceback
from pathlib import Path
from phase121_common import *
START=time.time();S={'status':'initializing','calls':0,'construction_calls':0,'gate_calls':0,'training_calls':0,'step':0}

def main(mode):
    setup();out=ROOT/('train_'+mode);out.mkdir(exist_ok=False);(out/'PID').write_text(str(os.getpid()))
    import numpy as np
    import torch
    from omnicloudmask import model_utils as mu
    from safetensors.torch import save_file
    from install_phase111_runtime import host_free
    torch.set_num_threads(4);torch.manual_seed(12002);np.random.seed(12002)
    assert torch.__version__=='2.11.0+cu128' and torch.cuda.is_available() and not torch.is_autocast_enabled()
    prior=load(ROOT/'CONTROLLER_STATUS.json')['calls_completed'];epoch=load(ROOT/'CONTROLLER_STATUS.json')['started_epoch']
    stop=threading.Event()
    def guard():
        S.update(elapsed_seconds=time.time()-START,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            gpu_reserved_peak_bytes=torch.cuda.max_memory_reserved())
        assert S['rss_peak_bytes']<=16*1024**3 and S['gpu_reserved_peak_bytes']<=20*1024**3
        assert prior+S['calls']<=8000 and time.time()-epoch<=14400
    def monitor():
        try:
            while not stop.wait(15):
                guard();free=host_free();assert free>=4000000000
                S['host_min_free_bytes']=min(S.get('host_min_free_bytes',free),free)
                size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file());assert size<=4000000000
                S['intermediate_peak_bytes']=max(S.get('intermediate_peak_bytes',0),size)
                write(out/'RUN_STATUS.json',S)
        except Exception:
            S.update(status='paused_resource',failure=traceback.format_exc());write(out/'RUN_STATUS.json',S)
            (out/'EXIT_CODE').write_text('1\n');os._exit(1)
    guard();assert host_free()>=4000000000;threading.Thread(target=monitor,daemon=True).start()
    for item in load(BASE/'ASSET_VERIFICATION.json')['files']:
        if 'vendor' in Path(item['path']).parts or 'weights' in Path(item['path']).parts:assert sha(item['path'])==item['sha256']
    fit=load(ROOT/'FIT_INPUT_LOCK.json')['bindings'];schedule=load(ROOT/'PAIRED_CROP_SCHEDULE.json')['rows']
    cache=[]
    for group in fit:
        arr,valid,truth=catalog_image(group,read_truth=True)
        target=np.full(valid.shape,255,np.uint8);target[valid]=truth
        cache.append((arr,target))
    assert len(cache)==153 and len(schedule)==1000
    def crop(record):
        arr,t=cache[record['fit_index']];y,x=record['top'],record['left']
        a=arr[:,y:y+512,x:x+512];b=t[y:y+512,x:x+512]
        if record['horizontal_flip']:a=a[:,:,::-1];b=b[:,::-1]
        a=mu.channel_norm(a,0);assert np.isfinite(a).all() and (b!=255).any()
        return torch.from_numpy(a[None].copy()).cuda(),torch.from_numpy(b[None].astype(np.int64)).cuda()
    depth=[0]
    def pre(m,a):
        if depth[0]==0:S['calls']+=1;S['construction_calls']+=1
        depth[0]+=1
    def post(m,a,o):depth[0]-=1
    a=torch.nn.modules.module.register_module_forward_pre_hook(pre);b=torch.nn.modules.module.register_module_forward_hook(post,always_call=True)
    index=list(csv.DictReader((BASE/'assets/vendor/omnicloudmask/model_download_links.csv').open()))
    weights=load(REPO/'src/research_results/phase111_20261014/PRE_EXECUTION_LOCK.json')['weights']
    def build(w):
        mu.load_model_from_weights.cache_clear()  # never share the mutable student with a cached teacher
        rec=next(r for r in index if r['file_name']==w['google_drive_filename'] and r['version']=='1')
        return mu.load_model_from_weights(model_name=rec['timm_model_name'],weights_path=BASE/'assets/weights'/w['hf_filename'],
            model_library='fastai',device=torch.device('cuda'),dtype=torch.float32,in_chans=3,n_out=4,compile_models=False,patch_size=1000,batch_size=1)
    try:
        student=build(next(w for w in weights if 'regnet' in w['architecture']))
        teachers=[build(w) for w in weights] if mode=='D' else []
    finally:a.remove();b.remove()
    for teacher in teachers:
        teacher.eval()
        for p in teacher.parameters():p.requires_grad_(False)
    assert not ({id(p) for p in student.parameters()} & {id(p) for t in teachers for p in t.parameters()})
    encoder,train,dec=decoder_names(student);trainset=set(train)
    for n,p in student.named_parameters():p.requires_grad_(n in trainset)
    def enc_state():
        return {n:v for n,v in student.state_dict().items() if n not in dec}
    initial_encoder=digest_state(enc_state());initial_decoder=digest_state({n:v for n,v in student.state_dict().items() if n in dec})
    teacher_hash=[digest_state(t.state_dict()) for t in teachers]
    signature={'modules':[[n,type(v).__module__+'.'+type(v).__qualname__] for n,v in student.named_modules()],
               'state':{n:list(v.shape) for n,v in student.state_dict().items()}}
    from fastai.vision.models.unet import DynamicUnet
    from fastai.vision.learner import create_unet_model
    whitelist={'encoder_boundary':'author build_fastai_model -> create_unet_model -> DynamicUnet.layers[0]=encoder',
        'DynamicUnet_init_source':inspect.getsource(DynamicUnet.__init__),
        'create_unet_model_source':inspect.getsource(create_unet_model),
        'encoder_names':list(enc_state()),'trainable_names':train,'decoder_state_names':dec,
        'parameters':[{ 'name':n,'shape':list(p.shape),'numel':p.numel(),'trainable':p.requires_grad} for n,p in student.named_parameters()],
        'buffers':[{'name':n,'shape':list(v.shape),'frozen':n not in dec} for n,v in student.named_buffers()],
        'total_parameters':sum(p.numel() for p in student.parameters()),'trainable_parameters':sum(p.numel() for p in student.parameters() if p.requires_grad),
        'graph_signature':signature,'initial_encoder_sha256':initial_encoder,'initial_decoder_sha256':initial_decoder}
    assert whitelist['total_parameters']==18039132 and whitelist['trainable_parameters']==14135988
    previous=load(REPO/'src/research_results/phase120_20261015/C_PARAMETER_WHITELIST.json')
    assert signature==previous['graph_signature'] and initial_encoder==previous['initial_encoder_sha256'] and initial_decoder==previous['initial_decoder_sha256']
    write(out/'PARAMETER_WHITELIST.json',whitelist)
    if mode=='D':
        c=load(ROOT/'train_C/PARAMETER_WHITELIST.json')
        assert initial_decoder==c['initial_decoder_sha256'] and initial_encoder==c['initial_encoder_sha256'] and signature==c['graph_signature']
    # Store only original decoder state; original frozen encoder is bound by official weight SHA.
    def save(path):save_file({n:v.detach().cpu().contiguous() for n,v in student.state_dict().items() if n in dec},str(path))
    save(out/'decoder_initial.safetensors')
    handles=[]
    def counted(m,a):S['calls']+=1;S[S['call_kind']+'_calls']+=1;guard()
    for m in [student]+teachers:handles.append(m.register_forward_pre_hook(counted))
    x,y=crop(schedule[0]);student.eval();S['call_kind']='gate'
    with torch.no_grad():
        init=student(x)
        # Reference is directly loaded original frozen RegNet state, independent object for D;
        # C initial state is verified tensorwise against the original safetensors.
        from safetensors.torch import load_file
        original=load_file(str(BASE/'assets/weights'/next(w for w in weights if 'regnet' in w['architecture'])['hf_filename']))
        assert all(torch.equal(v.detach().cpu(),original[n]) for n,v in student.state_dict().items())
        if teachers:
            # Weight metadata ordering is explicit; locate original RegNet member.
            ref=teachers[next(i for i,w in enumerate(weights) if 'regnet' in w['architecture'])]
            err=float((init-ref(x)).abs().max());assert err<=1e-5
        else:
            # A second eval forward checks exact deterministic original-R reproduction.
            err=float((init-student(x)).abs().max());assert err<=1e-5
    del original,init
    student.train();encoder.eval();outgate=student(x)
    ce,kl=objectives(outgate,y);ce.backward()
    grad=[p.grad for p in student.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grad) and any(g.abs().max()>0 for g in grad)
    assert all(p.grad is None for p in encoder.parameters())
    assert digest_state(enc_state())==initial_encoder
    write(out/'INITIAL_GRADIENT_GATE.json',{'original_R_eval_max_error':err,'original_state_tensorwise_equal':True,
        'decoder_gradients_finite_nonzero':True,'encoder_gradients_none':True,'encoder_buffers_unchanged':True,'calls':S['calls']})
    # The gradient gate can update decoder BN buffers: restore the full initial decoder before step1.
    student.load_state_dict(load_file(str(out/'decoder_initial.safetensors'),device='cuda'),strict=False)
    assert digest_state({n:v for n,v in student.state_dict().items() if n in dec})==initial_decoder
    student.zero_grad(set_to_none=True);del x,y,outgate,ce,kl,grad
    torch.manual_seed(12002)
    opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    losses=[];S.update(status='training',call_kind='training')
    student.train();encoder.eval()
    for step,record in enumerate(schedule,1):
        x,y=crop(record);opt.zero_grad(set_to_none=True)
        output=student(x);teacher_output=None
        if mode=='D':
            with torch.no_grad():teacher_output=torch.stack([t(x) for t in teachers]).mean(0)
        ce,kl=objectives(output,y,teacher_output);loss=ce+kl
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None for p in encoder.parameters())
        assert all(p.grad is None for t in teachers for p in t.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters() if p.requires_grad)
        opt.step();losses.append({'step':step,'CE':float(ce.detach()),'KL_T2_times4':float(kl.detach()),'loss':float(loss.detach())})
        S['step']=step
        if step%25==0:
            write(out/'RUN_STATUS.json',S);print(mode,step,losses[-1],S['calls'],flush=True)
        del x,y,output,teacher_output,ce,kl,loss
    assert digest_state(enc_state())==initial_encoder
    assert [digest_state(t.state_dict()) for t in teachers]==teacher_hash
    final_decoder=digest_state({n:v for n,v in student.state_dict().items() if n in dec});assert final_decoder!=initial_decoder
    final=ROOT/(mode+'_decoder_final.safetensors');save(final)
    # Only endpoint states, no intermediate checkpoints or teacher pixel cache.
    write(out/'TRAINING_RECEIPT.json',{'mode':mode,'steps':1000,'seed':12002,'losses':losses,
        'schedule_sha256':sha(ROOT/'PAIRED_CROP_SCHEDULE.json'),'final_model_sha256':sha(final),
        'initial_decoder_sha256':initial_decoder,'final_decoder_sha256':final_decoder,
        'encoder_unchanged':True,'teachers_unchanged':True,'teacher_state_sha256':teacher_hash,
        'teacher_student_disjoint_parameters':True,'decoder_updated':True,'selection':'fixed step1000, no development access',
        'training_calls':S['training_calls'],'gate_calls':S['gate_calls'],'construction_calls':S['construction_calls'],
        'total_parameters':whitelist['total_parameters'],'trainable_parameters':whitelist['trainable_parameters']})
    for h in handles:h.remove()
    guard();stop.set();S.update(status='completed_training_sealed',all_calls=S['calls']);write(out/'RUN_STATUS.json',S)
    (out/'EXIT_CODE').write_text('0\n')

if __name__=='__main__':
    mode=sys.argv[1];assert mode in ('C','D')
    try:main(mode)
    except Exception:
        target=ROOT/('train_'+mode);target.mkdir(exist_ok=True)
        S.update(status='paused_engineering',failure=traceback.format_exc());write(target/'RUN_STATUS.json',S)
        (target/'EXIT_CODE').write_text('1\n');print(S['failure'],file=sys.stderr,flush=True);os._exit(1)

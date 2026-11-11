"""Only missing fixed V1 E cube scores. No labels, fitting, downloads or cache reruns."""
import argparse,csv,gc,hashlib,io,json,os,resource,subprocess,sys,threading,time,traceback,zipfile
from pathlib import Path
R=Path('/home/scv/Cloud-Adapter-light');ROOT=Path('/home/scv/shared/phase202_20261017');BASE=Path('/home/scv/shared/phase111_ocm_runtime')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);t=p.with_suffix(p.suffix+'.writing');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args();cfg=load(ROOT/'EXECUTION_LOCK.json')
    assert sha(ROOT/'code/worker_phase202.py')==cfg['worker_sha256']
    if a.launch:
        assert not (ROOT/'LAUNCH.json').exists()
        with (ROOT/'console.log').open('xb') as f:q=subprocess.Popen([sys.executable,'-B',str(ROOT/'code/worker_phase202.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,cwd=R)
        write(ROOT/'LAUNCH.json',dict(PID=q.pid,epoch=time.time(),worker_sha256=cfg['worker_sha256']));print(json.dumps(load(ROOT/'LAUNCH.json')));return
    assert not (ROOT/'RUN_STATUS.json').exists();started=time.monotonic();state=dict(status='initializing',main_calls=0,construction_calls=0,completed=[],PID=os.getpid())
    stop=threading.Event()
    try:
        sys.path.insert(0,str(R/'src/tools'));from install_phase111_runtime import host_free
        assert host_free()>=4_000_000_000
        for key in ['HF_HOME','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR','TMPDIR']:os.environ[key]=str(BASE/'controlled_caches'/key.lower())
        os.environ.update(HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',GDAL_NUM_THREADS='1',OPJ_NUM_THREADS='1')
        sys.path.insert(0,str(BASE/'assets/vendor'))
        import numpy as np,torch
        from omnicloudmask import cloud_mask as cm,model_utils as mu
        torch.set_num_threads(4);assert torch.__version__=='2.11.0+cu128' and torch.cuda.is_available()
        assets=load(BASE/'ASSET_VERIFICATION.json');assert assets['status']=='all_files_verified'
        for v in assets['files']:
            if '/vendor/' in v['path'] or '/weights/' in v['path']:assert sha(v['path'])==v['sha256']
        def save():write(ROOT/'RUN_STATUS.json',state)
        def guard():
            state.update(calls=state['main_calls']+state['construction_calls'],seconds=time.monotonic()-started,cpu_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu_peak_bytes=torch.cuda.max_memory_reserved())
            assert state['calls']<=10000 and state['seconds']<=14400 and state['cpu_peak_bytes']<=16*1024**3 and state['gpu_peak_bytes']<=20*1024**3
        def monitor():
            try:
                while not stop.wait(15):
                    guard();assert host_free()>=4_000_000_000;assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<=6_000_000_000
            except BaseException:state.update(status='failed_resource',error=traceback.format_exc());save();(ROOT/'EXIT_CODE').write_text('1\n');os._exit(1)
        threading.Thread(target=monitor,daemon=True).start();depth=[0]
        def pre(m,args):
            if depth[0]==0:state['construction_calls']+=1;guard();save()
            depth[0]+=1
        def post(m,args,out):depth[0]-=1
        h1=torch.nn.modules.module.register_module_forward_pre_hook(pre);h2=torch.nn.modules.module.register_module_forward_hook(post,always_call=True)
        lock=load(R/'src/research_results/phase111_20261014/PRE_EXECUTION_LOCK.json');idx=list(csv.DictReader((BASE/'assets/vendor/omnicloudmask/model_download_links.csv').open()));models=[]
        try:
            for w in lock['weights']:
                v=next(x for x in idx if x['file_name']==w['google_drive_filename'] and x['version']=='1');p=BASE/'assets/weights'/w['hf_filename'];assert sha(p)==w['declared_sha256']
                models.append(mu.load_model_from_weights(model_name=v['timm_model_name'],weights_path=p,model_library='fastai',device=torch.device('cuda'),dtype=torch.float32,in_chans=3,n_out=4,compile_models=False,patch_size=1000,batch_size=1))
        finally:h1.remove();h2.remove()
        assert len(models)==2
        for m in models:
            assert not m.training
            for v in m.parameters():v.requires_grad_(False)
        def attempt(m,args):state['main_calls']+=1;guard();save()
        hooks=[m.register_forward_pre_hook(attempt) for m in models]
        out=ROOT/'scores';out.mkdir(exist_ok=False);records=[]
        with zipfile.ZipFile('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/subscenes.zip') as z:
            for rec in cfg['missing']:
                g=rec['group'];pid=g['product'];path=out/(pid+'.npy');assert not path.exists();state.update(status='running',current=pid);save()
                rawbytes=z.read('subscenes/'+pid+'.npy');assert hashlib.sha256(rawbytes).hexdigest()==g['image_member_sha256'];raw=np.load(io.BytesIO(rawbytes),allow_pickle=False)
                assert raw.shape==(1022,1022,13) and raw.dtype==np.float32
                image=raw[...,[3,2,8]].transpose(2,0,1).copy();good=np.isfinite(image)&(image!=0)&(image!=np.float32(65535/10000));image[~good]=0
                assert cm.check_patch_size(image,0,1000,300)==(300,1000)
                before=state['main_calls']
                result=cm.predict_from_array(image,patch_size=1000,patch_overlap=300,batch_size=1,inference_device='cuda',mosaic_device='cpu',inference_dtype=torch.float32,export_confidence=True,softmax_output=True,no_data_value=0,apply_no_data_mask=True,custom_models=models,pred_classes=4,compile_models=False,model_version=1,destination_model_dir=BASE/'assets/weights')
                assert result.shape==(4,1022,1022) and np.isfinite(result).all() and state['main_calls']-before<=8
                np.save(path,result[3],allow_pickle=False);records.append(dict(product=pid,group_id=g['group_id'],split=rec['split'],path=str(path),sha256=sha(path),bytes=path.stat().st_size,main_calls=state['main_calls']-before))
                state['completed'].append(pid);write(ROOT/'SCORE_INDEX.json',dict(status='in_progress',rows=records));guard();save();print('completed',len(records),pid,flush=True)
                del rawbytes,raw,image,good,result;gc.collect();torch.cuda.empty_cache()
        for h in hooks:h.remove()
        guard();state.update(status='completed',score_index_sha256=None,labels_accessed=False,training=0,fitting=0,new_downloads=0)
        write(ROOT/'SCORE_INDEX.json',dict(status='sealed',rows=records,worker_sha256=cfg['worker_sha256'],model_identity=lock['weights'],code_commit=lock['code_commit'],fixed_before_local_label_join=True));state['score_index_sha256']=sha(ROOT/'SCORE_INDEX.json');save();(ROOT/'EXIT_CODE').write_text('0\n')
    except BaseException:
        state.update(status='failed',error=traceback.format_exc(),seconds=time.monotonic()-started);write(ROOT/'RUN_STATUS.json',state);(ROOT/'EXIT_CODE').write_text('1\n');raise
    finally:stop.set()
if __name__=='__main__':main()

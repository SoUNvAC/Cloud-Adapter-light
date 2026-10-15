"""Isolated E/R/N deployment workers; original author full-scene inference."""
import argparse
import csv
import gc
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
import traceback
from phase119_common import ROOT,BASE,REPO,LOCK,OLD,load,write,sha,ram,setup,prepare,coordinates,image,catalog_image,verify_native_and_cache,original_pix_score

sys.dont_write_bytecode=True
S={'status':'initializing','construction_calls':0,'main_calls':0,'warmup_calls':0}
START=time.monotonic()


def run_worker(policy):
    cfg=setup();out=ROOT/policy;out.mkdir(exist_ok=False)
    (out/'PID').write_text(str(os.getpid()));S.update(policy=policy)
    coord=coordinates();verify_native_and_cache()
    import numpy as np
    import torch
    from omnicloudmask import cloud_mask as cm,model_utils as mu,raster_utils as ru
    from install_phase111_runtime import host_free
    torch.set_num_threads(4)
    assert torch.__version__=='2.11.0+cu128' and torch.cuda.is_available() and not torch.is_autocast_enabled()
    assert ram()['MemAvailable']>=18*1024**3 and host_free()>=4000000000
    prior=load(ROOT/'CONTROLLER_STATUS.json')['calls_completed']
    global_start=load(ROOT/'CONTROLLER_STATUS.json')['started_epoch']
    ended=threading.Event()
    def guard():
        S.update(elapsed_seconds=time.monotonic()-START,
                 rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 gpu_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                 all_calls=S['construction_calls']+S['main_calls']+S['warmup_calls'])
        assert S['rss_peak_bytes']<=16*1024**3 and S['gpu_reserved_peak_bytes']<=20*1024**3
        assert prior+S['all_calls']<=4000 and time.time()-global_start<=7200
    def monitor():
        try:
            while not ended.wait(15):
                guard();free=host_free();assert free>=4000000000
                gpu=int(subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())*1024**2
                assert gpu<=20*1024**3
                S['gpu_observed_peak_bytes']=max(S.get('gpu_observed_peak_bytes',0),gpu)
                S['host_min_free_bytes']=min(S.get('host_min_free_bytes',free),free)
                size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
                assert size<=4000000000;S['intermediate_peak_bytes']=max(S.get('intermediate_peak_bytes',0),size)
                write(out/'RUN_STATUS.json',S)
        except Exception:
            S.update(status='paused_resource',failure=traceback.format_exc());write(out/'RUN_STATUS.json',S)
            (out/'EXIT_CODE').write_text('1\n');os._exit(1)
    threading.Thread(target=monitor,daemon=True).start()
    for e in load(BASE/'ASSET_VERIFICATION.json')['files']:
        if 'vendor' in Path(e['path']).parts or 'weights' in Path(e['path']).parts:
            assert sha(e['path'])==e['sha256']
    depth=[0]
    def count(m,a):
        if depth[0]==0:S['construction_calls']+=1
        depth[0]+=1
    def dec(m,a,o):depth[0]-=1
    a=torch.nn.modules.module.register_module_forward_pre_hook(count)
    b=torch.nn.modules.module.register_module_forward_hook(dec,always_call=True)
    index=list(csv.DictReader((BASE/'assets/vendor/omnicloudmask/model_download_links.csv').open()))
    weights=load(REPO/'src/research_results/phase111_20261014/PRE_EXECUTION_LOCK.json')['weights']
    chosen=weights if policy=='E' else [next(w for w in weights if ('regnet' in w['architecture'])==(policy=='R'))]
    load_start=time.perf_counter();models=[]
    try:
        for w in chosen:
            rec=next(r for r in index if r['file_name']==w['google_drive_filename'] and r['version']=='1')
            models.append(mu.load_model_from_weights(model_name=rec['timm_model_name'],weights_path=BASE/'assets/weights'/w['hf_filename'],
                model_library='fastai',device=torch.device('cuda'),dtype=torch.float32,in_chans=3,n_out=4,compile_models=False,patch_size=1000,batch_size=1))
        torch.cuda.synchronize()
    finally:a.remove();b.remove()
    model_info={'policy':policy,'models':len(models),'architectures':[w['architecture'] for w in chosen],
                'weight_bytes':sum((BASE/'assets/weights'/w['hf_filename']).stat().st_size for w in chosen),
                'total_parameters':sum(p.numel() for m in models for p in m.parameters()),
                'trainable_parameters_when_loaded':sum(p.numel() for m in models for p in m.parameters() if p.requires_grad),
                'load_seconds':time.perf_counter()-load_start,'construction_calls':S['construction_calls'],
                'separate_process':True,'PID':os.getpid(),'no_other_member_loaded':policy=='E' or len(models)==1}
    assert len(models)==(2 if policy=='E' else 1) and all(not m.training for m in models)
    for m in models:
        for p in m.parameters():p.requires_grad_(False)
    model_info['trainable_parameters_during_inference']=0;write(out/'MODEL_INFO.json',model_info)
    plan=load(ROOT/'INPUT_PATCH_PLAN.json')['rows'];bench=cfg['benchmark_products']
    pix={r['product_id']:r for r in plan if r['pool']=='PixBox'}
    order=[pix[p] for p in bench]+([r for r in plan if r['pool']=='PixBox' and r['product_id'] not in bench]+[r for r in plan if r['pool']=='Catalogue'] if policy!='E' else [])
    gradient_axis=np.minimum(np.minimum(np.arange(1,1001),np.arange(1000,0,-1)),300).astype(np.float32)*np.float32(1/300)
    independent_gradient=gradient_axis[:,None]*gradient_axis[None,:]
    warm_values=None;receipts=[];scores={}
    # One pre-timing warmup of each required member, on the first fixed benchmark patch.
    warm_array,_,_=image(order[0]['product'],'C20',[q for q in coord if q['product_id']==bench[0]])
    idx=order[0]['actual_patch_coordinates'][0];patch,moved=ru.get_patch(warm_array,idx,0)
    assert list(moved)==idx
    with torch.no_grad():
        batch=torch.from_numpy(patch[None]).cuda();warm_values=[]
        for m in models:
            S['warmup_calls']+=1;warm_values.append(m(batch).detach().cpu().numpy())
    del warm_array,patch,batch;gc.collect();torch.cuda.empty_cache()
    for row in order:
        pool,pid=row['pool'],row['product_id'];benchmark=pool=='PixBox' and pid in bench
        first_output=None
        for repetition in range(2 if benchmark else 1):
            S.update(status='running',pool=pool,product_id=pid,repetition=repetition+1,patch_completed=0)
            guard();excluded=[0.];gpu_seconds=[0.];actual=[];latest={};checks=[]
            wall_start=time.perf_counter()
            if pool=='PixBox':
                points=[q for q in coord if q['product_id']==pid]
                arr,valid,bindings=image(row['product'],'C20',points)
                assert bindings==row['input_bindings']
                yy=np.array([int(q['row20']) for q in points]);xx=np.array([int(q['col20']) for q in points])
                joint=valid[yy,xx].copy();del valid
                cpu=np.zeros((4,len(points)),np.float32);grad=np.zeros(len(points),np.float32)
            else:
                arr,valid,_=catalog_image(row['group'],hash_seconds=excluded)
                cpu=np.zeros((4,1022,1022),np.float32);grad=np.zeros((1022,1022),np.float32)
            before_calls=S['main_calls'];old_infer=mu.infer_batch;old_store=mu.store_results
            def observed_infer(ms,batch):
                captured=[];events={};handles=[]
                def pre_hook(m,args):
                    S['main_calls']+=1
                    assert prior+S['construction_calls']+S['main_calls']+S['warmup_calls']<=4000
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();events[id(m)]=(start,end)
                def post_hook(m,args,result):
                    events[id(m)][1].record();captured.append(result.detach())
                for m in ms:handles.extend([m.register_forward_pre_hook(pre_hook),m.register_forward_hook(post_hook)])
                try:result=old_infer(ms,batch)
                finally:
                    for h in handles:h.remove()
                torch.cuda.synchronize()
                gpu_seconds[0]+=sum(a.elapsed_time(b)/1000 for a,b in events.values())
                check_start=time.perf_counter()
                raw=[t.cpu().numpy().copy() for t in captured]
                assert len(raw)==len(models) and all(t.shape==(1,4,1000,1000) and np.isfinite(t).all() for t in raw)
                independent=np.mean(np.stack(raw),axis=0,dtype=np.float32)
                error=float(np.max(np.abs(independent-result.detach().cpu().numpy())))
                assert error<=1e-5;checks.append(error);latest['mean']=independent
                if warm_values is not None and not receipts and not actual:
                    e=max(float(np.max(np.abs(x-y))) for x,y in zip(warm_values,raw));assert e<=1e-5
                    latest['warm_error']=e
                excluded[0]+=time.perf_counter()-check_start
                return result
            def observed_store(pred_batch,index_batch,pred_tracker,gradient,grad_tracker=None):
                check_start=time.perf_counter()
                assert np.max(np.abs(gradient.detach().cpu().numpy()-independent_gradient))<=1e-7
                for i,(t,b,l,r) in enumerate(index_batch):
                    if pool=='PixBox':
                        inside=(yy>=t)&(yy<b)&(xx>=l)&(xx<r);dy=yy[inside]-t;dx=xx[inside]-l
                        w=independent_gradient[dy,dx]
                        cpu[:,inside]+=latest['mean'][i,:,dy,dx].T*w[None];grad[inside]+=w
                    else:
                        cpu[:,t:b,l:r]+=latest['mean'][i]*independent_gradient
                        grad[t:b,l:r]+=independent_gradient
                    actual.append([t,b,l,r]);S['patch_completed']=len(actual)
                latest['author_grad']=grad_tracker
                excluded[0]+=time.perf_counter()-check_start
                return old_store(pred_batch,index_batch,pred_tracker,gradient,grad_tracker)
            mu.infer_batch=observed_infer;cm.infer_batch=observed_infer;mu.store_results=observed_store
            def forbidden(*a,**k):raise AssertionError('no automatic asset download')
            cm.get_models=forbidden
            try:
                output=cm.predict_from_array(arr,patch_size=1000,patch_overlap=300,batch_size=1,
                    inference_device='cuda',mosaic_device='cpu',inference_dtype=torch.float32,
                    export_confidence=True,softmax_output=True,no_data_value=0,apply_no_data_mask=True,
                    custom_models=models,pred_classes=4,compile_models=False,model_version=1,destination_model_dir=BASE/'assets/weights')
            finally:mu.infer_batch=old_infer;cm.infer_batch=old_infer;mu.store_results=old_store
            if pool=='PixBox':sampled=output[:,yy,xx].copy()
            else:sampled=output[:,valid].copy()
            torch.cuda.synchronize();whole_seconds=time.perf_counter()-wall_start-excluded[0]
            assert actual==row['actual_patch_coordinates']
            assert S['main_calls']-before_calls==len(actual)*len(models)
            validation_start=time.perf_counter()
            author_grad=latest['author_grad'].numpy()
            if pool=='PixBox':
                assert np.array_equal(grad,author_grad[yy,xx]);author_valid=np.any(arr[:,yy,xx]!=0,axis=0)
                raw_logits=cpu/np.maximum(grad[None],np.finfo(np.float32).tiny)
                valid_points=joint&author_valid&(grad>0)
            else:
                assert np.array_equal(grad,author_grad);author_valid=np.any(arr!=0,axis=0)[valid]
                raw_logits=(cpu/np.maximum(grad[None],np.finfo(np.float32).tiny))[:,valid]
                valid_points=author_valid&(grad[valid]>0)
            soft=np.exp(raw_logits-raw_logits.max(0,keepdims=True));soft/=soft.sum(0,keepdims=True)
            ref=np.clip(soft+np.float32(.001),.001,.999)*author_valid[None]
            error=float(np.max(np.abs(ref[:,valid_points]-sampled[:,valid_points]))) if valid_points.any() else 0.
            assert error<=1e-5
            receipt={'pool':pool,'product_id':pid,'policy':policy,'repetition':repetition+1,
                     'model_calls':S['main_calls']-before_calls,'model_GPU_seconds':gpu_seconds[0],
                     'read_to_point_seconds_excluding_SHA_and_independent_validation':whole_seconds,
                     'excluded_inference_CPU_validation_seconds':excluded[0],
                     'post_inference_CPU_validation_seconds':time.perf_counter()-validation_start,
                     'max_confidence_error':error,'max_raw_mean_error':max(checks),'warm_error':latest.get('warm_error'),
                     'valid':int(valid_points.sum()),'planned_points':len(valid_points),'coverage_gradient_exact':True}
            if policy=='E':
                old,_=original_pix_score(pid);by={r['point_id']:r for r in old['points']}
                oldv=np.array([by[q['point_id']]['shadow_confidence'] for q in points],np.float32)
                delta=float(np.max(np.abs(oldv[valid_points]-sampled[3,valid_points])))
                assert delta<=1e-5;receipt['old_E_max_error']=delta
            if repetition==0:
                if pool=='PixBox':
                    payload={'policy':policy,'points':[],'labels_accessed':False}
                    for i,q in enumerate(points):
                        v={k:q[k] for k in ('point_id','product_id','group_id','row10','col10','row20','col20')}
                        v.update(valid=bool(valid_points[i]),shadow_confidence=float(sampled[3,i]) if valid_points[i] else None)
                        payload['points'].append(v)
                    path=out/'scores'/pool/pid/'UNLABELLED_POINT_SCORES.json';write(path,payload)
                else:
                    path=out/'scores'/pool/(pid+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,scores=sampled[3],raw_logits=raw_logits,valid=valid_points,
                                        flat_indices=np.flatnonzero(valid))
                scores[pool+'/'+pid]=sha(path)
                if benchmark:first_output=output
            else:
                repeated_error=float(np.max(np.abs(first_output-output)))
                assert repeated_error<=1e-5;receipt['repeat_full_array_max_error']=repeated_error
                first_output=None
            receipts.append(receipt);write(out/'SCENE_RECEIPTS.json',receipts);guard();write(out/'RUN_STATUS.json',S)
            del arr,cpu,grad,latest,sampled,author_grad,raw_logits,soft,ref,output
            if pool=='Catalogue':del valid
            gc.collect();torch.cuda.empty_cache()
            print(policy,pool,pid,repetition+1,S['main_calls'],flush=True)
        if warm_values is not None:warm_values=None
    write(out/'SCORES_SEALED.json',{'scores_sha256':scores,'before_reference_join':True,'policy':policy})
    guard();S['status']='completed_scores_sealed';ended.set();write(out/'RUN_STATUS.json',S);(out/'EXIT_CODE').write_text('0\n')


def controller():
    setup();assert (ROOT/'PREPARE_EXIT_CODE').read_text().strip()=='0'
    assert not (ROOT/'PID').exists();(ROOT/'PID').write_text(str(os.getpid()))
    state={'status':'running','calls_completed':0,'started_epoch':time.time(),'order':['E','R','N']}
    write(ROOT/'CONTROLLER_STATUS.json',state)
    for policy in state['order']:
        with (ROOT/(policy+'.console.log')).open('xb') as log:
            p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',policy],stdout=log,stderr=subprocess.STDOUT)
            code=p.wait()
        assert code==0,(policy,code)
        s=load(ROOT/policy/'RUN_STATUS.json');assert s['status']=='completed_scores_sealed'
        state['calls_completed']+=s['all_calls'];state['last_completed']=policy
        write(ROOT/'CONTROLLER_STATUS.json',state)
    state.update(status='all_three_settings_scores_sealed_before_reference_join',elapsed_seconds=time.time()-state['started_epoch'])
    assert state['calls_completed']<=4000 and state['elapsed_seconds']<=7200
    write(ROOT/'ALL_SCORES_SEALED.json',{'settings':{p:sha(ROOT/p/'SCORES_SEALED.json') for p in state['order']},'labels_joined':False,'actual_calls':state['calls_completed']})
    write(ROOT/'CONTROLLER_STATUS.json',state);(ROOT/'EXIT_CODE').write_text('0\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',choices=['E','R','N']);args=p.parse_args()
    try:
        if args.prepare:prepare()
        elif args.worker:run_worker(args.worker)
        else:controller()
    except Exception:
        failure=traceback.format_exc();target=ROOT/args.worker if args.worker else ROOT
        if target.exists():
            S.update(status='paused_engineering_no_scientific_verdict',failure=failure)
            write(target/'RUN_STATUS.json',S);(target/('PREPARE_EXIT_CODE' if args.prepare else 'EXIT_CODE')).write_text('1\n')
        print(failure,file=sys.stderr,flush=True);os._exit(1)

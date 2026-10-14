"""Full-scene C20/H10, frozen Omni; no labels or statistical selection on remote."""
import argparse
import csv
import gc
import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time
import traceback
from phase118_common import BASE, ROOT, REPO, coordinates, image, load, ram, sha, write

sys.dont_write_bytecode = True
STATE = {'status': 'initializing', 'main_calls': 0, 'repeat_calls': 0, 'construction_calls': 0}
START = time.monotonic()


def setup():
    cfg = load(REPO/'src/research_results/phase118_20261014/PRE_EXECUTION_LOCK.json')
    for n,h in cfg['source_sha256'].items():
        assert sha(REPO/n) == h, n
    for key in ('HF_HOME','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR','CUDA_CACHE_PATH',
                'TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR','TMPDIR'):
        os.environ[key] = str(BASE/'controlled_caches'/key.lower())
    os.environ.update(HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',GDAL_NUM_THREADS='1',OPJ_NUM_THREADS='1')
    sys.path.insert(0,str(BASE/'assets/vendor'))
    return cfg


def prepare():
    cfg = setup()
    ROOT.mkdir(exist_ok=False)
    (ROOT/'PREPARE_PID').write_text(str(os.getpid()))
    memory = ram()
    write(ROOT/'RAM_GATE.json',dict(memory,required_available=26*1024**3,passed=memory['MemAvailable']>=26*1024**3))
    assert memory['MemAvailable'] >= 26*1024**3
    import numpy as np
    from omnicloudmask import cloud_mask as cm, raster_utils as ru
    from install_phase111_runtime import host_free
    assert host_free() >= 4000000000
    coord = coordinates()
    products = load(REPO/'src/research_results/phase104_20261013/object_manifest.json')['products']
    assert len(products) == 16
    plans = []
    for setting,size,overlap in (('C20',500,150),('H10',1000,300)):
        for product in products:
            pid = product['product_id']; points = [q for q in coord if q['product_id'] == pid]
            array, valid, bindings = image(product,setting,points)
            assert cm.check_patch_size(array,0,size,overlap) == (overlap,size)
            indexes = ru.make_patch_indexes(array.shape[2],array.shape[1],size,overlap)
            actual = []
            for index in indexes:
                patch, moved = ru.get_patch(array,index,0)
                if patch is not None:
                    assert patch.shape == (3,size,size) and np.isfinite(patch).all()
                    actual.append(list(moved))
                del patch
            suffix = '10' if setting == 'H10' else '20'
            yy = np.array([int(q['row'+suffix]) for q in points])
            xx = np.array([int(q['col'+suffix]) for q in points])
            row = {'setting':setting,'product_id':pid,'product':product,'patch_size':size,'overlap':overlap,
                   'actual_patch_coordinates':actual,'input_bindings':bindings,'points':len(points),
                   'valid_fixed_points':int(valid[yy,xx].sum())}
            plans.append(row)
            write(ROOT/'PREPARE_PROGRESS.json',{'rows':len(plans),'model_calls':0,'setting':setting,'product_id':pid,
                  'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 24*1024**3
            del array,valid
            gc.collect()
            print('plan',setting,pid,len(actual),flush=True)
    calls = 2*sum(len(r['actual_patch_coordinates']) for r in plans)+11+8
    assert calls <= 20000
    write(ROOT/'INPUT_PATCH_PLAN.json',{'status':'locked_before_model_construction','rows':plans,
          'planned_calls_including_11_construction_and_8_repeats':calls,'plan_call_cap':20000,
          'total_recovery_inclusive_cap':25000,'labels_accessed':False,'coordinate_source_sha256':cfg['coordinate_sha256'],
          'H10_NIR':'bilinear interpolation of original20m B8A; not new10m observations',
          'validity':'native0/65535 invalid; conservative one-native-pixel halo on interpolated bands; all3 joint support',
          'uint16_bilinear_CPU_bound_DN':1,'final_confidence_tolerance':1e-5})
    (ROOT/'PREPARE_EXIT_CODE').write_text('0\n')


def run():
    cfg = setup()
    assert (ROOT/'PREPARE_EXIT_CODE').read_text().strip() == '0'
    assert not (ROOT/'RUN_PID').exists(), 'no automatic rerun'
    (ROOT/'RUN_PID').write_text(str(os.getpid()))
    memory = ram()
    write(ROOT/'RUN_RAM_GATE.json',dict(memory,passed=memory['MemAvailable']>=26*1024**3))
    assert memory['MemAvailable'] >= 26*1024**3
    import numpy as np
    import torch
    from omnicloudmask import cloud_mask as cm, model_utils as mu
    from install_phase111_runtime import host_free
    torch.set_num_threads(4)
    assert torch.__version__ == '2.11.0+cu128' and torch.cuda.is_available() and not torch.is_autocast_enabled()
    assets = load(BASE/'ASSET_VERIFICATION.json')
    for e in assets['files']:
        if 'vendor' in Path(e['path']).parts or 'weights' in Path(e['path']).parts:
            assert sha(e['path']) == e['sha256']
    plan = load(ROOT/'INPUT_PATCH_PLAN.json')
    write(ROOT/'EXECUTION_PLAN_SEAL.json',{'sha256':sha(ROOT/'INPUT_PATCH_PLAN.json'),'code_sha256':sha(Path(__file__)),
          'planned_calls':plan['planned_calls_including_11_construction_and_8_repeats'],'labels_accessed':False})
    STATE.update(status='loading_models',host_free_before_bytes=host_free(),repeat_max_abs=[],
                 cpu_point_errors=[],cpu_ensemble_errors=[],rss_peak_bytes=0,gpu_reserved_peak_bytes=0)
    finished = threading.Event()

    def check():
        STATE['elapsed_seconds'] = time.monotonic()-START+STATE.get('previous_attempt_seconds',0)
        STATE['rss_peak_bytes'] = max(STATE['rss_peak_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        STATE['gpu_reserved_peak_bytes'] = max(STATE['gpu_reserved_peak_bytes'],torch.cuda.max_memory_reserved())
        STATE['all_calls'] = STATE['main_calls']+STATE['repeat_calls']+STATE['construction_calls']+STATE.get('previous_attempt_calls',0)
        assert STATE['rss_peak_bytes'] <= 24*1024**3 and STATE['gpu_reserved_peak_bytes'] <= 20*1024**3
        assert STATE['elapsed_seconds'] <= (18000 if STATE.get('recovery') else 14400)
        assert STATE['all_calls'] <= (25000 if STATE.get('recovery') else 20000)

    def monitor():
        try:
            while not finished.wait(10):
                check()
                free = host_free(); STATE['host_min_free_bytes'] = min(STATE.get('host_min_free_bytes',free),free)
                assert free >= 4000000000
                size = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
                STATE['intermediate_peak_bytes'] = max(STATE.get('intermediate_peak_bytes',0),size)
                assert size <= 4000000000
                used = int(os.popen('/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits').read().strip())*1024**2
                STATE['gpu_observed_peak_bytes'] = max(STATE.get('gpu_observed_peak_bytes',0),used)
                assert used <= 20*1024**3
                write(ROOT/'RUN_STATUS.json',STATE)
        except Exception:
            STATE.update(status='paused_resource_engineering',failure=traceback.format_exc())
            write(ROOT/'RUN_STATUS.json',STATE)
            os.kill(os.getpid(),signal.SIGTERM)

    threading.Thread(target=monitor,daemon=True).start()
    depth=[0]
    def before(m,a):
        if depth[0] == 0:
            STATE['construction_calls'] += 1
            assert STATE['construction_calls'] <= 11
            write(ROOT/'RUN_STATUS.json',STATE)
        depth[0] += 1
    def after(m,a,o): depth[0] -= 1
    pre=torch.nn.modules.module.register_module_forward_pre_hook(before)
    post=torch.nn.modules.module.register_module_forward_hook(after,always_call=True)
    models=[]
    original_lock=load(REPO/'src/research_results/phase111_20261014/PRE_EXECUTION_LOCK.json')
    with (BASE/'assets/vendor/omnicloudmask/model_download_links.csv').open() as f:
        author_index=list(csv.DictReader(f))
    try:
        for weight in original_lock['weights']:
            rec=next(r for r in author_index if r['file_name']==weight['google_drive_filename'] and r['version']=='1')
            models.append(mu.load_model_from_weights(model_name=rec['timm_model_name'],
                          weights_path=BASE/'assets/weights'/weight['hf_filename'],model_library='fastai',
                          device=torch.device('cuda'),dtype=torch.float32,in_chans=3,n_out=4,
                          compile_models=False,patch_size=1000,batch_size=1))
    finally:
        pre.remove(); post.remove()
    assert all(not m.training for m in models)
    for m in models:
        for p in m.parameters(): p.requires_grad_(False)
    coord=coordinates(); repeated={'C20':0,'H10':0}; receipts=[]; scores_hash={}
    for row in plan['rows']:
        mode,pid,size,overlap = row['setting'],row['product_id'],row['patch_size'],row['overlap']
        STATE.update(status='running',setting=mode,product_id=pid,patch_completed=0)
        out=ROOT/'scores'/mode/pid;out.mkdir(parents=True,exist_ok=False)
        started=time.monotonic(); calls_before=STATE['main_calls']
        points=[q for q in coord if q['product_id']==pid]
        image_array, valid, bindings=image(row['product'],mode,points)
        assert bindings==row['input_bindings']
        suffix='10' if mode=='H10' else '20'
        yy=np.array([int(q['row'+suffix]) for q in points]);xx=np.array([int(q['col'+suffix]) for q in points])
        point_valid=valid[yy,xx].copy();del valid
        cpu=np.zeros((4,len(points)),np.float32); grad=np.zeros(len(points),np.float32)
        axis=np.minimum(np.minimum(np.arange(1,size+1),np.arange(size,0,-1)),overlap).astype(np.float32)*np.float32(1/overlap)
        cpu_gradient=axis[:,None]*axis[None,:]
        latest={};actual_indexes=[];old_infer=mu.infer_batch;old_store=mu.store_results
        def observed_infer(ms,batch):
            collected=[]
            def attempted(m,a):
                STATE['main_calls']+=1
                assert STATE['main_calls']+STATE['repeat_calls']+STATE['construction_calls']+STATE.get('previous_attempt_calls',0)<=(25000 if STATE.get('recovery') else 20000)
            handles=[m.register_forward_pre_hook(attempted) for m in ms]
            handles += [m.register_forward_hook(lambda m,a,o:collected.append(o.detach().cpu().numpy().copy())) for m in ms]
            try: result=old_infer(ms,batch)
            finally:
                for h in handles: h.remove()
            assert len(collected)==2 and all(a.shape==(1,4,size,size) and np.isfinite(a).all() for a in collected)
            mean=np.mean(np.stack(collected),axis=0,dtype=np.float32)
            error=float(np.max(np.abs(mean-result.detach().cpu().numpy())))
            assert error<=1e-5
            latest['mean']=mean;latest['ensemble_error']=max(latest.get('ensemble_error',0),error)
            if repeated[mode] < 2:
                for i,m in enumerate(ms):
                    STATE['repeat_calls']+=1
                    with torch.no_grad(): duplicate=m(batch).detach().cpu().numpy()
                    e=float(np.max(np.abs(duplicate-collected[i])))
                    STATE['repeat_max_abs'].append(e);assert e<=1e-5
                repeated[mode]+=1
            return result
        def observed_store(pred_batch,index_batch,pred_tracker,gradient,grad_tracker=None):
            gradient_error=float(np.max(np.abs(gradient.detach().cpu().numpy()-cpu_gradient)))
            assert gradient_error<=1e-7
            for i,(t,b,l,r) in enumerate(index_batch):
                inside=(yy>=t)&(yy<b)&(xx>=l)&(xx<r);dy=yy[inside]-t;dx=xx[inside]-l
                weights=cpu_gradient[dy,dx]
                cpu[:,inside]+=latest['mean'][i,:,dy,dx].T*weights[None]
                grad[inside]+=weights
                actual_indexes.append([t,b,l,r]);STATE['patch_completed']=len(actual_indexes)
            latest['grad_tracker']=grad_tracker
            return old_store(pred_batch,index_batch,pred_tracker,gradient,grad_tracker)
        mu.infer_batch=observed_infer;cm.infer_batch=observed_infer;mu.store_results=observed_store
        def forbidden(*a,**k): raise AssertionError('no automatic download or fallback')
        cm.get_models=forbidden
        try:
            output=cm.predict_from_array(image_array,patch_size=size,patch_overlap=overlap,batch_size=1,
                  inference_device='cuda',mosaic_device='cpu',inference_dtype=torch.float32,
                  export_confidence=True,softmax_output=True,no_data_value=0,apply_no_data_mask=True,
                  custom_models=models,pred_classes=4,compile_models=False,model_version=1,
                  destination_model_dir=BASE/'assets/weights')
        finally:
            mu.infer_batch=old_infer;cm.infer_batch=old_infer;mu.store_results=old_store
        assert actual_indexes==row['actual_patch_coordinates']
        author_valid=np.any(image_array[:,yy,xx]!=0,axis=0)
        author_grad=latest['grad_tracker'][yy,xx].numpy()
        assert np.array_equal(grad>0,author_grad>0) and np.array_equal(grad,author_grad)
        with np.errstate(invalid='ignore',divide='ignore'):
            raw=cpu/grad[None]
            soft=np.exp(raw-np.max(raw,axis=0,keepdims=True));soft/=soft.sum(0,keepdims=True)
        ref=np.nan_to_num(np.clip(soft+np.float32(.001),.001,.999),nan=0)*author_valid[None]
        sampled=output[:,yy,xx]
        error=float(np.max(np.abs(ref-sampled)));assert error<=1e-5
        score_rows=[]
        for i,q in enumerate(points):
            ok=bool(point_valid[i] and author_valid[i] and grad[i]>0)
            score_rows.append({k:q[k] for k in ('point_id','product_id','group_id','row10','col10','row20','col20')})
            score_rows[-1].update(valid=ok,input_valid=bool(point_valid[i]),author_valid=bool(author_valid[i]),coverage=bool(grad[i]>0),
                                  shadow_confidence=float(sampled[3,i]) if ok else None)
        write(out/'UNLABELLED_POINT_SCORES.json',{'setting':mode,'points':score_rows,'no_labels':True})
        write(out/'CPU_POINT_RECONSTRUCTION.json',{'max_confidence_error':error,'ensemble_max_error':latest['ensemble_error'],
              'coverage_and_point_gradient_exact':True,'raw_point_logits':cpu.tolist(),'point_gradient':grad.tolist(),
              'actual_patch_coordinates':actual_indexes,'full_scene_forward':True,'dense_four_channel_file_saved':False})
        scores_hash[mode+'/'+pid]=sha(out/'UNLABELLED_POINT_SCORES.json')
        receipts.append({'setting':mode,'product_id':pid,'calls':STATE['main_calls']-calls_before,
                         'seconds':time.monotonic()-started,'points':len(points),'valid':sum(q['valid'] for q in score_rows),
                         'score_sha256':scores_hash[mode+'/'+pid],'cpu_error':error})
        STATE['cpu_point_errors'].append(error);STATE['cpu_ensemble_errors'].append(latest['ensemble_error'])
        del output,image_array,cpu,grad,sampled,author_grad,raw,soft,ref,latest
        gc.collect();torch.cuda.empty_cache();check()
        write(ROOT/'SCENE_RECEIPTS.json',receipts);write(ROOT/'RUN_STATUS.json',STATE)
        print('scene done',mode,pid,STATE['all_calls'],flush=True)
    assert len(receipts)==32 and all(v==2 for v in repeated.values()) and STATE['repeat_calls']==8
    check();STATE['status']='all_scores_sealed_before_local_reference_join'
    write(ROOT/'SCORES_SEALED.json',{'score_sha256':scores_hash,'points_each_setting':9500,'labels_accessed':False,
          'before_reference_join':True,'plan_sha256':sha(ROOT/'INPUT_PATCH_PLAN.json'),'all_calls':STATE['all_calls']})
    finished.set();write(ROOT/'RUN_STATUS.json',STATE);(ROOT/'EXIT_CODE').write_text('0\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--recover',action='store_true');args=ap.parse_args()
    try:
        if args.recover:
            assert not args.prepare
            original_root=ROOT
            assert (ROOT/'EXIT_CODE').read_text().strip()=='1'
            old_pid=int((ROOT/'RUN_PID').read_text())
            proc=Path(f'/proc/{old_pid}/stat')
            assert not proc.exists() or proc.read_text().split(') ')[1].split()[0]=='Z'
            old_state=load(ROOT/'RUN_STATUS.json')
            assert old_state['main_calls']==2 and old_state['repeat_calls']==2 and old_state['construction_calls']==11
            assert not (ROOT/'SCENE_RECEIPTS.json').exists()
            ROOT=original_root/'recovery_001';ROOT.mkdir(exist_ok=False)
            for name in ('INPUT_PATCH_PLAN.json','PREPARE_EXIT_CODE','CPU_HISTORICAL_INPUT_SHA_ACCEPTANCE.json'):
                (ROOT/name).write_bytes((original_root/name).read_bytes())
            STATE.update(recovery=True,previous_attempt_calls=15,previous_attempt_seconds=old_state['elapsed_seconds'],
                         first_attempt_preserved=str(original_root),recovery_scope='keyword API fix only; fixed original32 scene plan')
        prepare() if args.prepare else run()
    except Exception:
        if ROOT.exists():
            STATE.update(status='paused_engineering_no_scientific_verdict',failure=traceback.format_exc(),
                         elapsed_seconds=time.monotonic()-START+STATE.get('previous_attempt_seconds',0),
                         all_calls=STATE['main_calls']+STATE['repeat_calls']+STATE['construction_calls']+STATE.get('previous_attempt_calls',0))
            write(ROOT/('PREPARE_FAILURE.json' if args.prepare else 'RUN_STATUS.json'),STATE)
            (ROOT/('PREPARE_EXIT_CODE' if args.prepare else 'EXIT_CODE')).write_text('1\n')
        raise

"""One full locked scene, no labels or performance metrics. No automatic rerun."""
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import resource
import sys
import threading
import time
import traceback
import argparse
sys.dont_write_bytecode=True

REPO=Path(__file__).resolve().parents[2]
BASE=Path('/home/scv/shared/phase111_ocm_runtime')
PHASE=Path('/home/scv/shared/phase112_20261014')
OUT=PHASE/'run'
from install_phase111_runtime import host_free
import gc


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-directory',type=Path,default=OUT)
    parser.add_argument('--execution-lock',type=Path,default=REPO/'src/research_results/phase112_20261014/PRE_EXECUTION_LOCK.json')
    args=parser.parse_args()
    OUT=args.output_directory.resolve();assert OUT.is_relative_to(PHASE)
    OUT.mkdir(exist_ok=False)
    (OUT/'PID').write_text(str(os.getpid()))
    execution_config=json.loads(args.execution_lock.read_text())
    for name,expected in execution_config['source_sha256'].items():
        assert sha(REPO/name)==expected,('pre-forward lock changed',name)
    lock=json.loads((REPO/'src/research_results/phase111_20261014/PRE_EXECUTION_LOCK.json').read_text())
    for name,expected in lock['source_files_sha256'].items():
        assert sha(REPO/name)==expected,('frozen input identity changed',name)
    assets=json.loads((BASE/'ASSET_VERIFICATION.json').read_text())
    assert assets['status']=='all_files_verified'
    assert json.loads((BASE/'RUNTIME_IMPORT_ACCEPTANCE.json').read_text())['dependency_check']=='passed'
    for item in assets['files']:assert sha(item['path'])==item['sha256']
    for key in ['HF_HOME','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR','CUDA_CACHE_PATH',
                'TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR','TMPDIR']:
        os.environ[key]=str(BASE/'controlled_caches'/key.lower())
        Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1',GDAL_NUM_THREADS='1',OPJ_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0,str(BASE/'assets/vendor'))
    import numpy as np
    import torch
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    from scipy.ndimage import binary_dilation
    from omnicloudmask import cloud_mask as cm,model_utils as mu
    torch.set_num_threads(4)
    assert torch.__version__=='2.11.0+cu128' and torch.cuda.is_available()
    assert not torch.is_autocast_enabled()
    synthetic=np.full((8,8),100,np.uint16);synthetic[3,3]=65535
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',height=8,width=8,count=1,dtype='uint16',transform=from_origin(0,80,10,10),nodata=0) as ds:
            ds.write(synthetic,1);resampled=ds.read(1,out_shape=(4,4),resampling=Resampling.bilinear)
    invalid=binary_dilation(synthetic==65535,structure=np.ones((3,3),bool)).reshape(4,2,4,2).any(axis=(1,3))
    assert np.all(invalid[resampled!=100])
    write('SYNTHETIC_INPUT_OPERATOR_CHECK.json',{'status':'passed','saturated_influence_cells':np.argwhere(resampled!=100).tolist(),
           'all_affected_cells_excluded':True,'halo_native_pixels':1,'synthetic_only':True,'model_calls':0})
    started=time.monotonic()
    state={'status':'running','construction_graph_calls':[],'main_calls':0,'repeat_calls':0,
           'patches':[],'repeat_max_abs':[],'cpu_ensemble_max_abs':0.0,'gradient_max_abs':0.0,
           'rss_peak_bytes':0,'gpu_reserved_peak_bytes':0,'gpu_observed_delta_peak_bytes':0}
    state['raw_model_output_ranges']=[None,None]
    previous_calls=execution_config.get('previous_model_call_attempts',0)
    maximum_calls=1935
    state.update(previous_model_call_attempts=previous_calls,max_cumulative_model_calls=maximum_calls)
    globals()['_RUN_STATE']=state;globals()['_RUN_STARTED']=started
    def gpu_used_bytes():
        return int(subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used',
                    '--format=csv,noheader,nounits'],text=True).strip())*1024**2
    baseline_gpu=gpu_used_bytes()
    state['gpu_smi_baseline_used_bytes']=baseline_gpu
    torch.cuda.reset_peak_memory_stats()
    finished=threading.Event()
    def guard():
        try:
            monitor_resources()
        except Exception:
            (PHASE/'RESOURCE_GUARD_FAILURE.txt').write_text(traceback.format_exc())
            os.kill(os.getpid(),signal.SIGTERM)
    def monitor_resources():
        tick=0
        while not finished.wait(2):
            status=Path('/proc/self/status').read_text()
            rss=max(int(s.split()[1])*1024 for s in status.splitlines() if s.startswith(('VmRSS:','VmHWM:')))
            state['rss_peak_bytes']=max(state['rss_peak_bytes'],rss)
            vram=torch.cuda.max_memory_reserved()
            state['gpu_reserved_peak_bytes']=max(state['gpu_reserved_peak_bytes'],vram)
            observed_gpu=max(0,gpu_used_bytes()-baseline_gpu)
            state['gpu_observed_delta_peak_bytes']=max(state['gpu_observed_delta_peak_bytes'],observed_gpu)
            tick+=1
            if tick%15==0:
                assert host_free()>=4000000000
                assert sum(p.stat().st_size for p in PHASE.rglob('*') if p.is_file() and 'B8A' not in p.parts)<=4000000000
            if rss>16*1024**3 or max(vram,observed_gpu)>20*1024**3 or time.monotonic()-started>7200:
                state.update(status='stopped_resource_limit',elapsed_seconds=time.monotonic()-started)
                write('RUN_STATUS.json',state)
                os.kill(os.getpid(),signal.SIGTERM)
    threading.Thread(target=guard,daemon=True).start()
    # Count top-level graph calls made by model constructors, including shape probes.
    depth=[0]
    def before(module,args):
        if depth[0]==0:
            assert len(state['construction_graph_calls'])<11
            state['construction_graph_calls'].append({'module':type(module).__name__,
                'input_shapes':[list(x.shape) for x in args if isinstance(x,torch.Tensor)]})
            write('RUN_STATUS.json',state)
        depth[0]+=1
    def after(module,args,output):depth[0]-=1
    pre=torch.nn.modules.module.register_module_forward_pre_hook(before)
    post=torch.nn.modules.module.register_module_forward_hook(after,always_call=True)
    models=[]
    with (BASE/'assets/vendor/omnicloudmask/model_download_links.csv').open() as f:
        author_index=list(csv.DictReader(f))
    bindings_models=[]
    try:
        for w in lock['weights']:
            matching=[r for r in author_index if r['file_name']==w['google_drive_filename'] and r['version']=='1']
            assert len(matching)==1 and matching[0]['google_drive_id']==w['google_drive_file_id'] and matching[0]['model_library']=='fastai'
            name=matching[0]['timm_model_name']
            bindings_models.append({'timm_model_name':name,'weight_filename':w['hf_filename'],'model_library':'fastai'})
            models.append(mu.load_model_from_weights(model_name=name,weights_path=BASE/'assets/weights'/w['hf_filename'],
                          model_library='fastai',device=torch.device('cuda'),dtype=torch.float32,
                          in_chans=3,n_out=4,compile_models=False,patch_size=1000,batch_size=1))
    finally:
        pre.remove();post.remove()
        write('CONSTRUCTION_CALLS.json',{'calls':state['construction_graph_calls']})
    assert len(models)==2 and all(not m.training for m in models)
    head_channels=[list(m for m in model.modules() if isinstance(m,torch.nn.Conv2d))[-1].out_channels for model in models]
    assert head_channels==[4,4]
    assert all(p.dtype==torch.float32 for model in models for p in model.parameters())
    write('MODEL_LOAD_ACCEPTANCE.json',{'models':2,'state_dict_strict_load':'passed',
          'architecture_bindings':bindings_models,
          'declared_class_order':['Clear','Thick Cloud','Thin Cloud','Shadow'],'shadow_channel':3,
          'semantic_identity_basis':'fixed author V1 ensemble and source documentation; not inferred from labels',
          'actual_last_conv_output_channels':head_channels,
          'parameter_counts':[sum(p.numel() for p in model.parameters()) for model in models],
          'parameter_dtype':'torch.float32','autocast':False,
          'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
          'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
          'construction_graph_calls':len(state['construction_graph_calls'])})
    downloads=json.loads((PHASE/'DOWNLOAD_STATUS.json').read_text())
    assert downloads['status']=='all_files_verified' and len(downloads['files'])==15
    root_output=OUT; scene_receipts=[]
    (PHASE/'RUN_PID').write_text(str(os.getpid()))
    for scene_id in execution_config['order']:
        OUT=root_output/scene_id;OUT.mkdir(exist_ok=False)
        scene_started=time.monotonic();calls_before=state['main_calls']
        state.update(status='running',product_id=scene_id,patches=[],raw_model_output_ranges=[None,None],cpu_ensemble_max_abs=0.0,gradient_max_abs=0.0)
        assert host_free()>=4000000000
        manifest=json.loads((REPO/'src/research_results/phase104_20261013/object_manifest.json').read_text())
        pid=scene_id;product=next(p for p in manifest['products'] if p['product_id']==pid)
        original=Path('/home/scv/shared/data/pixbox_s2_cmix_5036991/phase104_20261013')
        original_status=json.loads((original/'download_status.json').read_text())
        arrays=[];valid=[];bindings=[]
        for band in ['B04','B03','B8A']:
            path=Path(downloads['files'][pid]['path']) if band=='B8A' else original/'imagery'/pid/(band+'.jp2')
            expected=downloads['files'][pid]['sha256'] if band=='B8A' else original_status['files'][pid+'/'+band+'.jp2']['sha256']
            assert sha(path)==expected
            with rasterio.open(path) as ds:
                g=product['grid20' if band=='B8A' else 'grid10']
                assert ds.crs.to_string()==product['crs'] and (ds.height,ds.width)==(g['NROWS'],g['NCOLS'])
                assert ds.count==1 and ds.dtypes==('uint16',)
                assert tuple(ds.transform)[:6]==(g['XDIM'],0,g['ULX'],0,g['YDIM'],g['ULY'])
                native=ds.read(1)
                native_valid=np.isfinite(native)&(native!=0)&(native!=65535)
                # Preserve author's uint16 bilinear read, separately propagate the
                # frozen native invalid footprint to the common 20m validity mask.
                a=ds.read(1,out_shape=(5490,5490),resampling=Resampling.bilinear) if band!='B8A' else native
                v=(~binary_dilation(~native_valid,structure=np.ones((3,3),bool))).reshape(5490,2,5490,2).all(axis=(1,3)) if band!='B8A' else native_valid
                a=a.copy();a[~v]=0
                arrays.append(a);valid.append(v)
                bindings.append({'band':band,'path':str(path),'sha256':expected,'native_shape':[ds.height,ds.width],
                                 'target_shape':[5490,5490],'bilinear_native_uint16':band!='B8A'})
                del native,native_valid
        image=np.stack(arrays);original_valid=np.logical_and.reduce(valid)
        del arrays,valid
        checked_overlap,checked_patch=cm.check_patch_size(image,0,1000,300)
        assert (checked_overlap,checked_patch)==(300,1000),'Author helper would alter frozen patch configuration'
        coordinate=Path('/home/scv/shared/phase103_positions_20261013/coordinate_metadata.csv')
        assert sha(coordinate)=='c2f029a349096738935d9239585c67ca6dfc5a4c71017f5e8634af982b86ce68'
        with coordinate.open(encoding='utf-8-sig') as f:
            reader=csv.DictReader(f);assert not set(reader.fieldnames)&{'label','class','reference','prediction'}
            points=[r for r in reader if r['product_id']==pid]
        write('INPUT_ACCEPTANCE.json',{'product_id':pid,'points':len(points),'bindings':bindings,
              'coordinate_sha256':sha(coordinate),'no_labels':True,'invalid_rule':'native 0/65535/nonfinite; one-native-pixel Chebyshev halo covers bilinear interpolation support, then any corresponding 2x2 invalid excludes20m cell; invalid channel set to nodata before normalization; B8A native20m no halo'})
        if execution_config.get('authorized_recovery'):
            assert sha(OUT/'INPUT_ACCEPTANCE.json')==sha(Path(execution_config['prior_attempt_directory'])/'INPUT_ACCEPTANCE.json')
        cpu=np.zeros((4,5490,5490),dtype=np.float32);grad=np.zeros((5490,5490),dtype=np.float32)
        # Author CUDA scalar division uses a rounded float32 reciprocal. NumPy
        # direct division has a different last-bit rounding; derive it independently.
        axis=np.minimum(np.minimum(np.arange(1,1001),np.arange(1000,0,-1)),300).astype(np.float32)*np.float32(1/300)
        gradient=axis[:,None]*axis[None,:]
        latest={};original_infer=mu.infer_batch;original_store=mu.store_results
        def observed_infer(ms,patch_batch):
            collected=[]
            def attempted(module,args):
                assert state['main_calls']<1920
                state['main_calls']+=1
                write('RUN_STATUS.json',state)
            handles=[m.register_forward_pre_hook(attempted) for m in ms]
            handles.extend(m.register_forward_hook(lambda m,a,o:collected.append(o.detach().cpu().numpy().copy())) for m in ms)
            try: result=original_infer(ms,patch_batch)
            finally:
                for h in handles:h.remove()
            assert len(collected)==2 and all(x.shape==(1,4,1000,1000) and x.dtype==np.float32 and np.isfinite(x).all() for x in collected)
            for i,x in enumerate(collected):
                old=state['raw_model_output_ranges'][i]
                state['raw_model_output_ranges'][i]=[min(float(x.min()),old[0]) if old else float(x.min()),
                                                     max(float(x.max()),old[1]) if old else float(x.max())]
            mean=np.mean(np.stack(collected),axis=0,dtype=np.float32)
            assert np.isfinite(mean).all() and torch.isfinite(result).all().item()
            state['cpu_ensemble_max_abs']=max(state['cpu_ensemble_max_abs'],float(np.max(np.abs(mean-result.detach().cpu().numpy()))))
            latest['mean']=mean
            patch_no=state['main_calls']//2
            if patch_no<=2:
                np.save(OUT/f'normalized_patch_{patch_no}.npy',patch_batch.detach().cpu().numpy(),allow_pickle=False)
                if execution_config.get('authorized_recovery'):
                    assert sha(OUT/f'normalized_patch_{patch_no}.npy')==sha(Path(execution_config['prior_attempt_directory'])/f'normalized_patch_{patch_no}.npy')
                for i,m in enumerate(ms):
                    assert state['repeat_calls']<4
                    state['repeat_calls']+=1
                    write('RUN_STATUS.json',state)
                    with torch.no_grad():repeat=m(patch_batch).detach().cpu().numpy()
                    assert repeat.shape==collected[i].shape and np.isfinite(repeat).all()
                    difference=float(np.max(np.abs(collected[i]-repeat)))
                    state['repeat_max_abs'].append(difference)
                    assert difference<=1e-5,'repeat tolerance failed; no retry'
            assert previous_calls+state['main_calls']+state['repeat_calls']+len(state['construction_graph_calls'])<=maximum_calls
            write('RUN_STATUS.json',state)
            return result
        def observed_store(pred_batch,index_batch,pred_tracker,gradient,grad_tracker=None):
            observed=gradient.detach().cpu().numpy()
            state['gradient_max_abs']=max(state['gradient_max_abs'],float(np.max(np.abs(observed-globals_gradient))))
            for pred,index in zip(latest['mean'],index_batch):
                t,b,l,r=index;cpu[:,t:b,l:r]+=pred*globals_gradient;grad[t:b,l:r]+=globals_gradient
                state['patches'].append(list(index))
            latest['author_grad_tracker']=grad_tracker
            return original_store(pred_batch,index_batch,pred_tracker,gradient,grad_tracker)
        globals_gradient=gradient
        mu.infer_batch=observed_infer;cm.infer_batch=observed_infer;mu.store_results=observed_store
        def no_automatic_models(*args,**kwargs):raise AssertionError('Automatic model download/fallback is forbidden')
        cm.get_models=no_automatic_models
        try:
            output=cm.predict_from_array(image,patch_size=1000,patch_overlap=300,batch_size=1,
                   inference_device='cuda',mosaic_device='cpu',inference_dtype=torch.float32,
                   export_confidence=True,softmax_output=True,no_data_value=0,apply_no_data_mask=True,
                   custom_models=models,pred_classes=4,compile_models=False,model_version=1,
                   destination_model_dir=BASE/'assets/weights')
        finally:
            mu.infer_batch=original_infer;mu.store_results=original_store
        author_valid=np.any(image!=0,axis=0);coverage=grad>0
        max_error=0.;confidence_min=1.;confidence_max=0.;sum_min=4.;sum_max=0.
        for y in range(0,5490,128):
            sl=slice(y,min(y+128,5490));g=grad[sl]
            with np.errstate(divide='ignore',invalid='ignore'):
                raw=cpu[:,sl,:]/g[None,:,:]
                soft=np.exp(raw-np.max(raw,axis=0,keepdims=True));soft/=soft.sum(axis=0,keepdims=True)
            ref=np.nan_to_num(np.clip(soft+np.float32(.001),.001,.999),nan=0.)*author_valid[sl][None,:,:]
            max_error=max(max_error,float(np.max(np.abs(ref-output[:,sl,:]))))
            common=author_valid[sl]&coverage[sl]
            if common.any():
                confidence_min=min(confidence_min,float(output[:,sl,:][:,common].min()))
                confidence_max=max(confidence_max,float(output[:,sl,:][:,common].max()))
                sums=output[:,sl,:][:,common].sum(axis=0)
                sum_min=min(sum_min,float(sums.min()));sum_max=max(sum_max,float(sums.max()))
        assert output.shape==(4,5490,5490) and np.isfinite(output).all()
        gradient_field_max_error=float(np.max(np.abs(grad-latest['author_grad_tracker'].cpu().numpy())))
        same_coverage=bool(np.array_equal(coverage,latest['author_grad_tracker'].cpu().numpy()>0))
        write('CPU_PRE_ACCEPTANCE.json',{'confidence_max_error':max_error,
              'gradient_field_max_error':gradient_field_max_error,'coverage_exactly_equal':same_coverage,
              'confidence_range':[confidence_min,confidence_max]})
        assert max_error<=1e-5 and confidence_min>=.001-1e-7 and confidence_max<=.999+1e-7
        assert same_coverage and gradient_field_max_error==0
        scores=[]
        for p in points:
            y,x=int(p['row20']),int(p['col20']);okay=bool(original_valid[y,x]&author_valid[y,x]&coverage[y,x])
            scores.append({'point_id':p['point_id'],'product_id':pid,'row20':y,'col20':x,
                           'original_valid':bool(original_valid[y,x]),'author_valid':bool(author_valid[y,x]),
                           'coverage':bool(coverage[y,x]),'valid':okay,'shadow_confidence':float(output[3,y,x]) if okay else None})
        write('UNLABELLED_POINT_SCORES.json',{'points':scores,'shadow_channel':3,'confidence_not_calibrated_probability':True})
        np.savez_compressed(OUT/'VALID_COVERAGE_MASKS.npz',original_valid=np.packbits(original_valid),author_valid=np.packbits(author_valid),coverage=np.packbits(coverage),shape=np.array([5490,5490]))
        write('CPU_OUTPUT_RECONSTRUCTION.json',{'max_absolute_error':max_error,'independent_ensemble_max_error':state['cpu_ensemble_max_abs'],
              'gradient_max_error':state['gradient_max_abs'],'full_scene_cpu_reconstructed_in_memory':True,
              'gradient_field_max_error':gradient_field_max_error,'coverage_exactly_equal':same_coverage,
              'coverage_zero_cells':int((~coverage).sum()),'nodata_cells':int((~author_valid).sum()),
              'valid_confidence_range':[confidence_min,confidence_max],'four_channels_sum_one_required':False})
        write('OUTPUT_SEMANTICS.json',{'raw_model_output_ranges':state['raw_model_output_ranges'],
              'raw_model_output_shape':[1,4,1000,1000],'raw_and_final_finite':True,
              'valid_final_confidence_range':[confidence_min,confidence_max],
              'valid_final_four_channel_sum_range':[sum_min,sum_max],
              'operator':'mean raw classes, gradient mosaic/division, class softmax, +0.001, clip[0.001,0.999]',
              'confidence_not_calibrated_probability':True})
        state.update(status='runtime_single_scene_passed',elapsed_seconds=time.monotonic()-scene_started,
                     unlabelled_score_sha256=sha(OUT/'UNLABELLED_POINT_SCORES.json'),point_count=len(points),
                     valid_points=sum(x['valid'] for x in scores),training=0,fitting=0,labels_accessed=False,performance_evaluated=False)
        state['rss_peak_bytes']=max(state['rss_peak_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        state['gpu_reserved_peak_bytes']=torch.cuda.max_memory_reserved()
        assert state['rss_peak_bytes']<=16*1024**3 and state['gpu_reserved_peak_bytes']<=20*1024**3
        assert state['main_calls']-calls_before<=128
        write('RUN_STATUS.json',state)
        scene_receipts.append({'product_id':scene_id,'main_calls':state['main_calls']-calls_before,'seconds':state['elapsed_seconds'],'score_sha256':state['unlabelled_score_sha256'],'points':state['point_count'],'valid_points':state['valid_points'],'cpu_confidence_error':max_error,'gradient_field_error':gradient_field_max_error,'coverage_equal':same_coverage})
        (PHASE/'SCENE_RECEIPTS.json').write_text(json.dumps(scene_receipts,indent=2)+'\n')
        del output,cpu,grad,image,original_valid,author_valid,coverage,latest,globals_gradient,gradient
        gc.collect();torch.cuda.empty_cache()
    OUT=root_output;state.update(status='all_15_scenes_passed',elapsed_seconds=time.monotonic()-started,scenes=15)
    finished.set();write('RUN_STATUS.json',state)
    print(json.dumps({k:v for k,v in state.items() if k not in ['patches','construction_graph_calls']}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():(OUT/'FAILURE.txt').write_text(traceback.format_exc())
        state=globals().get('_RUN_STATE')
        if state is not None:
            state.update(status='failed_no_automatic_rerun',failure_type=sys.exc_info()[0].__name__,
                         elapsed_seconds=time.monotonic()-globals()['_RUN_STARTED'])
            state['rss_peak_bytes']=max(state['rss_peak_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            torch=sys.modules.get('torch')
            if torch is not None and torch.cuda.is_initialized():
                state['gpu_reserved_peak_bytes']=max(state['gpu_reserved_peak_bytes'],torch.cuda.max_memory_reserved())
            write('RUN_STATUS.json',state)
        raise

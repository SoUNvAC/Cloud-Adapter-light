"""Acquire and run the predeclared single-case V1/V4 comparison; no training."""
import argparse,base64,csv,datetime,gc,hashlib,json,os,resource,signal,sys,threading,time,traceback,urllib.request,xml.etree.ElementTree as ET
from phase200_common import *
sys.dont_write_bytecode=True
STATE={};STAGE=None;START=0
def status(**kw):
    STATE.update(kw);write(ROOT/STAGE/'STATUS.json',STATE)
def check_sources(plan):
    for p,h in plan['source_sha256'].items():assert sha(REPO/p)==h,p
def acquire(plan):
    import numpy as np,rasterio
    from rasterio.enums import Resampling
    from scipy.ndimage import binary_dilation
    from PIL import Image
    from install_phase111_runtime import host_free
    assets=load(BASE/'ASSET_VERIFICATION.json');assert assets['status']=='all_files_verified'
    assert load(BASE/'RUNTIME_IMPORT_ACCEPTANCE.json')['dependency_check']=='passed'
    for x in assets['files']:
        if '/vendor/' in x['path'] or '/weights/' in x['path']:assert sha(x['path'])==x['sha256'],x['path']
    records=[];received=0
    for x in plan['objects']+plan['weights_V4']:
        assert host_free()-x['bytes']>=plan['host_min_free_bytes']
        path=ROOT/'assets'/((x['band']+'.jp2') if x.get('band','').startswith('B') else (x['band']+'.xml') if 'band' in x else x['filename'])
        path.parent.mkdir(parents=True,exist_ok=True);part=path.with_name(path.name+'.partial');assert not path.exists() and not part.exists()
        status(status='downloading',current=path.name,received_bytes=received,verified=records)
        h=hashlib.sha256();md5=hashlib.md5();n=0
        with urllib.request.urlopen(x['url'],timeout=90) as r,part.open('xb') as out:
            assert int(r.headers.get('Content-Length',x['bytes']))==x['bytes']
            for b in iter(lambda:r.read(1024**2),b''):
                received+=len(b);n+=len(b);assert received<=plan['download_cap_bytes'] and n<=x['bytes']
                out.write(b);h.update(b);md5.update(b)
                if n%(16*1024**2)==0:status(received_bytes=received,current_bytes=n)
        assert n==x['bytes']
        if 'sha256' in x:assert h.hexdigest()==x['sha256']
        if 'md5_base64' in x:assert base64.b64encode(md5.digest()).decode()==x['md5_base64']
        os.replace(part,path);records.append(dict(path=str(path),sha256=h.hexdigest(),bytes=n,source=x))
        status(received_bytes=received,verified=records);print('Verified',path.name,n,flush=True)
    write(ROOT/'ASSET_VERIFICATION.json',dict(status='all_files_verified',files=records,actual_download_bytes=received))
    xml=ET.parse(ROOT/'assets/MTD_MSIL1C.xml');vals=lambda name:[n.text for n in xml.iter() if n.tag.split('}')[-1]==name]
    radiometry=dict(quantification=vals('QUANTIFICATION_VALUE'),offsets=vals('RADIO_ADD_OFFSET'),special_text=vals('SPECIAL_VALUE_TEXT'),special_index=vals('SPECIAL_VALUE_INDEX'))
    assert radiometry['quantification']==['10000']
    arrays=[];valids=[];profile=None;bindings=[]
    for band in ['B04','B03','B8A']:
        p=ROOT/'assets'/(band+'.jp2')
        with rasterio.open(p) as ds:
            assert ds.count==1 and ds.dtypes==('uint16',) and ds.crs.to_epsg()==32611
            native=ds.read(1);v=(native!=0)&(native!=65535)
            if band!='B8A':
                assert native.shape==(10980,10980) and ds.res==(10,10)
                arr=ds.read(1,out_shape=(5490,5490),resampling=Resampling.bilinear)
                v=(~binary_dilation(~v,structure=np.ones((3,3),bool))).reshape(5490,2,5490,2).all(axis=(1,3))
            else:
                assert native.shape==(5490,5490) and ds.res==(20,20);arr=native;profile=ds.profile.copy()
            bindings.append(dict(band=band,native_shape=[ds.height,ds.width],crs=str(ds.crs),transform=list(ds.transform),sha256=sha(p)))
            arrays.append(arr);valids.append(v);del native
    image=np.stack(arrays);valid=np.logical_and.reduce(valids);image[:,~valid]=0
    del arrays,valids
    np.save(ROOT/'image20.npy',image,allow_pickle=False);np.save(ROOT/'valid20.npy',valid,allow_pickle=False)
    from rasterio.windows import from_bounds
    bbox=plan['display_bbox_utm11N'];win=from_bounds(*bbox,profile['transform']).round_offsets().round_lengths()
    r,c,h,w=map(int,[win.row_off,win.col_off,win.height,win.width]);assert r>=0 and c>=0 and r+h<=5490 and c+w<=5490
    # Display only: physical-offset correction. Inference retains author-loader DN convention.
    rgb=[]
    for band in ['B04','B03','B02']:
        with rasterio.open(ROOT/'assets'/(band+'.jp2')) as ds:
            from rasterio.windows import Window
            a=ds.read(1,window=Window(c*2,r*2,w*2,h*2),out_shape=(h,w),resampling=Resampling.bilinear)
        rgb.append(np.clip((a.astype(np.float32)-1000)/10000/.3,0,1)**(1/2.2))
    rgb=np.stack(rgb,axis=-1);rgb[~valid[r:r+h,c:c+w]]=0
    Image.fromarray((rgb*255).astype('uint8')).save(ROOT/'true_color.png')
    write(ROOT/'INPUT_RECEIPT.json',dict(product_id=plan['product_id'],bindings=bindings,radiometry=radiometry,
        image_sha256=sha(ROOT/'image20.npy'),valid_sha256=sha(ROOT/'valid20.npy'),valid_pixels=int(valid.sum()),
        transform=list(profile['transform']),crs=str(profile['crs']),shape=[5490,5490],display_window=dict(row=r,col=c,height=h,width=w),
        display_sha256=sha(ROOT/'true_color.png'),inference_radiometry='native DN, no offset correction, follows locked official loader',
        display_radiometry='(DN-1000)/10000; fixed 0..0.3 stretch gamma1/2.2; NOT inference input',
        paper_case_coverage=True,wall_seconds=time.monotonic()-START,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=plan['cpu_cap_bytes']
    status(status='prepared',received_bytes=received,verified_objects=len(records),model_calls=0,seconds=time.monotonic()-START)
def infer(plan):
    import numpy as np,torch
    from omnicloudmask import cloud_mask as cm,model_utils as mu
    from install_phase111_runtime import host_free
    assert torch.__version__=='2.11.0+cu128' and torch.cuda.is_available();torch.set_num_threads(4)
    sealed=load(ROOT/'POINTS_SEALED.json');assert sha(ROOT/'candidate_points.csv')==sealed['points_sha256']
    receipt=load(ROOT/'INPUT_RECEIPT.json');assert receipt['image_sha256']==sha(ROOT/'image20.npy') and receipt['valid_sha256']==sha(ROOT/'valid20.npy')
    image=np.load(ROOT/'image20.npy');valid=np.load(ROOT/'valid20.npy');assert image.shape==(3,5490,5490)
    assert cm.check_patch_size(image,0,1000,300)==(300,1000)
    with (BASE/'assets/vendor/omnicloudmask/model_download_links.csv').open() as f:index=list(csv.DictReader(f))
    assets=load(BASE/'ASSET_VERIFICATION.json');asset_map={Path(x['path']).name:x for x in assets['files']}
    for x in assets['files']:
        if '/vendor/' in x['path']:assert sha(x['path'])==x['sha256']
    STATE.update(status='running',model_calls=0,construction_calls=0,main_calls=0,versions={},cpu_peak_bytes=0,gpu_peak_bytes=0)
    finished=threading.Event();gpu_start=time.monotonic()
    def guard():
        while not finished.wait(15):
            try:
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;gpu=torch.cuda.max_memory_reserved()
                STATE['cpu_peak_bytes']=max(STATE['cpu_peak_bytes'],rss);STATE['gpu_peak_bytes']=max(STATE['gpu_peak_bytes'],gpu)
                reason=None
                if rss>plan['cpu_cap_bytes'] or gpu>plan['gpu_cap_bytes']:reason='memory'
                if time.monotonic()-gpu_start>plan['wall_seconds_cap']:reason='wall'
                if host_free()<plan['host_min_free_bytes']:reason='host_disk'
                if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,10,19,16,tzinfo=datetime.timezone.utc):reason='deadline'
                if reason:status(status='stopped_resource',reason=reason);os.kill(os.getpid(),signal.SIGTERM)
            except Exception:
                status(status='stopped_guard_failure',guard_error=traceback.format_exc());os.kill(os.getpid(),signal.SIGTERM)
    threading.Thread(target=guard,daemon=True).start()
    default_post=cm.postprocess_preds
    for version in [1,4]:
        started=time.monotonic();models=[];depth=[0];current=f'V{version}';torch.cuda.reset_peak_memory_stats()
        status(current=current)
        def before(m,args):
            if depth[0]==0:
                assert STATE['model_calls']<plan['planned_all_calls_cap'];STATE['model_calls']+=1;STATE['construction_calls']+=1;status()
            depth[0]+=1
        def after(m,args,out):depth[0]-=1
        pre=torch.nn.modules.module.register_module_forward_pre_hook(before);post=torch.nn.modules.module.register_module_forward_hook(after,always_call=True)
        bindings=[]
        try:
            for row in [r for r in index if r['version']==str(version)]:
                if version==1:
                    w=next(x for x in plan['weights_V1'] if x['google_drive_filename']==row['file_name']);p=BASE/'assets/weights'/w['hf_filename'];expected=w['declared_sha256']
                else:
                    w=next(x for x in plan['weights_V4'] if x['filename']==row['file_name']);p=ROOT/'assets'/w['filename'];expected=w['sha256']
                assert sha(p)==expected
                models.append(mu.load_model_from_weights(model_name=row['timm_model_name'],weights_path=p,model_library=row['model_library'],device=torch.device('cuda'),dtype=torch.float32,in_chans=3,n_out=4,compile_models=False,patch_size=1000,batch_size=1))
                bindings.append(dict(path=str(p),sha256=expected,architecture=row['timm_model_name'],library=row['model_library']))
        finally:pre.remove();post.remove()
        assert len(models)==2 and all(not m.training for m in models)
        calls_before=STATE['main_calls'];classes_path=ROOT/(current+'_classes.npy')
        def attempted(m,args):
            assert STATE['model_calls']<plan['planned_all_calls_cap'];STATE['model_calls']+=1;STATE['main_calls']+=1;status()
        handles=[m.register_forward_pre_hook(attempted) for m in models]
        def postprocess(preds,export_confidence,softmax_output,class_dim=0):
            # Classify raw class scores with the official default operation before clipping.
            cls=default_post(preds,False,False,class_dim=class_dim).cpu().numpy()[0]
            cls=cls.copy();cls[~valid]=255;np.save(classes_path,cls,allow_pickle=False)
            return default_post(preds,export_confidence,softmax_output,class_dim=class_dim)
        cm.postprocess_preds=postprocess
        try:
            result=cm.predict_from_array(image,patch_size=1000,patch_overlap=300,batch_size=1,inference_device='cuda',mosaic_device='cpu',inference_dtype=torch.float32,export_confidence=True,softmax_output=True,no_data_value=0,apply_no_data_mask=True,custom_models=models,pred_classes=4,compile_models=False,model_version=version,destination_model_dir=ROOT/'assets')
        finally:
            cm.postprocess_preds=default_post
            for h in handles:h.remove()
        assert result.shape==(4,5490,5490) and result.dtype==np.float32 and np.isfinite(result).all()
        assert STATE['main_calls']-calls_before<=128
        coverage=result.sum(axis=0)>0;assert np.all(coverage[valid])
        p=ROOT/(current+'_shadow.npy');np.save(p,result[3],allow_pickle=False)
        cls=np.load(classes_path);assert set(np.unique(cls))<=set([0,1,2,3,255])
        STATE['versions'][current]=dict(status='completed',bindings=bindings,main_calls=STATE['main_calls']-calls_before,seconds=time.monotonic()-started,shadow_sha256=sha(p),classes_sha256=sha(classes_path),gpu_peak_bytes=torch.cuda.max_memory_reserved(),cpu_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            decision='official raw-class argmax; 0Clear/1ThickCloud/2ThinCloud/3Shadow; invalid255',confidence='official softmax +0.001 then clip; not calibrated probability',valid_pixels=int(valid.sum()))
        status();del result,models,cls,coverage;mu.load_model_from_weights.cache_clear();gc.collect();torch.cuda.empty_cache()
    finished.set();status(status='completed',gpu_phase_seconds=time.monotonic()-gpu_start,seconds=time.monotonic()-START,points_sha256=sealed['points_sha256'],cpu_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
def main():
    global STAGE,START
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['acquire','infer']);a=p.parse_args();STAGE=a.stage;START=time.monotonic()
    plan=load(LOCK);check_sources(plan);setup()
    d=ROOT/STAGE;assert d.exists() and not (d/'STATUS.json').exists();(d/'WORKER_PID').write_text(str(os.getpid()))
    try:
        if STAGE=='acquire':acquire(plan)
        else:infer(plan)
        (d/'EXIT_CODE').write_text('0\n')
    except BaseException:
        status(status='failed',error=traceback.format_exc(),seconds=time.monotonic()-START);(d/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()

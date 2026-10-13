"""529 original Source windows plus2 repeats; only point scores persisted."""
import os,time,collections,resource,sys
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ.setdefault('GDAL_NUM_THREADS','1');os.environ.setdefault('OPJ_NUM_THREADS','1')
import numpy as np
import rasterio
from rasterio.windows import Window
from phase105_common import *

def freeze():
    lock=read(RESULT/'task_lock.json')
    assert sha(PROTOCOL)==lock['protocol_sha256']
    assert all(sha(REPO/p)==h for p,h in lock['code_sha256'].items())
    prior=REPO/'src/research_results/phase104_20261013'
    assert all(sha(prior/n)==h for n,h in lock['prior_sha256'].items())
    assert not (ROOT/'evaluation_lock.json').exists(),'Preserve any prior attempt; no automatic reforward'
    assert sha(SOURCE)==SOURCE_SHA
    for n,h in MODEL_SHA.items():assert sha(MODELS/(n+'.pkl'))==h
    runtime=read(prior/'runtime_chain_acceptance.json')
    for p,h in runtime['config_inheritance_sha256'].items():assert sha(REPO/p)==h
    op=read(prior/'input_operator_lock.json')
    for p,h in op['original_chain_sha256'].items():assert sha(REPO/p)==h
    cat=SHARED/'data/sentinel2_cloud_mask_catalogue_4172871'
    control=read(cat/'phase67_20261010/control_lock.json');recovery=cat/'phase65d_recovery_20261008/recovery_lock.json'
    assert control['source_checkpoint_sha256']==SOURCE_SHA and read(recovery)['source_checkpoint_sha256']==SOURCE_SHA
    assert sha(recovery)==control['recovery_lock_sha256']
    fit=read(MODELS/'fit_input_lock.json')
    prior92=SHARED/'phase92_20261012/delivery/fit_input_lock.json'
    assert sha(prior92)==fit['phase92_fit_input_sha256']
    assert sha(cat/'phase67_20261010/fit_input_audit.json')==read(prior92)['original_fit_input_audit_sha256']
    assert sha(MODELS/'fit_input_lock.json')==read(MODELS/'model_G7_lock.json')['fit_input_lock_sha256']
    accept=read(INPUT/'input_acceptance.json')
    assert sha(INPUT/'point_inputs_private.json')==accept['point_inputs_sha256']
    assert sha(INPUT/'window_index.json')==accept['window_index_sha256']
    status=read(INPUT/'download_status.json');assert status['status']=='all_files_verified'
    for row in status['files'].values():assert sha(INPUT/'imagery'/row['path'])==row['sha256']
    windows=read(INPUT/'window_index.json')['rows'];assert len(windows)==529
    assert [key(w) for w in repeats(windows)]==lock['repeats']
    write(ROOT/'evaluation_lock.json',dict(created_utc=now(),local_task_lock_sha256=sha(RESULT/'task_lock.json'),
        source_checkpoint_sha256=SOURCE_SHA,original153_models=MODEL_SHA,window_index_sha256=sha(INPUT/'window_index.json'),
        no_label_inputs=True,Source_lineage=dict(phase93_fit_input_sha256=sha(MODELS/'fit_input_lock.json'),
          phase67_control_sha256=sha(cat/'phase67_20261010/control_lock.json'),recovery_lock_sha256=sha(recovery)),
        actual_code_sha256=code(),original_runtime=runtime,max_forwards=531),True)
    return windows

def main():
    started=time.monotonic();windows=freeze()
    import torch
    from mmengine.dataset import pseudo_collate
    # Reuse the accepted implementation, without calling its old prepare/evaluate routines.
    import run_phase89 as original
    model,pipeline=original.setup('Source');before=original.state_sha(model)
    assert model.align_corners is False and not model.backbone.target_enabled
    rows=read(INPUT/'point_inputs_private.json')['rows'];points=collections.defaultdict(list)
    for r in rows:points[f"{r['product_id']}|{r['window']['y']}|{r['window']['x']}"].append(r)
    assert len(points)==529 and sum(map(len,points.values()))==9500
    dest=ROOT/'windows';dest.mkdir(exist_ok=False);counter=0;receipts=[];all_scores=[]
    torch.cuda.reset_peak_memory_stats();torch.set_grad_enabled(False)
    def one(w,repeat=False):
        nonlocal counter
        assert datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()<'2026-10-20','Budget deadline'
        k=key(w);rr=points[k];h,wid=w['height'],w['width'];spec=np.empty((h,wid,3),np.float32)
        for j,b in enumerate(('B04','B03','B02')):
            with rasterio.open(INPUT/'imagery'/w['product_id']/(b+'.jp2')) as ds:
                dn=ds.read(1,window=Window(w['x']*2,w['y']*2,wid*2,h*2))
            ok=((dn!=0)&(dn!=65535)).reshape(h,2,wid,2).all((1,3))
            toa=dn.astype(np.float32)/10000.
            a=toa.reshape(h,2,wid,2).mean((1,3),dtype=np.float64).astype(np.float32);a[~ok]=0.
            spec[...,j]=a
            for p in rr:
                if p['bands'][b]['valid']:
                    assert a[p['window']['point_y'],p['window']['point_x']]==np.float32(p['bands'][b]['TOA'])
        rgb=np.zeros((1020,1020,3),np.float32);rgb[:h,:wid]=spec*np.float32(255.)
        sample=pipeline(dict(img=rgb[...,::-1].copy(),img_shape=(1020,1020),ori_shape=(1020,1020),img_path=w['product_id']))
        assert tuple(sample['inputs'].shape)==(3,512,512)
        counter+=1;assert counter<=531
        stem=f'{counter:04d}';started_window=time.monotonic()
        write(dest/(stem+'_STARTED.json'),dict(window=k,repeat=repeat,forward_attempt=counter,utc=now(),
              RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()),True)
        with torch.no_grad():p=model.test_step(pseudo_collate([sample]))[0]
        assert tuple(p.seg_logits.data.shape)==(3,1020,1020)
        prob=p.seg_logits.data.softmax(0).permute(1,2,0).cpu().numpy().astype(np.float32)
        assert np.isfinite(prob).all() and np.allclose(prob.sum(-1),1,rtol=0,atol=3e-7)
        out=[]
        for q in rr:
            value=float(prob[q['window']['point_y'],q['window']['point_x'],2]);assert 0<=value<=1
            out.append(dict(point_id=q['point_id'],product_id=q['product_id'],row20=q['row20'],col20=q['col20'],Source=value))
        write(dest/(stem+'_points.json'),dict(window=k,repeat=repeat,rows=out),True)
        receipt=dict(window=k,repeat=repeat,attempt=counter,point_count=len(out),
            point_file=stem+'_points.json',point_file_sha256=sha(dest/(stem+'_points.json')),
            seconds=time.monotonic()-started_window,RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        write(dest/(stem+'_receipt.json'),receipt,True);receipts.append(receipt)
        print('Source window',counter,531,k,'repeat',repeat,flush=True)
        return out
    baseline={}
    for w in windows:
        out=one(w);all_scores.extend(out)
        if key(w) in read(RESULT/'task_lock.json')['repeats']:baseline[key(w)]=out
    checks=[]
    for w in repeats(windows):
        out=one(w,True);old=baseline[key(w)];assert [r['point_id'] for r in out]==[r['point_id'] for r in old]
        err=max(abs(a['Source']-b['Source']) for a,b in zip(out,old));assert err<=1e-5,('repeat failure',key(w),err)
        checks.append(dict(window=key(w),max_abs_point_probability_difference=err,tolerance=1e-5,passed=True))
    assert counter==531 and original.state_sha(model)==before
    assert len(all_scores)==len({r['point_id'] for r in all_scores})==9500
    write(ROOT/'source_points_no_labels.json',dict(rows=all_scores),True)
    write(ROOT/'forward_receipts.json',dict(rows=receipts,unique_windows=529,repeat_windows=2,actual_forwards=counter,
        repeats=checks,Source_state_before_after_equal=True,Source_state_sha256=before,
        source_points_sha256=sha(ROOT/'source_points_no_labels.json'),
        seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),True)

if __name__=='__main__':
    with rasterio.Env(GDAL_CACHEMAX=128*1024*1024):main()

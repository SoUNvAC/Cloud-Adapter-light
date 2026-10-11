"""Explicitly authorized disk-failure recovery; fixed inference, no fitting."""
import hashlib,json,os,shutil,time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import run_phase89 as p

OLD=p.ROOT
NEW=p.SHARED/'phase89_recovered_20261011'
AUDIT=p.SHARED/'phase89_resume_audit_20261011.json'
PAYLOAD=p.SHARED/'phase89_probability_payload_audit_20261011.json'

def durable_json(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())

def durable_array(path,value):
    with path.open('xb') as f:
        np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())

def initialize():
    p.locked();a=p.read(AUDIT);q=p.read(PAYLOAD)
    assert a['summary']['objects']==89 and all(x['matches'] for x in a['objects'])
    assert a['summary']['prepared_files']==54 and all(x['matches'] for x in a['prepared'])
    assert len(a['blocks'])==54 and len(a['missing_blocks'])==482
    assert q['valid_payloads']==54 and q['invalid_payloads']==112
    valid={(r['product'],r['key']):r for r in q['rows'] if r['payload_valid']}
    assert set(valid)=={(r['product'],r['key']) for r in a['blocks']}
    assert not NEW.exists(),'Preserve recovery attempt; no automatic retry'
    # Exactly the existing prepared arrays are linked, never duplicated or modified.
    prep=p.read(OLD/'prepared/index.json')
    for r in prep['rows']:
        for f in r['files'].values():assert p.digest(OLD/f['path'])==f['sha256']
    for r in a['blocks']:
        source=OLD/'scores_Source'/r['product']/(r['key']+'.npy')
        assert p.digest(source)==r['sha256']==valid[(r['product'],r['key'])]['sha256']
        assert p.digest(source.with_name(r['key']+'_forward_receipt.json'))==r['receipt_sha256']
    NEW.mkdir();(NEW/'delivery').mkdir();(NEW/'prepared').mkdir()
    shutil.copyfile(OLD/'input_scale_threshold_reference_lock.json',NEW/'input_scale_threshold_reference_lock.json')
    shutil.copyfile(OLD/'prepared/index.json',NEW/'prepared/index.json')
    for r in prep['rows']:
        for f in r['files'].values():
            dest=NEW/f['path'];dest.parent.mkdir(parents=True,exist_ok=True);os.link(OLD/f['path'],dest)
    for f in (OLD/'delivery').iterdir():
        assert f.is_file() and f.stat().st_size<10_000_000
        shutil.copyfile(f,NEW/'delivery'/f.name)
    plan=dict(created_utc=p.now(),original_root=str(OLD),recovery_root=str(NEW),
        authorization='User explicitly approved 112 damaged-block repeat forwards and 482 missing-output forwards; preserve failure files.',
        audit_sha256=p.digest(AUDIT),payload_audit_sha256=p.digest(PAYLOAD),
        helper_sha256=p.digest(__file__),reused=a['blocks'],
        damaged=[r for r in q['rows'] if not r['payload_valid']],missing=a['missing_blocks'],
        reused_count=54,damaged_repeat_count=112,missing_output_count=482,new_forward_count=594,
        historical_forward_total=None,historical_count_reason='Crash damaged receipts/logs; exact historical attempts unknown.',
        reused_output_precrash_SHA_certified=False,prepared_hardlinks_read_only_by_runner=True,
        host_free_GB_user_report=16,estimated_additional_GB=8,no_cleanup_no_training_no_fit_no_tuning=True)
    durable_json(NEW/'recovery_lock.json',plan)
    p.ROOT=NEW;p.locked()
    return plan

def infer(name,plan):
    import torch
    from mmengine.dataset import pseudo_collate
    assert p.digest(p.CK[name])==(p.SOURCE_SHA if name=='Source' else p.ADAPTED_SHA)
    root=NEW/('scores_'+name);root.mkdir();model,pipeline=p.setup(name);before=p.state_sha(model)
    reused={(r['product'],r['key']):r for r in plan['reused'] if r['model']==name}
    allowed={(r['product'],r['key']) for r in plan['missing'] if r['model']==name}
    if name=='Source':allowed|={(r['product'],r['key']) for r in plan['damaged']}
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();rows=[];count=0
    for r in p.read(NEW/'prepared/index.json')['rows']:
        product=r['product'];d=root/product;d.mkdir();spec=p.array(r,'spectra20');h,w=r['shape60']
        probs=np.empty((h,w,3),np.float32);blocks=[]
        for y in range(0,h,340):
            for x in range(0,w,340):
                if datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()>='2026-10-20':raise RuntimeError('Budget deadline reached')
                hh=min(340,h-y);ww=min(340,w-x);key=f'{y:04d}_{x:04d}';loc=(product,key)
                rgb=np.zeros((1020,1020,3),np.float32)
                rgb[:hh*3,:ww*3]=np.asarray(spec[y*3:(y+hh)*3,x*3:(x+ww)*3,[2,1,0]])*np.float32(255.)
                assert np.isfinite(rgb).all();input_sha=hashlib.sha256(rgb.tobytes()).hexdigest();t=time.monotonic()
                if loc in reused:
                    old=OLD/('scores_'+name)/product/(key+'.npy');receipt=old.with_name(key+'_forward_receipt.json')
                    assert p.digest(old)==reused[loc]['sha256'] and p.digest(receipt)==reused[loc]['receipt_sha256']
                    assert p.read(receipt)['input_rgb_sha256']==input_sha
                    os.link(old,d/(key+'.npy'));shutil.copyfile(receipt,d/(key+'_forward_receipt.json'))
                    a=np.load(old,allow_pickle=False);origin='reused_structural_input_verified_not_precrash_SHA_certified'
                else:
                    assert loc in allowed,(name,loc,'not authorized')
                    durable_json(d/(key+'_forward_receipt.json'),dict(started_utc=p.now(),y60=y,x60=x,true_shape20=[hh*3,ww*3],input_rgb_sha256=input_sha))
                    sample=pipeline(dict(img=rgb[...,::-1].copy(),img_shape=(1020,1020),ori_shape=(1020,1020),img_path=product))
                    assert tuple(sample['inputs'].shape)==(3,512,512)
                    with torch.no_grad():out=model.test_step(pseudo_collate([sample]))[0]
                    assert tuple(out.seg_logits.data.shape)==(3,1020,1020)
                    assert torch.equal(out.pred_sem_seg.data[0],out.seg_logits.data.argmax(0))
                    a=out.seg_logits.data.softmax(0).permute(1,2,0).cpu().numpy().astype(np.float32)[:hh*3,:ww*3].copy()
                    assert np.isfinite(a).all() and (a>=0).all() and (a<=1).all() and np.allclose(a.sum(-1),1,rtol=0,atol=3e-7)
                    durable_array(d/(key+'.npy'),a);count+=1;origin='authorized_recovery_forward'
                probs[y:y+hh,x:x+ww]=p.area(a,3).astype(np.float32)
                block=dict(y60=y,x60=x,h60=hh,w60=ww,path=str((d/(key+'.npy')).relative_to(NEW)),sha256=p.digest(d/(key+'.npy')),
                    seconds=time.monotonic()-t,padding20=int(1020**2-hh*ww*9),seam_support='nonoverlap; no blend',input_shape=[1020,1020],network_shape=[512,512],origin=origin)
                durable_json(d/(key+'_completed.json'),dict(**block,input_rgb_sha256=input_sha,completed_utc=p.now()))
                blocks.append(block);print('recovery',name,product,key,origin,'new_forwards',count,flush=True)
        assert len(blocks)==36;durable_array(d/'prob60.npy',probs)
        rows.append(dict(product=product,group_id=r['group_id'],blocks=blocks,prob60_path=str((d/'prob60.npy').relative_to(NEW)),prob60_sha256=p.digest(d/'prob60.npy')))
    after=p.state_sha(model);assert before==after and count==(270 if name=='Source' else 324)
    durable_json(root/'index.json',dict(lock_sha256=p.digest(NEW/'input_scale_threshold_reference_lock.json'),checkpoint_sha256=p.digest(p.CK[name]),
        before_state_sha256=before,after_state_sha256=after,forward_count=count,reused_block_count=len(reused),unique_block_count=324,
        historical_forward_count=None,elapsed_seconds=time.monotonic()-start,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),rows=rows))
    del model;torch.cuda.empty_cache()

def main():
    plan=initialize()
    for name in ['Source','MsRE']:infer(name,plan)
    p.shallow();p.analyze();p.verify()
    import finalize_phase89 as final
    original_write=p.write;original_summary=final.summarize
    def honest_write(path,value):
        if path.name=='COMPLETED.json':
            value.pop('network_forwards',None)
            value.update(unique_network_block_model_pairs=648,new_recovery_forwards=594,reused_blocks=54,
                authorized_damaged_block_repeats=112,historical_forward_total=None,recovery_lock_sha256=p.digest(NEW/'recovery_lock.json'))
        if path.name=='input_and_runtime_index.json':value.update(recovery=plan,source_attempt='run02 disk failure; authorized recovery preserving old files')
        original_write(path,value)
    def honest_summary(path):
        original_summary(path)
        with (path/'README.md').open('a',encoding='utf-8') as f:
            f.write('\n\n恢复说明：磁盘故障后复用54块，授权补算112块损坏输出并继续482块无输出任务，本次新增594次前向。历史总前向次数因日志损坏无法确定。复用块通过恢复时结构与输入身份检查，但缺少故障前输出SHA封存，不能宣称故障前SHA已核实。准备数据硬链接复用，故障原目录保留；无训练、拟合、调阈值。\n')
    p.write=honest_write;final.summarize=honest_summary;final.main()

if __name__=='__main__':main()

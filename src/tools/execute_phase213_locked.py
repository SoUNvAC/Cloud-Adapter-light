"""Phase213: pre-test source recheck, immutable source-only selection, fixed test."""
from pathlib import Path
import argparse, collections, csv, hashlib, json, os, resource, sys, time, traceback

REPO=Path('/home/scv/Cloud-Adapter-light'); SRC=REPO/'src'
BASE=Path('/home/scv/shared/phase212_retrain_20261020_run02')
ROOT=Path('/home/scv/shared/phase213_20261021_run04'); OUT=ROOT/'evaluation'
SOURCE=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'
SOURCE_SHA='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
ENDPOINTS={'l8':('best_mIoU_iter_3500.pth','cad66c102421230d1749b8c2e3bcfed36a47591a380484ecb8b000abf6114c31'),'sparcs':('best_mIoU_iter_3000.pth','98a2c1844463b8dbfaaca87354e84006a72059b16e3199636c5a445f45956c8e')}
MANIFESTS={'l8':(SRC/'work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv','885c2d7f61cae23409a7b1ccbe65739c302fc1e59b089aab0cc05badf674872b'),'sparcs':(SRC/'work_dirs/phase64_sparcs_protocol_verified_v2/scene_disjoint_manifest.csv','7fd78c026d7ee76e3fc8e3dc6f775ddeb86312242650d6ea085502622d05738c')}
ALPHAS=[0,.25,.5,.75,1]; CALLS=0; START=time.monotonic(); LEDGER=None; HOST_FREE=None

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''): h.update(b)
    return h.hexdigest()

def save(name,value):
    p=OUT/name; tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)); tmp.replace(p)

def main():
    global CALLS,LEDGER
    os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1'
    import numpy as np, torch
    from export_phase12_onnx import build_wrapper
    from validate_phase12_onnxruntime import load_rgb_image
    from evaluate_phase64_source_only_parent import paired_directory_records,manifest_records,load_parent_target,SOURCE_TO_PARENT,L8_TO_PARENT,SPARCS_TO_PARENT
    OUT.mkdir(exist_ok=False); LEDGER=(OUT/'CALL_LEDGER.jsonl').open('x',buffering=1)
    def status(stage,**kw):
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        gpu=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        assert rss<=16*1024**3 and gpu<=20*1024**3
        assert time.monotonic()-START<10800
        save('RUN_STATUS.json',dict(stage=stage,calls=CALLS,expected_calls=11420,elapsed_seconds=time.monotonic()-START,RSS_peak_bytes=rss,GPU_peak_allocated_bytes=gpu,**kw))
    status('preflight')
    lock_path=ROOT/'code/ALPHA_SELECTION_LOCK.json'; lock=json.loads(lock_path.read_text())
    assert sha(lock_path)=='53dc42499c8c2f1af238804cf0e25b8328598f64146f81d2c0a536de544516c9'
    assert lock['test_read_before_lock'] is False and sha(SOURCE)==SOURCE_SHA
    maps={'source':SOURCE_TO_PARENT,'l8':L8_TO_PARENT,'sparcs':SPARCS_TO_PARENT}
    rows={}; metadata={}
    for t,(p,expected) in MANIFESTS.items():
        assert sha(p)==expected
        allrows=list(csv.DictReader(p.open())); rows[t]=manifest_records(p,'target_test')
        test_scenes={r['scene'] for r in rows[t]}; other_scenes={r['scene'] for r in allrows if r['new_split'] in ['target_train','target_val']}
        assert not test_scenes&other_scenes
        for r in rows[t]:
            for key in ['image_path','mask_path']:
                q=Path(r[key]);q=q if q.is_absolute() else SRC/q;assert q.is_file();r[key]=str(q)
        metadata[t]={'manifest_sha256':expected,'inputs':len(rows[t]),'scenes':len(test_scenes),'scene_disjoint_from_original_train_val':True}
    assert [len(rows[t]) for t in ['l8','sparcs']]==[2167,10]
    source_root=SRC/'data/cloudsen12_high_l1c'
    if not source_root.is_dir():source_root=REPO/'data/cloudsen12_high_l1c'
    rows['source']=paired_directory_records(source_root,'val')
    bindings={r['png_name']:r for r in csv.DictReader((BASE/'eval_code/SOURCE_ROI_BINDINGS.csv').open())}
    assert len(rows['source'])==535 and len({r['roi_id'] for r in bindings.values()})==107
    for r in rows['source']:
        assert sha(r['image_path'])==bindings[r['name']]['image_sha256'];r['scene']=bindings[r['name']]['roi_id']
    wrapper,_=build_wrapper(argparse.Namespace(config='configs/protocol/phase64_source_parent_rgb.py',checkpoint=str(SOURCE),precision='fp32',active_block_indices=None))
    wrapper.eval()
    for p in wrapper.parameters():p.requires_grad_(False)
    full=dict(list(wrapper.segmentor.named_parameters())+list(wrapper.segmentor.named_buffers()))
    source_state=torch.load(SOURCE,map_location='cpu')['state_dict']
    assert not set(source_state)-set(full)
    assert set(full)-set(source_state)=={'data_preprocessor.mean','data_preprocessor.std'}
    assert all(torch.equal(full[n].detach().cpu(),v) for n,v in source_state.items())
    assert torch.equal(full['data_preprocessor.mean'].detach().cpu().flatten(),torch.tensor([123.675,116.28,103.53]))
    assert torch.equal(full['data_preprocessor.std'].detach().cpu().flatten(),torch.tensor([58.395,57.12,57.375]))
    source_head={n:v.detach().cpu().clone() for n,v in full.items() if n.startswith('decode_head.')}
    frozen={n:v.detach().cpu().clone() for n,v in full.items() if not n.startswith('decode_head.')}
    params={n for n,_ in wrapper.segmentor.named_parameters() if n.startswith('decode_head.')}
    buffers=set(source_head)-params;heads={};proof={};hashes=json.loads((ROOT/'code/HEAD_HASHES.json').read_text())
    assert hashes==json.loads((BASE/'interpolation/HEAD_HASHES.json').read_text())
    for t,(fn,expected) in ENDPOINTS.items():
        p=BASE/t/fn;assert sha(p)==expected;target=torch.load(p,map_location='cpu')['state_dict'];assert set(target)==set(source_head)
        assert all(torch.equal(target[n],source_head[n]) for n in buffers)
        heads[t]={0:source_head,1:target};proof[t]={'0':{'sha256':SOURCE_SHA,'path':str(SOURCE)},'1':{'sha256':expected,'path':str(p)}}
        for a in [.25,.5,.75]:
            p=BASE/'interpolation/heads'/f'{t}_a{str(a).replace(".","p")}.pth';h=sha(p);assert h==hashes[p.name]
            st=torch.load(p,map_location='cpu')['state_dict'];assert set(st)==set(source_head)
            for n,s in source_head.items():
                v=(s.double()+a*(target[n].double()-s.double())).to(s.dtype) if n in params else s
                assert st[n].shape==s.shape and st[n].dtype==s.dtype and torch.equal(st[n],v)
            heads[t][a]=st;proof[t][str(a)]={'sha256':h,'path':str(p),'formula_exact':True}
    save('PARAMETER_ACCEPTANCE.json',{'head_parameters':len(params),'head_buffers':len(buffers),'nonhead_tensors':len(frozen),'full_Source_exact_tensors':len(source_state),'runtime_nonpersistent_normalization_buffers':['data_preprocessor.mean','data_preprocessor.std'],'buffers_equal':True,'all_parameters_frozen':True,'heads':proof})
    def loadhead(h):
        state=wrapper.segmentor.state_dict();state.update(h);wrapper.segmentor.load_state_dict(state,strict=True)
        now=dict(list(wrapper.segmentor.named_parameters())+list(wrapper.segmentor.named_buffers()));assert all(torch.equal(now[n].detach().cpu(),v) for n,v in frozen.items())
    def evaluate(pool,a,target):
        global CALLS
        loadhead(heads[target][a]);total=np.zeros((3,3),np.int64);groups=collections.defaultdict(lambda:np.zeros((3,3),np.int64));per=[];ph=hashlib.sha256()
        with torch.inference_mode():
            for i,r in enumerate(rows[pool]):
                assert CALLS<12000 and time.monotonic()-START<10800
                x=torch.from_numpy(load_rgb_image(Path(r['image_path']),512)).cuda();CALLS+=1
                LEDGER.write(json.dumps({'call':CALLS,'pool':pool,'target':target,'alpha':a,'input':i,'elapsed_seconds':time.monotonic()-START})+'\n')
                s=wrapper(x).float().clamp_min(0);s=s/s.sum(1,keepdim=True).clamp_min(1e-12);assert torch.isfinite(s).all()
                pred=s.argmax(1)[0].cpu().numpy().astype(np.uint8);ph.update(pred.tobytes())
                y=load_parent_target(Path(r['mask_path']),maps[pool],pred.shape);valid=y!=255
                cm=np.bincount(3*y[valid]+pred[valid],minlength=9).reshape(3,3);total+=cm;groups[r['scene']]+=cm
                per.append({'name':r['name'],'group':r['scene'],'confusion':cm.tolist(),'ignore_pixels':int((~valid).sum()),'image_sha256':sha(r['image_path']) if a==0 else None,'mask_sha256':sha(r['mask_path']) if a==0 else None})
                if i%100==0 or i+1==len(rows[pool]):status('evaluating',pool=pool,target=target,alpha=a,completed_inputs=i+1,total_inputs=len(rows[pool]))
        now=dict(list(wrapper.segmentor.named_parameters())+list(wrapper.segmentor.named_buffers()));assert all(torch.equal(now[n].detach().cpu(),v) for n,v in frozen.items())
        result={'pool':pool,'target':target,'alpha':a,'confusion':total.tolist(),'per_group':{g:c.tolist() for g,c in sorted(groups.items())},'per_input':per,'prediction_sha256':ph.hexdigest(),'inputs':len(per),'groups':len(groups)}
        save(f'{pool}_a{str(a).replace(".","p")}.json',result);return result
    source=evaluate('source',0,'l8');old=json.loads((BASE/'interpolation/Source__source.json').read_text())
    assert source['confusion']==old['confusion'] and source['prediction_sha256']==old['prediction_sha256'] and source['per_group']==old['per_group']
    def miou(cm):
        x=np.asarray(cm,dtype=float);tp=np.diag(x);u=x.sum(0)+x.sum(1)-tp;return float(100*np.nanmean(np.divide(tp,u,out=np.full(3,np.nan),where=u>0)))
    base_score=miou(source['confusion']);candidates={};cache_hashes={};pinned_cache=json.loads((ROOT/'code/SOURCE_CACHE_HASHES.json').read_text())
    for t in ['l8','sparcs']:
        candidates[t]={}
        for a in ALPHAS:
            p=BASE/'interpolation'/('Source__source.json' if a==0 else f'{t}_a{str(a).replace(".","p")}__source.json')
            assert sha(p)==pinned_cache[p.name]
            v=json.loads(p.read_text());assert v['inputs']==535 and v['groups']==107
            candidates[t][str(a)]=miou(v['confusion']);cache_hashes[p.name]=sha(p)
        for r in lock['rows']:
            if r['target']==t:
                feasible=[a for a in ALPHAS if base_score-candidates[t][str(a)]<=r['budget_pp']]
                assert max(feasible)==r['chosen_alpha']
    # This durable seal precedes any target-test image or label open.
    save('FINAL_ADMISSION_LOCK.json',{'epoch':time.time(),'original_lock_sha256':sha(lock_path),'source_recheck_exact':True,'source_prediction_sha256':source['prediction_sha256'],'source_confusion':source['confusion'],'source_candidates_mIoU':candidates,'source_cache_sha256':cache_hashes,'heads':proof,'target_metadata':metadata,'mapping':{k:v.tolist() for k,v in maps.items()},'code_sha256':sha(__file__),'calls_before_target':CALLS,'expected_total_calls':11420,'test_opened_before_seal':False,'selection_unchanged':True,'prior_failed_calls':0})
    for t in ['l8','sparcs']:
        for a in ALPHAS:evaluate(t,a,t)
    assert CALLS==11420
    status('complete');persist=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file());assert persist<4*1024**3
    save('RUN_COMPLETE.json',{'calls':CALLS,'expected_calls':11420,'elapsed_seconds':time.monotonic()-START,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'RSS_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'persistent_bytes':persist,'fits':0,'host_disk_preflight':json.loads((ROOT/'HOST_DISK.json').read_text()),'failed_prior_calls':0})
    (ROOT/'EXIT_CODE').write_text('0\n')

if __name__=='__main__':
    try:main()
    except BaseException:
        OUT.mkdir(exist_ok=True);save('FAILURE.json',{'calls':CALLS,'elapsed_seconds':time.monotonic()-START,'traceback':traceback.format_exc()});(ROOT/'EXIT_CODE').write_text('1\n');raise
    finally:
        if LEDGER:LEDGER.close()

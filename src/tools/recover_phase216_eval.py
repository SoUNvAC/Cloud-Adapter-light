"""Phase216 evaluation-only recovery; no optimizer, backward or new fitting."""
from pathlib import Path
import argparse,collections,copy,csv,hashlib,json,os,resource,shutil,sys,threading,time,traceback
from execute_phase216 import ROOT,SRC,BASE,SOURCE,SOURCE_SHA,ALPHAS,sha,js,dump as original_dump,hostfree
def main():
    os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1'
    import numpy as np,torch
    from export_phase12_onnx import build_wrapper
    from validate_phase12_onnxruntime import load_rgb_image
    from evaluate_phase64_source_only_parent import manifest_records,paired_directory_records,load_parent_target,SOURCE_TO_PARENT,SPARCS_TO_PARENT
    assert (ROOT/'EXIT_CODE').read_text().strip()=='1'
    assert not Path('/proc/'+(ROOT/'PID').read_text().strip()).exists()
    assert not (ROOT/'EVAL_RECOVERY_STARTED.json').exists()
    diagnosis=js(ROOT/'EVAL_NUMERICAL_DIAGNOSIS.json');assert diagnosis['probes'][1]['results']['legacy_normalized']['matches_historical']
    assert sha(SOURCE)==SOURCE_SHA
    oldstatus=js(ROOT/'RUN_STATUS.json');old_seconds=oldstatus['GPU_stage_seconds']
    diag_upper=(ROOT/'EVAL_NUMERICAL_DIAGNOSIS.json').stat().st_mtime-(ROOT/'code/diagnose_phase216_eval.py').stat().st_mtime
    assert 0<diag_upper<120
    previous=[json.loads(x) for x in (ROOT/'CALL_LEDGER.jsonl').read_text().splitlines()]
    assert len(previous)==20542 and [r['ordinal'] for r in previous]==list(range(1,20543))
    counts=collections.Counter(r['kind'] for r in previous);prior_evaluation_calls=counts['evaluation_full_segmentor']
    assert prior_evaluation_calls==535 and counts['diagnostic_Source_full_segmentor']==2
    failure=ROOT/'failures/first_evaluation';failure.mkdir(parents=True,exist_ok=False)
    for n in ['RUN_STATUS.json','EXIT_CODE','FAILURE.txt','FAILURE_COST.json','PID']:shutil.copy2(ROOT/n,failure/n)
    assert (ROOT/'evaluation').resolve().parent==ROOT.resolve()
    (ROOT/'evaluation').rename(failure/'evaluation')
    (ROOT/'EXIT_CODE').rename(failure/'ACTIVE_EXIT_CODE_BEFORE_RECOVERY')
    (ROOT/'PID').write_text(str(os.getpid()))
    start=time.monotonic();GPU_START=start;phase='evaluation_recovery_build';active_arm=None;step=0
    done=threading.Event();ledger=(ROOT/'CALL_LEDGER.jsonl').open('a',buffering=1)
    oldresources=js(ROOT/'RESOURCE_SAMPLES.json');resource_samples=[];free=hostfree()
    def dump(p,x):
        p=Path(p)
        if p.name=='EVAL_INPUT_LOCK.json' and p.exists():assert js(p)==x;return
        original_dump(p,x)
    dump(ROOT/'EVAL_RECOVERY_STARTED.json',{'epoch':time.time(),'PID':os.getpid(),'prior_units':sum(counts.values()),'additional_planned_units':4995,'final_planned_units':25537,'prior_GPU_stage_seconds':old_seconds,'diagnostic_process_upper_seconds':diag_upper,'fit_restarts':0,'evaluation_change':'restore historical torch/cudnn deterministic=False and legacy per-pixel sum-normalized argmax','no_sealed_head_overwritten':True,'recovery_code_SHA':sha(Path(__file__))})
    def status():dump(ROOT/'RUN_STATUS.json',{'stage':phase,'arm':active_arm,'step':step,'PID':os.getpid(),'counts':dict(counts),'total_units':sum(counts.values()),'GPU_stage_seconds':old_seconds+diag_upper+time.monotonic()-start,'evaluation_recovery':True})
    def count(kind):
        assert sum(counts.values())<30000 and old_seconds+diag_upper+time.monotonic()-start<14400
        counts[kind]+=1;ledger.write(json.dumps({'ordinal':sum(counts.values()),'kind':kind,'stage':phase,'arm':active_arm,'step':step,'recovery_seconds':time.monotonic()-start})+'\n')
    def guard():
        last_host=time.monotonic()
        while not done.wait(15):
            try:
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;gpu=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0;reserved=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
                size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.name.endswith('.tmp'))
                assert rss<=16*1024**3 and max(gpu,reserved)<=20*1024**3 and size<4_000_000_000
                assert old_seconds+diag_upper+time.monotonic()-start<14400
                if time.monotonic()-last_host>180:hostfree();last_host=time.monotonic()
                resource_samples.append({'epoch':time.time(),'rss_bytes':rss,'GPU_allocated_peak':gpu,'GPU_reserved_peak':reserved,'persistent_bytes':size});dump(ROOT/'RECOVERY_RESOURCE_SAMPLES.json',resource_samples);status()
            except BaseException:
                (ROOT/'RECOVERY_RESOURCE_FAILURE.txt').write_text(traceback.format_exc());(ROOT/'EXIT_CODE').write_text('71\n');os._exit(71)
    # Replay stays in its original training mode; only evaluation process flags change.
    torch.set_num_threads(4);torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False
    threading.Thread(target=guard,daemon=True).start()
    request=js(ROOT/'SOURCE_POOL_REQUEST.json')
    try:
        wrapper,_=build_wrapper(argparse.Namespace(config='configs/protocol/phase64_source_parent_rgb.py',checkpoint=str(SOURCE),precision='fp32',active_block_indices=None));model=wrapper.segmentor;model.eval()
        for p in model.parameters():p.requires_grad=False
        initial=torch.load(ROOT/'heads/initial.pth',map_location='cpu')['state_dict']
        assert all(torch.equal(v,model.decode_head.state_dict()[n].detach().cpu()) for n,v in initial.items())
        def tensors():return dict(list(model.named_parameters())+list(model.named_buffers()))
        frozen={n:v.detach().cpu().clone() for n,v in tensors().items() if not n.startswith('decode_head.')}
        def digest(d):
            h=hashlib.sha256()
            for n,t in sorted(d.items()):h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        assert digest(frozen)==js(ROOT/'GATE_ACCEPTED.json')['nonhead_digest']
        teacher=copy.deepcopy(model.decode_head).eval();teacher_digest=digest(teacher.state_dict())
        for p in teacher.parameters():p.requires_grad=False
        def frozen_check():
            now=tensors();assert all(torch.equal(v,now[n].detach().cpu()) for n,v in frozen.items());assert digest(teacher.state_dict())==teacher_digest and not model.backbone.training
        endpoints={}
        for arm in ['C','D']:
            p=ROOT/f'heads/{arm}_iter4000.pth';assert sha(p)==js(ROOT/f'{arm}_TRAIN_COMPLETE.json')['head_SHA'];endpoints[arm]=torch.load(p,map_location='cpu')['state_dict']
        active_arm=None;phase='evaluation_preflight';model.eval();wrapper.eval()
        for p in model.parameters():p.requires_grad=False
        basegroups={x['png_name']:x for x in csv.DictReader((BASE/'eval_code/SOURCE_ROI_BINDINGS.csv').open())}
        manifest=SRC/'work_dirs/phase64_sparcs_protocol_verified_v2/scene_disjoint_manifest.csv'
        assert sha(manifest)=='7fd78c026d7ee76e3fc8e3dc6f775ddeb86312242650d6ea085502622d05738c'
        pools={'source':paired_directory_records(SRC/'data/cloudsen12_high_l1c','val'),'val':manifest_records(manifest,'target_val'),'test':manifest_records(manifest,'target_test')}
        for r in pools['source']:
            assert sha(r['image_path'])==basegroups[r['name']]['image_sha256'];r['scene']=basegroups[r['name']]['roi_id']
        assert [len(pools[s]) for s in ['source','val','test']]==[535,10,10]
        # Input hashes only here, no target reference decoding or target metric join.
        support={s:[dict(r,image_sha256=sha(r['image_path']),mask_sha256=sha(r['mask_path'])) for r in rs] for s,rs in pools.items()}
        dump(ROOT/'EVAL_INPUT_LOCK.json',support)
        oldsupport=js(BASE/'interpolation/INPUT_SUPPORT_LOCK.json')['pools']
        for s,old in [('source','source'),('val','sparcs')]:
            a={r['name']:(r['image_sha256'],r['mask_sha256']) for r in support[s]};b={r['name']:(r['image_sha256'],r['mask_sha256']) for r in oldsupport[old]};assert a==b
        evalout=ROOT/'evaluation';evalout.mkdir()
        maps={'source':SOURCE_TO_PARENT,'val':SPARCS_TO_PARENT,'test':SPARCS_TO_PARENT}
        source_cache=js(BASE/'interpolation/Source__source.json')
        interpolants={}
        for arm in ['C','D']:
            for alpha in ALPHAS:
                head={n:(v.clone() if alpha==0 else endpoints[arm][n].clone() if alpha==1 else (v.double()+alpha*(endpoints[arm][n].double()-v.double())).to(v.dtype)) for n,v in initial.items()}
                assert all(torch.isfinite(t).all() for t in head.values());interpolants[arm,alpha]=head
                if alpha not in [0,1]:
                    sealed=torch.load(ROOT/f'heads/{arm}_a{alpha}.pth',map_location='cpu')['state_dict'];assert all(torch.equal(sealed[n],v) for n,v in head.items())
        def evaluate(arm,alpha,pool):
            nonlocal phase,step,active_arm
            phase='evaluate_'+pool;active_arm=arm;model.decode_head.load_state_dict(interpolants[arm,alpha]);model.eval();frozen_check()
            groups=collections.defaultdict(lambda:np.zeros((3,3),np.int64));total=np.zeros((3,3),np.int64);per=[];predsha=hashlib.sha256()
            with torch.inference_mode():
                for step,r in enumerate(pools[pool],1):
                    x=torch.from_numpy(load_rgb_image(Path(r['image_path']),512)).cuda();count('evaluation_full_segmentor')
                    s=wrapper(x).float().clamp_min(0);pred=(s/s.sum(1,keepdim=True).clamp_min(1e-12)).argmax(1)[0].cpu().numpy().astype(np.uint8);predsha.update(pred.tobytes())
                    y=load_parent_target(Path(r['mask_path']),maps[pool],pred.shape);valid=y!=255;c=np.bincount(3*y[valid]+pred[valid],minlength=9).reshape(3,3)
                    groups[r['scene']]+=c;total+=c;per.append({'name':r['name'],'scene':r['scene'],'confusion':c.tolist()})
            result={'arm':arm,'alpha':alpha,'pool':pool,'confusion':total.tolist(),'per_group':{g:c.tolist() for g,c in sorted(groups.items())},'per_input':per,'inputs':len(per),'prediction_SHA':predsha.hexdigest()}
            dump(evalout/f'{arm}_a{alpha}__{pool}.json',result);status();return result
        fresh_source=evaluate('C',0,'source')
        assert fresh_source['confusion']==source_cache['confusion'],'Source FP32 baseline mismatch'
        source_cache=fresh_source;dump(evalout/'Source__source.json',source_cache)
        for arm in ['C','D']:
            for alpha in ALPHAS[1:]:evaluate(arm,alpha,'source')
        def miou(c):
            c=np.asarray(c,dtype=np.float64);d=np.diag(c);u=c.sum(0)+c.sum(1)-d;return float(100*np.nanmean(np.divide(d,u,out=np.full(3,np.nan),where=u>0)))
        def groupdict(r):return r.get('per_group',r.get('per_scene'))
        sourcebase=groupdict(source_cache);assert len(sourcebase)==107
        source_metrics={};selection=[]
        for arm in ['C','D']:
            source_metrics[arm]={}
            for alpha in ALPHAS:
                r=source_cache if alpha==0 else js(evalout/f'{arm}_a{alpha}__source.json');g=groupdict(r)
                assert set(g)==set(sourcebase) and all(np.array_equal(np.asarray(g[k]).sum(1),np.asarray(sourcebase[k]).sum(1)) for k in g)
                losses=sorted((miou(sourcebase[k])-miou(g[k]) for k in g),reverse=True)
                source_metrics[arm][str(alpha)]={'pool_loss_pp':miou(source_cache['confusion'])-miou(r['confusion']),'T11_pp':float(np.mean(losses[:11])),'worst_ROI_loss_pp':losses[0],'ROI_loss_gt5':sum(x>5 for x in losses)}
            for budget in [1,3,5]:
                feasible=[a for a in ALPHAS if source_metrics[arm][str(a)]['pool_loss_pp']<=budget and source_metrics[arm][str(a)]['T11_pp']<=5]
                selection.append({'arm':arm,'budget_pp':budget,'alpha':max(feasible)})
        dump(ROOT/'SOURCE_SELECTION_LOCK.json',{'epoch':time.time(),'source_metrics':source_metrics,'rows':selection,'rule':'largest alpha with pool_loss<=B and signed reranked T11<=5; no tolerance','target_metrics_loaded_before_lock':False})
        # Only after the durable source-only lock, join/compute target validation and test.
        fresh_val=evaluate('C',0,'val');assert fresh_val['confusion']==js(BASE/'interpolation/Source__sparcs.json')['confusion'];dump(evalout/'Source__val.json',fresh_val)
        fresh_test=evaluate('C',0,'test');assert fresh_test['confusion']==js(Path('/home/scv/shared/phase213_20261021_run04/evaluation/sparcs_a0.json'))['confusion'];dump(evalout/'Source__test.json',fresh_test)
        for arm in ['C','D']:
            for alpha in ALPHAS[1:]:
                for pool in ['val','test']:evaluate(arm,alpha,pool)

        phase='complete';active_arm=None;step=0;frozen_check()
        assert counts['evaluation_full_segmentor']==5530 and counts['target_full_segmentor']==8002
        assert all(counts[k]==4001 for k in ['source_shared_backbone','source_teacher_head','source_student_head'])
        assert sum(counts.values())==25537
        dump(ROOT/'ACCESS_AUDIT.json',{'source_train_label_reads':0,'source_test_label_reads':0,'forbidden_attempts':[],'source_image_training_uses':4000,'source_gate_uses':1,'source_support':'all512x512 resized RGB; no padding or label filtering','official_test_images':'initial byte hashes only, no decode','evidence':'original training source loader and Python open guard; no system-wide syscall audit; diagnostic and recovery read source val annotations only','diagnostic_extra_Source_calls':2,'same_product_metadata_caveat':request['metadata_relation']})
        free=hostfree();done.set()
        dump(ROOT/'COMPLETE.json',{'counts':dict(counts),'total_units':sum(counts.values()),'elapsed_seconds':time.time()-js(ROOT/'EXECUTION_STARTED.json')['epoch'],'GPU_stage_seconds':old_seconds+diag_upper+time.monotonic()-start,'prior_failed_GPU_stage_seconds':old_seconds,'diagnostic_process_upper_seconds':diag_upper,'GPU_recovery_seconds':time.monotonic()-start,'GPU_peak_allocated_bytes':max(torch.cuda.max_memory_allocated(),max(x['GPU_allocated_peak'] for x in oldresources)),'GPU_peak_reserved_bytes':max(torch.cuda.max_memory_reserved(),max(x['GPU_reserved_peak'] for x in oldresources)),'CPU_peak_bytes':max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,max(x['rss_bytes'] for x in oldresources)),'prior_failed_peak_limitation':'prior CUDA cumulative peaks observed by15-second guard; final exit snapshot unavailable','host_volumes_final':free,'head_SHA':{p.name:sha(p) for p in (ROOT/'heads').glob('*.pth')},'source_selection_SHA':sha(ROOT/'SOURCE_SELECTION_LOCK.json'),'source_SHA_unchanged':sha(SOURCE)==SOURCE_SHA,'training_restarts':0,'failed_and_diagnostic_calls_included':537,'evaluation_runtime':'original deterministicFalse, legacy normalized argmax FP32'})
        status();(ROOT/'EXIT_CODE').write_text('0\n')
    except BaseException:
        done.set();(ROOT/'EVAL_RECOVERY_FAILURE.txt').write_text(traceback.format_exc());status();dump(ROOT/'EVAL_RECOVERY_FAILURE_COST.json',{'counts':dict(counts),'total_units':sum(counts.values()),'recovery_seconds':time.monotonic()-start});(ROOT/'EXIT_CODE').write_text('1\n');raise
    finally:ledger.close()
if __name__=='__main__':main()

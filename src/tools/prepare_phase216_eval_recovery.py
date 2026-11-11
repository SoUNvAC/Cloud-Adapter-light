"""Generate reviewable evaluation-only recovery from frozen original eval code."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[2]
PREFIX=r'''"""Phase216 evaluation-only recovery; no optimizer, backward or new fitting."""
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
'''
SUFFIX=r'''
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
'''
def main():
    original=ROOT/'src/tools/execute_phase216.py';text=original.read_text(encoding='utf-8')
    assert hashlib.sha256(original.read_bytes()).hexdigest()=='284ec89365f5dfccbd6082bc4e5ea232daaf539341b2d4e408023fa0856a1458'
    tail=text.split("        active_arm=None;phase='evaluation_preflight'",1)[1].split("        phase='complete';active_arm=None;step=0;frozen_check();assert",1)[0]
    tail="        active_arm=None;phase='evaluation_preflight'"+tail
    needle="if alpha not in [0,1]:torch.save({'state_dict':head,'alpha':alpha,'arm':arm,'Source_SHA':SOURCE_SHA,'formula':'CPU float64 then source dtype'},ROOT/f'heads/{arm}_a{alpha}.pth')"
    assert tail.count(needle)==1
    tail=tail.replace(needle,"if alpha not in [0,1]:\n                    sealed=torch.load(ROOT/f'heads/{arm}_a{alpha}.pth',map_location='cpu')['state_dict'];assert all(torch.equal(sealed[n],v) for n,v in head.items())")
    needle="pred=wrapper(x).argmax(1)[0].cpu().numpy().astype(np.uint8);predsha.update(pred.tobytes())"
    assert tail.count(needle)==1
    tail=tail.replace(needle,"s=wrapper(x).float().clamp_min(0);pred=(s/s.sum(1,keepdim=True).clamp_min(1e-12)).argmax(1)[0].cpu().numpy().astype(np.uint8);predsha.update(pred.tobytes())")
    code=PREFIX+tail+SUFFIX;ast.parse(code)
    assert '.backward(' not in code and 'build_optim_wrapper' not in code and '.train(' not in code
    out=ROOT/'src/tools/recover_phase216_eval.py';assert not out.exists();out.write_text(code,encoding='utf-8')
    print(json.dumps({'recovery':str(out),'SHA':hashlib.sha256(out.read_bytes()).hexdigest(),'no_training_code':True}))
if __name__=='__main__':main()

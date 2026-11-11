"""Bounded Phase216 only. Original target loss plus fixed source semantic KL."""
from pathlib import Path
import argparse, collections, copy, csv, hashlib, json, os, random, resource
import subprocess, sys, threading, time, traceback
ROOT=Path('/home/scv/shared/phase216_20261023')
SRC=Path('/home/scv/Cloud-Adapter-light/src')
BASE=Path('/home/scv/shared/phase212_retrain_20261020_run02')
SOURCE=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'
SOURCE_SHA='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
ALPHAS=[0,.25,.5,.75,1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def dump(p,x):
    p=Path(p);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False,default=str));t.replace(p)
def hostfree():
    raw=subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',"Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FreeSpace | ConvertTo-Json -Compress"],text=True)
    rows=json.loads(raw);assert min(r['FreeSpace'] for r in rows)>=4*1024**3;return rows
def main():
    os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')]
    os.environ.update(XFORMERS_DISABLED='1',PHASE64_TARGET='sparcs',PHASE64_METHOD='shared_parent',PHASE64_PARENT_CHECKPOINT=str(SOURCE))
    import numpy as np,torch
    import torch.nn.functional as F
    from mmengine.config import Config
    from mmengine.dataset import InfiniteSampler,pseudo_collate
    from mmengine.registry import init_default_scope,PARAM_SCHEDULERS
    from mmengine.optim import build_optim_wrapper
    from mmseg.registry import DATASETS
    import cloud_adapter.datasets,cloud_adapter.models
    from export_phase12_onnx import build_wrapper
    from validate_phase12_onnxruntime import load_rgb_image
    from evaluate_phase64_source_only_parent import manifest_records,paired_directory_records,load_parent_target,SOURCE_TO_PARENT,SPARCS_TO_PARENT
    from unittest.mock import patch
    assert not (ROOT/'EXECUTION_STARTED.json').exists(),'Do not repeat or erase an earlier attempt'
    assert js(ROOT/'SOURCE_IDENTITY_ACCEPTED.json')['overlap']=={'val':0,'test':0}
    assert sha(SOURCE)==SOURCE_SHA
    identity=js(Path('/home/scv/shared/phase213_20261021_run04/FINAL_COST_IDENTITY.json'))['remote_code_identity']
    assert all(sha(SRC/n)==h for n,h in identity.items())
    dump(ROOT/'RUNTIME_CODE_LOCK.json',{'historical_code_SHA':identity,'Source_SHA':SOURCE_SHA,'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'target_config_SHA':sha(SRC/'configs/protocol/phase64_target_parent_rgb.py')})
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=True)
    init_default_scope('mmseg')
    start=time.monotonic();counts=collections.Counter();phase='preparing';step=0;active_arm=None
    ledger=(ROOT/'CALL_LEDGER.jsonl').open('x',buffering=1)
    done=threading.Event();resource_samples=[];last_host=time.monotonic();free=hostfree();GPU_START=None
    dump(ROOT/'EXECUTION_STARTED.json',{'epoch':time.time(),'PID':os.getpid(),'planned_units':25000,'planned_training_updates':{'C':4000,'D':4000},'common_Source_evaluation':'555 fresh calls, compare historical confusion; no uncertain cache reuse','calls_cap':30000,'GPU_seconds_cap':14400,'host_volumes':free,'code_SHA':sha(Path(__file__))})
    def status():
        dump(ROOT/'RUN_STATUS.json',{'stage':phase,'arm':active_arm,'step':step,'PID':os.getpid(),'counts':dict(counts),'total_units':sum(counts.values()),'elapsed_seconds':time.monotonic()-start,'GPU_stage_seconds':0 if GPU_START is None else time.monotonic()-GPU_START})
    def guard():
        nonlocal last_host,free
        while not done.wait(15):
            try:
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
                gpu=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
                reserved=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
                size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.name.endswith('.tmp'))
                assert rss<=16*1024**3 and gpu<=20*1024**3 and reserved<=20*1024**3 and size<=4_000_000_000
                assert GPU_START is None or time.monotonic()-GPU_START<14400
                if time.monotonic()-last_host>180:free=hostfree();last_host=time.monotonic()
                resource_samples.append({'epoch':time.time(),'rss_bytes':rss,'GPU_allocated_peak':gpu,'GPU_reserved_peak':reserved,'persistent_bytes':size,'host_min_free':min(x['FreeSpace'] for x in free)})
                dump(ROOT/'RESOURCE_SAMPLES.json',resource_samples);status()
            except BaseException:
                (ROOT/'RESOURCE_FAILURE.txt').write_text(traceback.format_exc());(ROOT/'EXIT_CODE').write_text('71\n');os._exit(71)
    def count(kind):
        assert sum(counts.values())<30000 and (GPU_START is None or time.monotonic()-GPU_START<14400)
        counts[kind]+=1;ledger.write(json.dumps({'ordinal':sum(counts.values()),'kind':kind,'stage':phase,'arm':active_arm,'step':step,'seconds':time.monotonic()-start})+'\n')
    # Python audit hook denies every CloudSEN train/test annotation file open.
    denied=[];access=collections.Counter()
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=os.path.abspath(os.fsdecode(args[0])).replace('\\','/')
        if 'cloudsen12_high_l1c/ann_dir/train/' in p or 'cloudsen12_high_l1c/ann_dir/test/' in p:
            denied.append(p);raise RuntimeError('Forbidden source train/test label access')
        if 'cloudsen12_high_l1c/ann_dir/val/' in p:access['source_val_label_file_opens']+=1
    sys.addaudithook(audit)
    threading.Thread(target=guard,daemon=True).start()
    try:
        cfg=Config.fromfile('configs/protocol/phase64_target_parent_rgb.py')
        cfg.randomness.seed=21601
        assert cfg.train_cfg.max_iters==4000 and cfg.train_dataloader.batch_size==1
        dataset=DATASETS.build(cfg.train_dataloader.dataset);dataset.full_init();assert len(dataset)==60
        sampler=iter(InfiniteSampler(dataset,shuffle=True,seed=21601))
        request=js(ROOT/'SOURCE_POOL_REQUEST.json');selected=request['selected'];assert len(selected)==256
        def tensor_sha(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
        def augment(index,seed,capture=False):
            oldnp=np.random.get_state();oldpy=random.getstate();np.random.seed(seed);random.seed(seed);draws=[]
            def wrapped(name,fn):
                def call(*a,**kw):
                    v=fn(*a,**kw);draws.append({'function':name,'args':a,'kwargs':kw,'value':v.tolist() if hasattr(v,'tolist') else v});return v
                return call
            try:
                with patch.object(np.random,'randint',wrapped('randint',np.random.randint)),patch.object(np.random,'uniform',wrapped('uniform',np.random.uniform)),patch.object(np.random,'choice',wrapped('choice',np.random.choice)):
                    data=dataset.pipeline(dataset.get_data_info(index))
                proof={'image_SHA':tensor_sha(data['inputs']),'mask_SHA':tensor_sha(data['data_samples'].gt_sem_seg.data),'draws':draws,'flip':data['data_samples'].metainfo.get('flip'),'flip_direction':data['data_samples'].metainfo.get('flip_direction')}
                return data,proof
            finally:np.random.set_state(oldnp);random.setstate(oldpy)
        schedule=[]
        for i in range(4000):
            index=next(sampler);info=dataset.get_data_info(index);seed=216010000+i
            _,proof=augment(index,seed,True)
            schedule.append(dict(iteration=i+1,index=index,data_seed=seed,image_path=info['img_path'],mask_path=info['seg_map_path'],source_sample_id=selected[i%256]['sample_id'],source_png_name=selected[i%256]['png_name'],**proof))
            if i%500==0:step=i;status()
        dump(ROOT/'TARGET_SOURCE_SCHEDULE.json',schedule)
        dump(ROOT/'TRAINING_CONFIG_LOCK.json',{'seed':21601,'data_seed_definition':'216010000 + zero-based step, independent from torch model RNG','sampler':'original InfiniteSampler seed21601 shuffle=True','schedule_SHA':sha(ROOT/'TARGET_SOURCE_SCHEDULE.json'),'training_dataset':cfg.train_dataloader.dataset,'optim_wrapper':cfg.optim_wrapper,'param_scheduler':cfg.param_scheduler,'train_cfg_override':{'max_iters':4000,'validation':False,'checkpoints':'initial and step4000 only'},'KL':{'lambda':1,'T':1,'direction':'teacher||student','aggregation':'last class softmax (drop no-object) times bilinear512 align_cornersFalse mask sigmoid, then semantic softmax FP32','support':'all512x512; original inputs resized512, zero geometric padding','torch_model_RNG':'seed21601 each arm; source branch fork_rng preserves target RNG'},'source_sequence_counts':dict(collections.Counter(r['source_sample_id'] for r in schedule)),'prior_Source_budget':request['source_prior_budget'],'target_labels':'all60 scenes dense labels both arms','effective_pipeline_hash_basis':'recorded randint/uniform/choice draws, crop first two randint offsets, photometric values and packed image/mask SHA for every step'})
        phase='building';step=0;GPU_START=time.monotonic();status()
        wrapper,_=build_wrapper(argparse.Namespace(config='configs/protocol/phase64_source_parent_rgb.py',checkpoint=str(SOURCE),precision='fp32',active_block_indices=None))
        model=wrapper.segmentor
        checkpoint=torch.load(SOURCE,map_location='cpu')['state_dict']
        def tensors():return dict(list(model.named_parameters())+list(model.named_buffers()))
        now=tensors();assert all(n in now and torch.equal(v,now[n].detach().cpu()) for n,v in checkpoint.items())
        assert set(now)-set(checkpoint)=={'data_preprocessor.mean','data_preprocessor.std'}
        initial={n:v.detach().cpu().clone() for n,v in model.decode_head.state_dict().items()}
        assert not dict(model.decode_head.named_buffers())
        frozen={n:v.detach().cpu().clone() for n,v in now.items() if not n.startswith('decode_head.')}
        def digest(d):
            h=hashlib.sha256()
            for n,t in sorted(d.items()):h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        teacher=copy.deepcopy(model.decode_head).eval()
        for p in teacher.parameters():p.requires_grad=False
        teacher_digest=digest(teacher.state_dict())
        (ROOT/'heads').mkdir();torch.save({'state_dict':initial,'Source_SHA':SOURCE_SHA},ROOT/'heads/initial.pth')
        def trainmode():
            model.train();model.backbone.eval()
            for name,module in model.named_modules():
                if name and not name.startswith('decode_head'):module.eval()
            model.decode_head.train()
            for n,p in model.named_parameters():p.requires_grad=n.startswith('decode_head.')
            assert sum(p.numel() for p in model.parameters() if p.requires_grad)==1316516
        def frozen_check():
            current=tensors();assert all(torch.equal(v,current[n].detach().cpu()) for n,v in frozen.items())
            assert digest(teacher.state_dict())==teacher_digest
            assert not model.backbone.training and all(p.grad is None for n,p in model.named_parameters() if not n.startswith('decode_head.'))
        def target(row):
            data,proof=augment(row['index'],row['data_seed'])
            assert proof['image_SHA']==row['image_SHA'] and proof['mask_SHA']==row['mask_SHA'] and proof['draws']==row['draws']
            batch=model.data_preprocessor(pseudo_collate([data]),True)
            count('target_full_segmentor')
            losses=model(**batch,mode='loss');loss,logs=model.parse_losses(losses)
            assert torch.isfinite(loss);return loss,{k:float(v.detach()) for k,v in logs.items()}
        def semantic(head,features,kind):
            count(kind);cls,masks=head(features,None)
            return torch.einsum('bqc,bqhw->bchw',F.softmax(cls[-1].float(),-1)[...,:-1],F.interpolate(masks[-1].float(),size=(512,512),mode='bilinear',align_corners=False).sigmoid())
        def source_kl(row):
            path=SRC/'data/cloudsen12_high_l1c/img_dir/train'/row['source_png_name']
            x=torch.from_numpy(load_rgb_image(path,512)).cuda()
            # Fork torch RNG so extra source calls cannot change next target-loss randomness.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                count('source_shared_backbone')
                with torch.no_grad():features=model.extract_feat(wrapper.normalize(x));s=semantic(teacher,features,'source_teacher_head');logp=F.log_softmax(s.float(),dim=1);p=logp.exp()
                student=semantic(model.decode_head,features,'source_student_head');logq=F.log_softmax(student.float(),dim=1)
                kl=(p*(logp-logq)).sum(1).mean()
                assert torch.isfinite(kl) and kl.detach().item()>=-1e-6
                return kl
        phase='gate';trainmode();torch.manual_seed(21601);torch.cuda.manual_seed_all(21601);np.random.seed(21601);random.seed(21601)
        model.decode_head.load_state_dict(initial);a,loga=target(schedule[0]);a.backward();gradC=[n for n,p in model.named_parameters() if p.grad is not None];model.zero_grad(set_to_none=True)
        model.decode_head.load_state_dict(initial);torch.manual_seed(21601);torch.cuda.manual_seed_all(21601);np.random.seed(21601);random.seed(21601)
        b,logb=target(schedule[0]);kl=source_kl(schedule[0]);assert loga==logb and abs(float(kl.detach()))<1e-6
        (b+kl).backward();gradD=[n for n,p in model.named_parameters() if p.grad is not None]
        assert gradC and gradD and all(n.startswith('decode_head.') for n in gradC+gradD)
        assert all(p.grad is None for p in teacher.parameters());frozen_check()
        dump(ROOT/'GATE_ACCEPTED.json',{'target_loss_C':loga,'target_loss_D':logb,'initial_KL':float(kl.detach()),'grad_C':gradC,'grad_D':gradD,'trainable_each':1316516,'checkpoint_exact_tensors':len(checkpoint),'extra_runtime_buffers':list(set(now)-set(checkpoint)),'nonhead_digest':digest(frozen),'teacher_digest':teacher_digest,'initial_head_digest':digest(initial),'initial_head_file_SHA':sha(ROOT/'heads/initial.pth'),'gate_updates':0,'counts':dict(counts)})
        model.zero_grad(set_to_none=True);del a,b,kl;torch.cuda.empty_cache()
        endpoints={}
        for arm in ['C','D']:
            active_arm=arm;phase='training';step=0;model.decode_head.load_state_dict(initial);trainmode()
            torch.manual_seed(21601);torch.cuda.manual_seed_all(21601);np.random.seed(21601);random.seed(21601)
            opt=build_optim_wrapper(model,copy.deepcopy(cfg.optim_wrapper))
            schedulers=[PARAM_SCHEDULERS.build(copy.deepcopy(c),default_args={'optimizer':opt}) for c in cfg.param_scheduler]
            lossfile=(ROOT/f'{arm}_LOSSES.jsonl').open('x',buffering=1);trainclock=time.monotonic();armcounts=dict(counts)
            dump(ROOT/f'{arm}_INITIAL.json',{'head_digest':digest(model.decode_head.state_dict()),'initial_file_SHA':sha(ROOT/'heads/initial.pth'),'optimizer_groups':[{'lr':g['lr'],'weight_decay':g['weight_decay'],'parameters':sum(p.numel() for p in g['params'])} for g in opt.optimizer.param_groups],'source_head_matches':digest(model.decode_head.state_dict())==digest(initial),'schedule_SHA':sha(ROOT/'TARGET_SOURCE_SCHEDULE.json')})
            opt.zero_grad()
            for i,row in enumerate(schedule):
                step=i+1;tl,logs=target(row);k=source_kl(row) if arm=='D' else tl.new_zeros(())
                total=tl+k;opt.backward(total)
                grads=[p.grad for p in model.decode_head.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                opt.step();opt.zero_grad()
                lr=opt.optimizer.param_groups[0]['lr']
                for scheduler in schedulers:scheduler.step()
                lossfile.write(json.dumps({'step':step,'target_loss':float(tl.detach()),'source_KL':float(k.detach()),'total_loss':float(total.detach()),'lr':lr,'target_terms':logs,'target_input_SHA':row['image_SHA'],'source_sample_id':row['source_sample_id'] if arm=='D' else None})+'\n')
                if step%100==0:frozen_check();status()
                del tl,k,total,grads
            frozen_check();lossfile.close()
            endpoint={n:v.detach().cpu().clone() for n,v in model.decode_head.state_dict().items()};endpoints[arm]=endpoint
            p=ROOT/f'heads/{arm}_iter4000.pth';torch.save({'state_dict':endpoint,'Source_SHA':SOURCE_SHA,'steps':4000,'seed':21601,'arm':arm,'schedule_SHA':sha(ROOT/'TARGET_SOURCE_SCHEDULE.json')},p)
            dump(ROOT/f'{arm}_TRAIN_COMPLETE.json',{'steps':4000,'head_SHA':sha(p),'head_digest':digest(endpoint),'initial_digest':digest(initial),'frozen_nonhead_digest':digest(frozen),'teacher_digest':teacher_digest,'train_seconds':time.monotonic()-trainclock,'counts_delta':{k:counts[k]-armcounts.get(k,0) for k in counts},'no_val_or_periodic_checkpoint':True})
            del opt,schedulers;torch.cuda.empty_cache()
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
                if alpha not in [0,1]:torch.save({'state_dict':head,'alpha':alpha,'arm':arm,'Source_SHA':SOURCE_SHA,'formula':'CPU float64 then source dtype'},ROOT/f'heads/{arm}_a{alpha}.pth')
        def evaluate(arm,alpha,pool):
            nonlocal phase,step,active_arm
            phase='evaluate_'+pool;active_arm=arm;model.decode_head.load_state_dict(interpolants[arm,alpha]);model.eval();frozen_check()
            groups=collections.defaultdict(lambda:np.zeros((3,3),np.int64));total=np.zeros((3,3),np.int64);per=[];predsha=hashlib.sha256()
            with torch.inference_mode():
                for step,r in enumerate(pools[pool],1):
                    x=torch.from_numpy(load_rgb_image(Path(r['image_path']),512)).cuda();count('evaluation_full_segmentor')
                    pred=wrapper(x).argmax(1)[0].cpu().numpy().astype(np.uint8);predsha.update(pred.tobytes())
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
        phase='complete';active_arm=None;step=0;frozen_check();assert counts['evaluation_full_segmentor']==4995 and counts['target_full_segmentor']==8002
        assert counts['source_shared_backbone']==4001 and counts['source_teacher_head']==4001 and counts['source_student_head']==4001
        dump(ROOT/'ACCESS_AUDIT.json',{'forbidden_attempts':denied,'source_train_label_reads':0,'source_test_label_reads':0,'counts':dict(access),'source_image_training_uses':4000,'source_gate_uses':1,'source_support':'all512x512 resized RGB; no model padding','official_test_images':'preflight SHA bytes only; no decode','same_product_metadata_caveat':request['metadata_relation']})
        free=hostfree();done.set()
        dump(ROOT/'COMPLETE.json',{'counts':dict(counts),'total_units':sum(counts.values()),'elapsed_seconds':time.monotonic()-start,'GPU_stage_seconds':time.monotonic()-GPU_START,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'host_volumes_final':free,'head_SHA':{p.name:sha(p) for p in (ROOT/'heads').glob('*.pth')},'source_selection_SHA':sha(ROOT/'SOURCE_SELECTION_LOCK.json'),'source_SHA_unchanged':sha(SOURCE)==SOURCE_SHA})
        status();(ROOT/'EXIT_CODE').write_text('0\n')
    except BaseException:
        done.set();(ROOT/'FAILURE.txt').write_text(traceback.format_exc());status();dump(ROOT/'FAILURE_COST.json',{'counts':dict(counts),'total_units':sum(counts.values()),'seconds':time.monotonic()-start});(ROOT/'EXIT_CODE').write_text('1\n');raise
    finally:ledger.close()
if __name__=='__main__':main()

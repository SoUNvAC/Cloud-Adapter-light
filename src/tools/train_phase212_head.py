"""Run exactly one original Phase64 target fit; retain only best head state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT=Path('/home/scv/Cloud-Adapter-light'); SRC=ROOT/'src'
BASE=Path('/home/scv/shared/phase212_retrain_20261020_run02')
PRIOR_FAILED_SECONDS=1.3337932010181248
SOURCE=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'
SOURCE_SHA='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):
    tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(x,indent=2,default=str));tmp.replace(p)


def main():
    a=argparse.ArgumentParser();a.add_argument('--target',choices=['l8','sparcs'],required=True);args=a.parse_args();target=args.target
    out=BASE/target;out.mkdir(exist_ok=False)
    os.chdir(SRC);sys.path.insert(0,str(SRC));os.environ.update(XFORMERS_DISABLED='1',PHASE64_TARGET=target,PHASE64_METHOD='shared_parent',PHASE64_PARENT_CHECKPOINT=str(SOURCE))
    assert sha(SOURCE)==SOURCE_SHA
    import torch
    from mmengine.config import Config
    from mmengine.hooks import Hook
    from mmengine.runner import Runner
    from mmseg.registry import HOOKS
    import cloud_adapter
    import cloud_adapter.models
    import cloud_adapter.datasets
    cfg=Config.fromfile('configs/protocol/phase64_target_parent_rgb.py')
    assert cfg.train_cfg.max_iters==4000 and cfg.train_cfg.val_interval==500
    assert cfg.randomness.seed==64 and cfg.train_dataloader.batch_size==1
    cfg.work_dir=str(out);cfg.resume=False
    # Disable periodic and last checkpoints, keep original mIoU selection at every validation.
    cfg.default_hooks.checkpoint.update(interval=4001,save_last=False,max_keep_ckpts=1,save_optimizer=False,save_param_scheduler=False)
    clock=time.monotonic();calls={'train':0,'val':0};frozen={};handle=None;done=threading.Event();fatal=[]
    ledger=(out/'CALL_LEDGER.jsonl').open('x',buffering=1)
    source=torch.load(SOURCE,map_location='cpu')['state_dict']
    def tensors(m):
        return dict(list(m.named_parameters())+list(m.named_buffers()))
    def digest(items):
        h=hashlib.sha256()
        for n,t in sorted(items.items()):
            h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def status(stage,**extra):
        dump(out/'RUN_STATUS.json',dict(stage=stage,target=target,PID=os.getpid(),calls=dict(calls),elapsed_seconds=time.monotonic()-clock,**extra))
    def guard():
        while not done.wait(10):
            try:
                pending=[os.getpid()];seen=set();rss=0
                while pending:
                    pid=pending.pop()
                    if pid in seen:continue
                    seen.add(pid)
                    try:
                        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                            if line.startswith('VmRSS:'):rss+=int(line.split()[1])*1024
                        pending.extend(int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
                    except (FileNotFoundError,ProcessLookupError):pass
                b=0
                for path in BASE.rglob('*'):
                    try:
                        if path.is_file():b+=path.stat().st_size
                    except FileNotFoundError:pass  # Atomic status replacement is expected.
                used=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
                assert rss<=16*1024**3,('CPU_rss',rss)
                assert used<=20*1024**3,('GPU_peak_allocated',used)
                assert b<=500_000_000,('persistent_bytes',b)
                prior=sum(json.loads(p.read_text()).get('calls',{}).get('train',0)+json.loads(p.read_text()).get('calls',{}).get('val',0) for p in BASE.glob('*/COMPLETED.json') if p.parent!=out)
                assert prior+sum(calls.values())<=16000,('total_calls',prior+sum(calls.values()))
                prior_seconds=sum(json.loads(p.read_text()).get('elapsed_seconds',0) for p in BASE.glob('*/COMPLETED.json') if p.parent!=out)
                assert PRIOR_FAILED_SECONDS+prior_seconds+time.monotonic()-clock<=14400,'training runtime budget'
                status('running',rss_bytes=rss,GPU_peak_allocated_bytes=used,new_persistent_bytes=b)
            except Exception:
                (out/'RESOURCE_FAILURE.txt').write_text(traceback.format_exc());os._exit(71)
    @HOOKS.register_module()
    class Phase212HeadAudit(Hook):
        priority='LOWEST'
        def before_train(self,runner):
            nonlocal handle
            m=runner.model;m.train()
            all_t=tensors(m)
            mismatches=[n for n,t in source.items() if n not in all_t or not torch.equal(t.cpu(),all_t[n].detach().cpu())]
            assert not mismatches,('Source_initialization_mismatch',mismatches[:12])
            trainable={n:p for n,p in m.named_parameters() if p.requires_grad}
            assert trainable and all(n.startswith('decode_head.') for n in trainable),list(trainable)
            assert sum(p.numel() for p in trainable.values())==1316516
            assert not m.backbone.training
            for n,t in all_t.items():
                if not n.startswith('decode_head.'):frozen[n]=t.detach().cpu().clone()
            assert len(runner.train_dataloader.dataset)==(1560 if target=='l8' else 60)
            assert len(runner.val_dataloader.dataset)==(963 if target=='l8' else 10)
            dump(out/'INITIAL_IDENTITY.json',{'Source_SHA':SOURCE_SHA,'source_checkpoint_tensors_exact':len(source),'source_mismatch':mismatches,'all_non_head_digest':digest(frozen),'frozen_tensors':len(frozen),'trainable_head_parameters':1316516,'head_state_bytes':sum(t.numel()*t.element_size() for n,t in all_t.items() if n.startswith('decode_head.')),'config':'phase64_target_parent_rgb.py','target':target,'seed':64})
            def count(mod,inputs):
                prior=sum(sum(json.loads(p.read_text())['calls'].values()) for p in BASE.glob('*/COMPLETED.json') if p.parent!=out)
                assert prior+sum(calls.values())<16000,'Global training-stage call cap before forward'
                mode='train' if mod.training else 'val';calls[mode]+=1
                ledger.write(json.dumps({'call':sum(calls.values()),'mode':mode,'runner_iteration':runner.iter,'elapsed_seconds':time.monotonic()-clock})+'\n')
            handle=m.register_forward_pre_hook(count)
            status('initialized_source_verified');threading.Thread(target=guard,daemon=True).start()
        def before_save_checkpoint(self,runner,checkpoint):
            checkpoint['state_dict']={n:t.detach().cpu().clone() for n,t in tensors(runner.model).items() if n.startswith('decode_head.')}
            checkpoint.pop('optimizer',None);checkpoint.pop('param_schedulers',None)
            checkpoint['phase212_reconstruction']={'Source_SHA':SOURCE_SHA,'Source_path':str(SOURCE),'frozen_non_head_digest':digest(frozen),'format':'Source complete model plus strict decode_head override','target':target,'new_retraining_not_old_original':True}
            assert checkpoint['state_dict'] and all(n.startswith('decode_head.') for n in checkpoint['state_dict'])
        def after_train(self,runner):
            all_t=tensors(runner.model);bad=[n for n,t in frozen.items() if n not in all_t or not torch.equal(t,all_t[n].detach().cpu())]
            assert not bad,('Frozen_non_head_changed',bad[:10])
            assert calls['train']==4000 and calls['val']==8*(963 if target=='l8' else 10),calls
            dump(out/'FROZEN_FINAL_AUDIT.json',{'mismatches':bad,'frozen_tensors':len(frozen),'digest':digest(frozen),'Source_SHA':SOURCE_SHA,'calls':calls})
    cfg.custom_hooks=list(cfg.get('custom_hooks',[]))+[dict(type='Phase212HeadAudit')]
    status('building_runner_zero_calls')
    try:
        runner=Runner.from_cfg(cfg)
        if cfg.get('deterministic_warn_only',False):torch.use_deterministic_algorithms(True,warn_only=True)
        runner.train()
        heads=list(out.glob('best_mIoU_iter_*.pth'));assert len(heads)==1,[str(p) for p in heads]
        saved=torch.load(heads[0],map_location='cpu');assert all(n.startswith('decode_head.') for n in saved['state_dict'])
        dump(out/'COMPLETED.json',{'status':'complete_retrained_head','target':target,'head_path':str(heads[0]),'head_sha256':sha(heads[0]),'head_bytes':heads[0].stat().st_size,'calls':calls,'elapsed_seconds':time.monotonic()-clock,'Source_SHA':SOURCE_SHA,'new_retraining_not_old_checkpoint':True,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        status('complete');(out/'EXIT_CODE').write_text('0\n')
    except BaseException:
        (out/'FAILURE.txt').write_text(traceback.format_exc());status('failed');(out/'EXIT_CODE').write_text('1\n');raise
    finally:
        done.set();ledger.close()


if __name__=='__main__':main()

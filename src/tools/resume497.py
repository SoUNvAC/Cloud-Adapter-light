"""Audited continuation after remote reboot. Original training code stays locked."""
import json,hashlib,shutil,time,traceback
from pathlib import Path
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import HOOKS
from mmseg.utils import register_all_modules
from phase497.integration import Audit497,digest
R=Path('/home/scv/shared/phase497_20261110')
CHECKPOINT=R/'recovery_checkpoint_28000.pth'
def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
@HOOKS.register_module()
class ResumeAudit497(Audit497):
    def before_train(self,runner):
        assert runner.iter==28000
        c=torch.load(CHECKPOINT,map_location='cpu')
        assert equal(runner.model.state_dict(),c['state_dict'])
        assert equal(runner.optim_wrapper.state_dict(),c['optimizer'])
        assert equal([s.state_dict() for s in runner.param_schedulers],c['param_schedulers'])
        self.frozen=digest(runner.model.backbone.encoder.state_dict().items())
        original=json.loads((R/'texture/INITIAL.json').read_text())
        assert self.frozen==original['frozen_sha']
        self.initial={n:p.detach().cpu().clone() for n,p in runner.model.named_parameters() if p.requires_grad}
        assert set(self.initial)==set(original['names'])
        ids=[id(p) for g in runner.optim_wrapper.optimizer.param_groups for p in g['params']]
        assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in runner.model.parameters() if p.requires_grad}
        self.reference={x['step']:x['batch_sha'] for x in map(json.loads,Path('/home/scv/shared/phase496_20261110/texture/INPUT_TRACE.jsonl').read_text().splitlines())}
        (R/'RESUME_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,iteration=runner.iter,model_exact=True,optimizer_exact=True,schedulers_exact=True,frozen_unchanged=True,checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),rng_state_saved=False,note='Checkpoint did not store RNG state; no claim of bitwise uninterrupted equivalence. MMEngine advances the dataloader 28000 batches.'),indent=2))
    def before_train_iter(self,runner,batch_idx,data_batch=None):
        super().before_train_iter(runner,batch_idx,data_batch)
        step=runner.iter+1
        if step%2000==0:
            row=json.loads((R/'texture/INPUT_TRACE.jsonl').read_text().splitlines()[-1])
            with (R/'RESUME_TRACE_CHECK.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,matches_phase496=row['batch_sha']==self.reference[step]))+'\n')
def main():
    lock=json.loads((R/'texture/INPUT_LOCK.json').read_text())
    for n,h in lock['code_sha256'].items():assert hashlib.sha256((R/'code/phase497'/n).read_bytes()).hexdigest()==h
    assert not (R/'texture/COMPLETE.json').exists()
    if not CHECKPOINT.exists():shutil.copy2(R/'texture/iter_28000.pth',CHECKPOINT)
    register_all_modules()
    cfg=Config.fromfile(str(R/'code/phase497/config.py'))
    cfg.model.backbone.variant='texture';cfg.work_dir=str(R/'texture')
    cfg.resume=True;cfg.load_from=str(CHECKPOINT);cfg.custom_hooks=[dict(type='ResumeAudit497')]
    (R/'STATUS.json').write_text(json.dumps(dict(state='resuming',from_step=28000,time=time.time())))
    runner=Runner.from_cfg(cfg);runner.train();assert runner.iter==40000
    (R/'texture/COMPLETE.json').write_text(json.dumps(dict(steps=40000,variant='texture',resumed_from=28000)))
    (R/'STATUS.json').write_text(json.dumps(dict(state='evaluating',resumed_from=28000,time=time.time())))
    from phase497.evaluate import main as evaluate
    evaluate()
    (R/'STATUS.json').write_text(json.dumps(dict(state='complete',resumed_from=28000,time=time.time())))
if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'RESUME_FAILURE.txt').write_text(traceback.format_exc());raise

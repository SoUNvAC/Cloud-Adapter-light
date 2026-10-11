"""Strict Source-to-LoRA preflight followed by exactly one fixed 4000-step run."""
import os,sys,time,subprocess
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase80_common import *

def setup():
    os.environ['PHASE65_PREPARED_MANIFEST']=str(PREP/'manifest.json')
    os.environ['PHASE65_SOURCE_CHECKPOINT']=str(SOURCE)
    from mmseg.utils import register_all_modules
    register_all_modules()
    import cloud_adapter.models,cloud_adapter.datasets.phase65_catalogue

def main():
    deadline();md=base_check()
    assert not WORK.exists() and not ROOT.exists(),'Preserve prior attempt'
    assert torch.cuda.is_available()
    processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    active=[s for s in processes.splitlines() if ('python' in s and
            any(v in s for v in ['tools/train.py','run_phase80_training.py']) and
            str(os.getpid())!=s.split()[0] and 'bash -' not in s and 'ssh ' not in s)]
    if active:raise RuntimeError('Other training process: '+str(active))
    free,total=torch.cuda.mem_get_info()
    if free<20*1024**3:raise RuntimeError('Insufficient free CUDA memory')
    if __import__('shutil').disk_usage(WORK.parent).free<10*1024**3:raise RuntimeError('Insufficient checkpoint disk')
    WORK.mkdir();ROOT.mkdir();setup()
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.runner import Runner
    from mmengine.hooks import Hook
    from mmseg.registry import MODELS,DATASETS
    torch.manual_seed(65);np.random.seed(65)
    cfg=Config.fromfile(str(REPO/'src/configs/protocol/phase80_lora_rgb.py'))
    old=Config.fromfile(str(REPO/'src/configs/protocol/phase65a_explore_rgb.py'))
    for key in ['train_dataloader','val_dataloader','test_dataloader','param_scheduler','optim_wrapper','randomness','train_cfg']:
        assert cfg[key]==old[key],('Inherited setting drift',key)
    assert cfg.optim_wrapper.type=='OptimWrapper' and cfg.optim_wrapper.optimizer.type=='AdamW'
    assert cfg.default_hooks.checkpoint.interval==500 and cfg.default_hooks.checkpoint.max_keep_ckpts==-1
    assert cfg.default_hooks.checkpoint.save_best=='mIoU' and cfg.default_hooks.checkpoint.rule=='greater'
    source_state=torch.load(SOURCE,map_location='cpu')['state_dict']
    src_cfg=Config.fromfile(str(REPO/'src/configs/protocol/phase64_source_parent_rgb.py'))
    src_cfg.model.backbone.init_cfg=None
    src=MODELS.build(src_cfg.model);src.init_weights();src.load_state_dict(source_state,strict=True)
    src.eval().cuda()
    ds=DATASETS.build(dict(cfg.val_dataloader.dataset,cohort='fit_representatives'))
    sample=ds[0];product=Path(sample['data_samples'].img_path).name.removesuffix('_rgb.npy')
    with torch.no_grad():expected=src.test_step(pseudo_collate([sample]))[0].seg_logits.data.cpu()
    del src;torch.cuda.empty_cache()
    # Reset RNG so LoRA initialization is independent of Source construction.
    torch.manual_seed(65);np.random.seed(65)
    model=MODELS.build(cfg.model);model.init_weights();state=model.state_dict()
    extra=state_keys(source_state,state)
    assert len(extra)==12*4*2
    merged={**state,**source_state};model.load_state_dict(merged,strict=True)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in source_state.items())
    assert all(torch.count_nonzero(model.state_dict()[k])==0 for k in extra if k.endswith('lora_b'))
    model.train();trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    assert set(trainable)==set(extra),'Non-LoRA trainable parameter'
    model.eval().cuda();errors={}
    for enabled in [False,True]:
        model.backbone.set_lora_enabled(enabled)
        with torch.no_grad():observed=model.test_step(pseudo_collate([sample]))[0].seg_logits.data.cpu()
        errors[str(enabled)]=float((observed-expected).abs().max())
        assert errors[str(enabled)]<=1e-6,errors
    model.train();torch.cuda.reset_peak_memory_stats()
    train_ds=DATASETS.build(cfg.train_dataloader.dataset)
    batch=model.data_preprocessor(pseudo_collate([train_ds[0]]),training=True)
    losses=model.loss(batch['inputs'],batch['data_samples'])
    loss=sum((sum(v) if isinstance(v,list) else v).mean() for k,v in losses.items() if 'loss' in k)
    assert torch.isfinite(loss);loss.backward()
    gradients={n:dict(connected=p.grad is not None,finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else False,
                     norm=float(p.grad.norm()) if p.grad is not None else None)
               for n,p in model.named_parameters() if p.requires_grad}
    assert all(g['connected'] and g['finite'] for g in gradients.values())
    assert any(g['norm']>0 for g in gradients.values())
    assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in source_state.items())
    model.zero_grad(set_to_none=True)
    preflight=dict(status='passed',source_sha256=SOURCE_SHA,manifest_sha256=MANIFEST_SHA,
        split_sha256=SPLIT_SHA,fit_input_product=product,zero_residual_max_absolute_logit_error=errors,
        base_tensors_identical=True,strict_load=True,new_keys=extra,trainable_names=trainable,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),gradients=gradients,
        preflight_loss=float(loss.detach()),gpu=torch.cuda.get_device_name(),free_cuda_bytes=free,total_cuda_bytes=total,
        training_products=155,training_groups=153,fit_representatives=153,development_groups=32,
        valid_label_pixels_by_training_product=int(sum(np.sum(np.load(PREP/r['mask_path'])!=255) for r in md['rows'] if r['split']=='fit')))
    write(ROOT/'preflight.json',preflight)
    cfg.work_dir=str(WORK/'lora_seed65');cfg.load_from=None
    cfg.dump(str(ROOT/'resolved_config.py'))
    files=[PROTOCOL,Path(__file__),Path(__file__).with_name('phase80_common.py'),
           Path(__file__).with_name('run_phase80_analysis.py'),Path(__file__).with_name('phase65d_recovery_math.py'),
           Path(__file__).with_name('phase65d_metrics.py'),Path(__file__).with_name('run_phase65d_recovery.py'),
           REPO/'src/cloud_adapter/models/backbones/phase64_lora_dinov2.py',
           REPO/'src/cloud_adapter/models/backbones/phase64_lora_layers.py',
           REPO/'src/cloud_adapter/models/segmentors/frozen_encoder_decoder.py',
           Path(__file__).with_name('test_phase80.py'),Path(__file__).with_name('run_phase80.sh'),
           REPO/'src/configs/protocol/phase80_lora_rgb.py']
    write(ROOT/'training_lock.json',dict(source_sha256=SOURCE_SHA,manifest_sha256=MANIFEST_SHA,split_sha256=SPLIT_SHA,
        code_sha256={str(p.relative_to(REPO)):digest(p) for p in files},
        inherited_phase65a_config_sha256=digest(REPO/'src/configs/protocol/phase65a_explore_rgb.py'),
        preflight_sha256=digest(ROOT/'preflight.json'),resolved_config_sha256=digest(ROOT/'resolved_config.py'),
        seed=65,max_iters=4000,checkpoint_selection='development mIoU only, strict greater; earliest tie',
        no_confirmation_access=True,no_alcd=True))
    # Pass the already strictly loaded/frozen model; optimizer sees LoRA parameters only.
    class StrictRunner(Runner):
        def build_model(self,model_config):
            return model
    runner=StrictRunner.from_cfg(cfg)
    class DeadlineAndFreeze(Hook):
        def before_train_iter(self,runner,batch_idx,data_batch=None):
            deadline()
            assert {n for n,p in runner.model.named_parameters() if p.requires_grad}==set(extra)
        def after_val_epoch(self,runner,metrics=None):
            with (ROOT/'training_curve.jsonl').open('a') as f:
                __import__('json').dump(dict(iteration=runner.iter,metrics=metrics),f);f.write('\n')
    runner.register_hook(DeadlineAndFreeze(),priority='VERY_HIGH')
    torch.use_deterministic_algorithms(True,warn_only=True)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    runner.train()
    assert runner.iter==4000
    # Every source tensor, including buffers, must remain exactly unchanged.
    final=runner.model.state_dict()
    assert all(torch.equal(final[k].cpu(),v) for k,v in source_state.items())
    files=sorted((WORK/'lora_seed65').glob('iter_*.pth'))
    assert {p.name for p in files}=={'iter_'+str(i)+'.pth' for i in range(500,4001,500)}
    curve=[__import__('json').loads(s) for s in (ROOT/'training_curve.jsonl').read_text().splitlines()]
    assert [r['iteration'] for r in curve]==list(range(500,4001,500))
    best=max(curve,key=lambda r:(r['metrics']['mIoU'],-r['iteration']))['iteration']
    bestfiles=list((WORK/'lora_seed65').glob('best_mIoU*.pth'))
    assert len(bestfiles)==1 and bestfiles[0].name=='best_mIoU_iter_'+str(best)+'.pth'
    checkpoint_rows=[dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size) for p in files+bestfiles]
    write(ROOT/'training_completed.json',dict(status='completed_4000_steps',seconds=time.monotonic()-started,
         peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
         base_tensors_still_identical=True,checkpoints=checkpoint_rows,best_iteration=best,
         best_checkpoint=str(bestfiles[0]),best_checkpoint_sha256=digest(bestfiles[0]),curve=curve,
         trainable_parameters=preflight['trainable_parameters']))
    print('Phase80 fixed training completed',best,flush=True)

if __name__=='__main__':main()

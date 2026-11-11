"""One target update plus 0.1 source loss, with isolated source RNG and BN."""
import copy
import hashlib
import json
import random
import types
import numpy as np
import torch
from torch import nn


def install_replay(runner, source_cfg, root):
    cfg=copy.deepcopy(source_cfg)
    cfg.update(batch_size=1,num_workers=2,persistent_workers=True)
    cfg['sampler']=dict(type='InfiniteSampler',shuffle=True,seed=43802)
    py_state=random.getstate();np_state=np.random.get_state()
    with torch.random.fork_rng():
        loader=runner.build_dataloader(cfg,seed=43802)
        iterator=iter(loader)
    random.setstate(py_state);np.random.set_state(np_state)
    assert len(loader.dataset)>0
    paths=[str(loader.dataset.get_data_info(i)['img_path']) for i in range(len(loader.dataset))]
    assert all('/train/' in x for x in paths)
    (root/'REPLAY_LOCK.json').write_text(json.dumps(dict(images=len(paths),paths_sha256=hashlib.sha256('\n'.join(paths).encode()).hexdigest(),seed=43802,weight=.1,bn_source_eval=True,extra_source_batches=4000)))
    calls=0

    def train_step(model,data,optim_wrapper):
        nonlocal calls
        calls+=1
        with optim_wrapper.optim_context(model):
            target=model.data_preprocessor(data,True)
            target_loss,target_logs=model.parse_losses(model._run_forward(target,mode='loss'))
        optim_wrapper.backward(target_loss)
        # Source loss must not perturb target augmentation/model random streams.
        py_state=random.getstate();np_state=np.random.get_state()
        bn=[(m,m.training) for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
        before={n:b.detach().clone() for n,b in model.named_buffers() if n.endswith(('running_mean','running_var','num_batches_tracked'))}
        source_grad=[];handles=[]
        try:
            for m,_ in bn:m.eval()
            with torch.random.fork_rng():
                random.seed(43802000+calls);np.random.seed(43802000+calls);torch.manual_seed(43802000+calls)
                sample=next(iterator)
                with optim_wrapper.optim_context(model):
                    source=model.data_preprocessor(sample,True)
                    source_loss,_=model.parse_losses(model._run_forward(source,mode='loss'))
                assert torch.isfinite(source_loss)
                if calls in (1,10,100) or calls%500==0:
                    for p in model.parameters():
                        if p.requires_grad:
                            handles.append(p.register_hook(lambda g:source_grad.append(g.detach().abs().sum())))
                optim_wrapper.update_params(.1*source_loss)
        finally:
            for h in handles:h.remove()
            for m,training in bn:m.train(training)
            random.setstate(py_state);np.random.set_state(np_state)
        assert all(torch.equal(before[n],b) for n,b in model.named_buffers() if n in before)
        if source_grad:
            magnitude=float(torch.stack(source_grad).sum())
            assert magnitude>0 and np.isfinite(magnitude)
            (root/'REPLAY_CHECK.json').write_text(json.dumps(dict(step=calls,source_gradient_l1=magnitude,source_loss=float(source_loss.detach()),bn_buffers_unchanged=True,one_optimizer_update_per_target_batch=True)))
        target_logs['source_loss']=source_loss.detach()
        return target_logs

    runner.model.train_step=types.MethodType(train_step,runner.model)

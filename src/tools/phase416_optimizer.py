"""Keep baseline clipping on MsRE; clip the detached semantic head separately."""
import os,json
from pathlib import Path
import torch
from mmengine.optim import OptimWrapper
from mmengine.registry import OPTIM_WRAPPERS

@OPTIM_WRAPPERS.register_module()
class Phase416OptimWrapper(OptimWrapper):
    def _clip_grad(self):
        assert hasattr(self,'semantic_ids') and len(self.semantic_ids)==2
        params=[p for g in self.optimizer.param_groups for p in g['params'] if p.requires_grad and p.grad is not None]
        main=[p for p in params if id(p) not in self.semantic_ids]
        aux=[p for p in params if id(p) in self.semantic_ids]
        assert main and len(aux)==2
        main_norm=self.clip_func(main,**self.clip_grad_kwargs)
        aux_norm=self.clip_func(aux,**self.clip_grad_kwargs)
        self.message_hub.update_scalar('train/'+self.grad_name,float(main_norm))
        self.message_hub.update_scalar('train/aux_grad_norm',float(aux_norm))
        if 'PHASE400_ROOT' in os.environ:
            self.clip_step=getattr(self,'clip_step',0)+1
            with (Path(os.environ['PHASE400_ROOT'])/'CLIP_TRACE.jsonl').open('a') as f:
                f.write(json.dumps({'step':self.clip_step,'main_tensors':len(main),'aux_tensors':len(aux),'main_norm':float(main_norm),'aux_norm':float(aux_norm)})+'\n')

def optimizer_self_test():
    # Auxiliary gradients vary by orders of magnitude; main updates must match
    # the original OptimWrapper exactly, including AdamW state evolution.
    base=torch.nn.Parameter(torch.tensor([1.,-2.]))
    main=torch.nn.Parameter(base.detach().clone())
    aux=[torch.nn.Parameter(torch.ones(2)),torch.nn.Parameter(torch.ones(1))]
    a=OptimWrapper(torch.optim.AdamW([base],lr=1e-4),clip_grad=dict(max_norm=1.,norm_type=2))
    b=Phase416OptimWrapper(torch.optim.AdamW([main]+aux,lr=1e-4),clip_grad=dict(max_norm=1.,norm_type=2))
    b.semantic_ids={id(p) for p in aux}
    for scale in [0.01,100.,10000.]:
        base.grad=torch.tensor([3.,4.]);main.grad=base.grad.clone()
        for p in aux:p.grad=torch.full_like(p,scale)
        a.step();b.step()
        assert torch.equal(base,main)
    assert all(not torch.equal(p,torch.ones_like(p)) for p in aux)

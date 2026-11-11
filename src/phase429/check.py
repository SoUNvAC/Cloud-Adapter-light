"""Structural checks with the official architecture, without pretrained weights.

Run: PYTHONPATH=<official DINOv3 repo>:$PYTHONPATH python -m phase429.check
These are implementation checks, not segmentation performance experiments.
"""
import json
import os
from pathlib import Path
import torch
from torch.nn import functional as F
from dinov3.hub.backbones import dinov3_vits16plus
from .network import CloudBackbone

def main():
    torch.set_num_threads(4)
    torch.manual_seed(42901)
    device = os.environ.get('PHASE429_CHECK_DEVICE','cpu')
    encoder = dinov3_vits16plus(pretrained=False).to(device)
    baseline = CloudBackbone(encoder=encoder, variant='baseline').to(device).eval()
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device=='cuda' else None
    fused = CloudBackbone(encoder=encoder, variant='fused').to(device).eval()
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    for name in ('spatial', 'interactions'):
        a, b = getattr(baseline,name).state_dict(), getattr(fused,name).state_dict()
        assert all(torch.equal(v,b[k]) for k,v in a.items())
    image = torch.randn(2,3,64,96,device=device)
    with torch.no_grad():
        native = encoder.get_intermediate_layers(image, n=[2,5,8,11], reshape=True, norm=True)
        actual = baseline(image, adapt=False)
        for i,(a,b) in enumerate(zip(actual,native)):
            torch.testing.assert_close(a,F.interpolate(b,scale_factor=(4,2,1,0.5)[i],mode='bilinear',align_corners=False))
    baseline.train()
    assert not encoder.training and all(not p.requires_grad for p in encoder.parameters())
    frozen = {n:p.detach().clone() for n,p in encoder.named_parameters()}
    for model in (baseline,fused):
        model.train()
        active = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(active,lr=0.001)
        initial = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        for _ in range(3):
            optimizer.zero_grad()
            outputs = model(image)
            # Nonconstant probe avoids the nearly constant squared LayerNorm norm.
            loss = sum((x*torch.randn_like(x)).mean() for x in outputs)
            loss.backward()
            assert all(p.grad is None for p in encoder.parameters())
            optimizer.step()
        names = dict(model.named_parameters())
        for prefix in ('spatial.','interactions.') + (('refinement.','local.') if model.variant=='fused' else ()):
            assert any(not torch.equal(p,names[n]) for n,p in initial.items() if n.startswith(prefix)), prefix
        if model.variant=='fused':
            assert all(p.grad is not None and p.grad.abs().sum()>0 for p in model.local.parameters())
            # Spatial locality: shifting detail by one patch must change the read.
            x=torch.randn(2,24,384,device=device);detail=torch.randn(2,64,16,24,device=device)
            assert not torch.equal(model.local.read(x,detail,4,6),model.local.read(x,detail.roll(4,-1),4,6))
        assert all(torch.equal(p,frozen[n]) for n,p in encoder.named_parameters())
    # Real Mask2Former construction, loss/backprop, optimizer coverage, and shared head init.
    from mmengine.config import Config
    from mmengine.optim import build_optim_wrapper
    from mmseg.utils import register_all_modules
    from mmseg.registry import MODELS
    from mmseg.structures import SegDataSample
    from mmengine.structures import PixelData
    register_all_modules()
    from . import integration
    cfg=Config.fromfile(str(Path(__file__).with_name('config.py')))
    heads=[]
    for variant in ('baseline','fused'):
        torch.manual_seed(42901)
        model_cfg=cfg.model.copy()
        model_cfg.backbone=dict(type='DinoCloud429',variant=variant,encoder=encoder)
        model=MODELS.build(model_cfg).to(device)
        model.init_weights()
        heads.append(integration.digest(model.decode_head.state_dict().items()))
        wrapper=build_optim_wrapper(model,cfg.optim_wrapper)
        active={id(p) for p in model.parameters() if p.requires_grad}
        opt={id(p) for g in wrapper.optimizer.param_groups for p in g['params']}
        assert active==opt
        shape=(512,512) if device=='cuda' else (64,96)
        sample=SegDataSample(metainfo=dict(img_shape=shape,ori_shape=shape,pad_shape=shape))
        sample.gt_sem_seg=PixelData(data=torch.randint(0,3,(1,*shape),device=device))
        probe=torch.randn(2,3,*shape,device=device)
        losses=model.loss(probe,[sample,sample.clone()])
        loss,_=model.parse_losses(losses)
        assert torch.isfinite(loss)
        wrapper.update_params(loss)
        with torch.no_grad():
            model.eval();scores=model.encode_decode(probe[:1],[sample.metainfo])
            assert scores.shape==(1,3,*shape) and torch.isfinite(scores).all()
    assert heads[0]==heads[1], 'head initialization differs between arms'
    result=dict(passed=True, official_unadapted_parity=True, register_tokens_excluded=True,
        rectangular_input=True, common_initialization_equal=True, head_initialization_equal=True,
        gradient_through_frozen_blocks=True, frozen_unchanged=True, local_read_write_gradients=True,
        actual_segmentation_loss_and_prediction=True, optimizer_exact=True,
        pretrained_weights_used=False, formal_training_started=False, device=device)
    print('PHASE429_CHECK='+json.dumps(result))

if __name__=='__main__':
    main()

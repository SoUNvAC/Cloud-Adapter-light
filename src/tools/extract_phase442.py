"""Frozen backbone features; common native-token indices across three models."""
import hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
R=Path('/home/scv/shared/phase442_20261103');P=Path('/home/scv/shared/phase431_20261102');SRC=Path('/home/scv/Cloud-Adapter-light/src')
ARM=sys.argv[1];OUT=R/ARM
sys.path[:0]=[str(P/'code'),'/home/scv/shared/phase429_20261102/dinov3']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_sha(model):
    h=hashlib.sha256()
    for n,t in model.state_dict().items():h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def save(n,x):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def main():
    from mmengine.config import Config
    from mmengine.dataset import Compose,pseudo_collate
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules
    register_all_modules();import phase431.integration
    spec=importlib.util.spec_from_file_location('catalogue441',SRC/'cloud_adapter/datasets/phase65_catalogue.py');spec.loader.exec_module(importlib.util.module_from_spec(spec))
    OUT.mkdir(exist_ok=False);os.chdir(SRC);torch.set_num_threads(3);torch.manual_seed(44101);torch.backends.cudnn.benchmark=False
    cfg=Config.fromfile(str(P/'code/phase431/config.py'));cfg.model.backbone.variant='fused'
    checkpoint=Path('/home/scv/shared/phase429_20261102/dinov3_splus_native.pth')
    if ARM=='official':
        from dinov3.hub.backbones import dinov3_vits16plus
        model=dinov3_vits16plus(pretrained=True,weights=str(checkpoint)).cuda().eval()
        preprocessor=MODELS.build(cfg.model.data_preprocessor).cuda().eval();bnpath=None
        expected=torch.load(checkpoint,map_location='cpu');assert all(torch.equal(t.cpu(),expected[n]) for n,t in model.state_dict().items());del expected
    else:
        model=MODELS.build(cfg.model).cuda().eval();preprocessor=model.data_preprocessor
        checkpoint=P/'fused/best_mIoU_iter_32000.pth' if ARM=='source' else Path('/home/scv/shared/phase433_20261103/431/iter_4000.pth')
        bnpath=Path('/home/scv/shared/phase434_20261103')/('source_fit_bn_buffers.pth' if ARM=='source' else 'final_fit_bn_buffers.pth')
        model.load_state_dict({**torch.load(checkpoint,map_location='cpu')['state_dict'],**torch.load(bnpath,map_location='cpu')},strict=True)
    model.requires_grad_(False);before=state_sha(model);values={}
    def put(name,x):
        if name != "final":return
        if x.ndim==4:x=F.adaptive_avg_pool2d(x,(32,32)).flatten(2).transpose(1,2)
        assert x.shape[0]==1 and x.shape[1]==1024
        values[name]=x[0].detach()
    if ARM!='official':
        def spatial(m,a,o):put('spatial_context',o[0]);put('spatial_detail',o[1])
        model.backbone.spatial.register_forward_hook(spatial)
        for i,m in enumerate(model.backbone.interactions):
            def interaction(m,a,o,i=i):put(f's{i+1}_pre',a[0]);put(f's{i+1}_attention',o)
            m.register_forward_hook(interaction)
        def refine(m,a,o):put(f's{int(a[2])+1}_post',o)
        model.backbone.refinement.register_forward_hook(refine)
        def output(m,a,o):
            if int(a[1])==3:put('final',o)
        model.backbone.output_residual.register_forward_hook(output)
    save('LOCK.json',dict(model=ARM,checkpoint=str(checkpoint),checkpoint_sha=sha(checkpoint),bn=str(bnpath),bn_sha=sha(bnpath) if bnpath else None,
        state_sha=before,input_sha=sha(R/'INPUTS.json'),worker_sha=sha(__file__),training_updates=0,
        post_definition='After full token refinement; attention separately records immediate cross-attention output; final after last norm and output residual before pyramid resize',
        official='Unmodified official encoder, raw block outputs 3/6/9/12 and final normalized patches; no CNN/interaction/head'))
    rows=json.loads((R/'INPUTS.json').read_text());start=time.monotonic();signatures=[]
    for row in rows:
        domain=row['domain'];info=dict(img_path=row['image'],seg_map_path=row['mask'],seg_fields=[],label_map={0:0,1:1,2:1,3:2} if domain=='A' else None,reduce_zero_label=False)
        pipe=Compose([dict(type='LoadImageFromFile' if domain=='A' else 'LoadPhase65RGB'),dict(type='Resize',scale=(512,512)),dict(type='PackSegInputs')])
        sample=pipe(info);values.clear()
        with torch.inference_mode():
            batch=preprocessor(pseudo_collate([sample]),training=False);image=batch['inputs'];assert image.shape==(1,3,512,512)
            if ARM=='official':
                x,(h,w)=model.prepare_tokens_with_masks(image);rope=model.rope_embed(H=h,W=w)
                for i,block in enumerate(model.blocks):
                    x=block(x,rope)
                    if i in [2,5,8,11]:put(f's{[2,5,8,11].index(i)+1}_block',x[:,5:])
                put('final',model.norm(x[:,5:]))
            else:model.backbone(image)
        ids=np.arange(1024,dtype=np.int64)
        path=OUT/'features'/f"{row['index']:03d}.npz";path.parent.mkdir(exist_ok=True)
        arrays={name:x[ids].float().cpu().numpy() for name,x in values.items()}
        assert all(np.isfinite(x).all() for x in arrays.values())
        np.savez_compressed(path,**arrays)
        signatures.append(dict(index=row['index'],input_tensor_sha=hashlib.sha256(image.cpu().numpy().tobytes()).hexdigest(),feature_sha=sha(path),tokens=len(ids),locations=list(arrays)))
        if row['index']%10==0:save('STATUS.json',dict(completed=row['index']+1,total=567,seconds=time.monotonic()-start))
    assert state_sha(model)==before and all(not m.training for m in model.modules()) and all(p.grad is None for p in model.parameters())
    save('SIGNATURES.json',signatures);save('COMPLETE.json',dict(passed=True,forwards=567,training_updates=0,state_unchanged=True,seconds=time.monotonic()-start))
if __name__=='__main__':
    try:main()
    except BaseException:
        OUT.mkdir(exist_ok=True);save('FAILURE.json',dict(error=traceback.format_exc()));raise

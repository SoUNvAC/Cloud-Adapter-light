"""Paired GPU-forward latency on the two final networks; excludes IO/preprocessing."""
from pathlib import Path
import sys,json,statistics,time
import torch
R=Path('/home/scv/shared/phase499_20261110')
assert json.loads((R/'STATUS.json').read_text())['state']=='complete'
sys.path[:0]=[str(R/'code'),'/home/scv/shared/phase496_20261110/code','/home/scv/Cloud-Adapter-light/src']
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmseg.utils import register_all_modules
from mmseg.registry import MODELS
import phase496.integration
import phase499.integration
register_all_modules()
torch.set_num_threads(4)
models={}
for v in ['phase496','phase499']:
    root=Path('/home/scv/shared')/(v+'_20261110')
    cfg=Config.fromfile(str(root/'code'/v/'config.py'))
    cfg.model.backbone.update(variant='texture',repo='/home/scv/shared/phase429_20261102/dinov3',weights='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth')
    model=MODELS.build(cfg.model)
    load_checkpoint(model,str(root/'texture/iter_40000.pth'),map_location='cpu',strict=True)
    models[v]=model.cuda().eval()
torch.manual_seed(496)
x=torch.randn(1,3,512,512,device='cuda')
meta=[dict(ori_shape=(512,512),img_shape=(512,512),pad_shape=(512,512),padding_size=[0,0,0,0])]
times={v:[] for v in models}
with torch.inference_mode():
    for i in range(52):
        order=['phase496','phase499'] if i%2==0 else ['phase499','phase496']
        for v in order:
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();out=models[v].encode_decode(x,meta);b.record();b.synchronize()
            assert out.shape==(1,3,512,512) and torch.isfinite(out).all()
            if i>=20:times[v].append(a.elapsed_time(b))
result={'input':[1,3,512,512],'dtype':'float32','warmup':20,'repeats':32,'scope':'encode_decode, no IO/preprocessing','gpu':torch.cuda.get_device_name(0),'measurements_ms':times}
result['median_ms']={v:statistics.median(x) for v,x in times.items()}
result['total_parameters']={v:sum(p.numel() for p in m.parameters()) for v,m in models.items()}
(R/'COST.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))

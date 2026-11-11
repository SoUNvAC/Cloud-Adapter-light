from pathlib import Path
import sys,os,json,copy,time
import numpy as np
import torch
from PIL import Image
R=Path('/home/scv/shared/phase494_20261109');P=Path('/home/scv/shared/phase431_20261102')
sys.path[:0]=['/home/scv/shared/phase450_20261106',str(P/'code'),'/home/scv/shared/phase400_20261031/code','/home/scv/shared/phase488_20261108','/home/scv/shared/phase493_20261109']
os.chdir('/home/scv/Cloud-Adapter-light/src')
from infer_shadow493 import Frozen431WithOcclusion as New
from infer_shadow488 import Frozen431WithOcclusion as Old
from train_shadow450 import digest
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;register_all_modules();import phase431.integration
cfg=Config.fromfile(str(P/'code/phase431/config.py'));cfg.model.backbone.variant='fused';cfg.model.backbone.repo='/home/scv/shared/phase429_20261102/dinov3';cfg.model.backbone.weights='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth'
models={}
for name,root,cls in [('488B',Path('/home/scv/shared/phase488_20261108/B/network_run'),Old),('493B',Path('/home/scv/shared/phase493_20261109/B/network_run'),New)]:
 teacher=MODELS.build(cfg.model);teacher.load_state_dict(torch.load(P/'fused/best_mIoU_iter_32000.pth',map_location='cpu')['state_dict'],strict=True)
 s=torch.load(root/'branch_6000.pth',map_location='cpu');loaded=teacher.load_state_dict(s['student_patch'],strict=False);assert not loaded.unexpected_keys and all(n.startswith('backbone.encoder.') for n in loaded.missing_keys)
 models[name]=cls(teacher,s['state_dict']).cuda().eval()
row=next(x for x in json.loads(Path('/home/scv/shared/phase442_20261103/INPUTS.json').read_text()) if x['domain']=='A')
rgb=torch.from_numpy(np.array(Image.open(row['image']).convert('RGB'))).permute(2,0,1)[None].cuda();before={n:digest(m) for n,m in models.items()};samples={n:[] for n in models}
with torch.inference_mode():
 for _ in range(5):
  for m in models.values():m(rgb)
 torch.cuda.synchronize()
 for i in range(30):
  for name in (list(models) if i%2==0 else list(reversed(models))):
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();models[name](rgb);end.record();end.synchronize();samples[name].append(start.elapsed_time(end))
assert all(digest(m)==before[n] for n,m in models.items())
med={n:float(np.median(v)) for n,v in samples.items()}
result=dict(passed=True,batch_size=1,image_index=row['index'],warmup_per_model=5,repetitions=30,alternating_order=True,CUDA_event_median_ms=med,CUDA_event_p95_ms={n:float(np.quantile(v,.95)) for n,v in samples.items()},ratio=med['493B']/med['488B'],samples_ms=samples,note='Single fixed image GPU forward only, excludes load/IO/preprocessing; not dataset throughput or training speed',state_unchanged=True)
result.update(GPU=torch.cuda.get_device_name(0),torch_version=torch.__version__,cuda_version=torch.version.cuda,cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
(R/'INFERENCE_COST.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='samples_ms'}))

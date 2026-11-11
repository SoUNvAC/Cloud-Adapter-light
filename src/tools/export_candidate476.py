"""Export the measured B-off candidate without an implicit runtime switch."""
from pathlib import Path
import os,sys,json,hashlib
import numpy as np
import torch
from PIL import Image
R=Path('/home/scv/shared/phase476_20261107');D=R/'B/network_run';P=Path('/home/scv/shared/phase431_20261102')
sys.path[:0]=['/home/scv/shared/phase450_20261106',str(P/'code'),'/home/scv/shared/phase400_20261031/code']
os.chdir('/home/scv/Cloud-Adapter-light/src')
from infer_shadow476 import Frozen431WithOcclusion
from train_shadow450 import digest
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
register_all_modules();import phase431.integration
torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
off=json.loads((D/'ABLATION_OFF.json').read_text());step=off['choices']['best_shadow']
state=torch.load(D/f'branch_{step}.pth',map_location='cpu')
state['state_dict']={n:t for n,t in state['state_dict'].items() if not n.startswith('material.')}
state['candidate']='B learned decoder plus unchanged450 branch; material layer removed, no runtime toggle'
path=D/f'candidate_without_material_{step}.pth';torch.save(state,path)
cfg=Config.fromfile(str(P/'code/phase431/config.py'));cfg.model.backbone.variant='fused';cfg.model.backbone.repo='/home/scv/shared/phase429_20261102/dinov3';cfg.model.backbone.weights='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth'
teacher=MODELS.build(cfg.model);teacher.load_state_dict(torch.load(P/'fused/best_mIoU_iter_32000.pth',map_location='cpu')['state_dict'],strict=True)
loaded=teacher.load_state_dict(state['student_patch'],strict=False);assert not loaded.unexpected_keys and all(n.startswith('backbone.encoder.') for n in loaded.missing_keys)
net=Frozen431WithOcclusion(teacher,state['state_dict']).cuda().eval();assert not hasattr(net.branch,'material');before=digest(net)
rows=[r for r in json.loads(Path('/home/scv/shared/phase442_20261103/INPUTS.json').read_text()) if r['domain']=='A'];assert len(rows)==535
for row in rows:
 rgb=np.array(Image.open(row['image']).convert('RGB'))
 with torch.inference_mode():pred=net(torch.from_numpy(rgb).permute(2,0,1)[None].cuda())[0].argmax(1)[0].cpu().numpy()
 expected=np.load(D/'ablation_off'/f'step_{step}'/f"{row['index']:03d}.npz")['pred'];assert np.array_equal(pred,expected)
assert digest(net)==before
result=dict(passed=True,step=step,images=535,all_predictions_identical_to_B_off=True,material_parameters_removed=True,state_unchanged=True,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,metrics=off['results'][0],target_supervision_percent=0)
(R/'CANDIDATE_EXPORT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='metrics'}))

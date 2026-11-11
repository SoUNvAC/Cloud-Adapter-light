"""Read-only independent audit; invoked remotely through remote496.py."""
import sys,json,hashlib
from pathlib import Path
import torch
R=Path('/home/scv/shared/phase496_20261110')
sys.path.insert(0,str(R/'code'))
from phase496.texture import ShadowRedistribution
from phase496.integration import ExactIoU496
torch.set_num_threads(2)
audit={}
common=[]
for v in ['control','texture']:
    root=R/v
    initial=json.loads((root/'INITIAL.json').read_text())
    smoke=torch.load(R/('preflight_'+v)/'iter_10.pth',map_location='cpu')['state_dict']
    audit[v]={'trainable_parameters':initial['trainable_parameters'],'update_check':json.loads((root/'UPDATE_CHECK.json').read_text())}
    common.append(initial)
    locked=json.loads((root/'INPUT_LOCK.json').read_text())
    assert all(hashlib.sha256((R/'code/phase496'/n).read_bytes()).hexdigest()==h for n,h in locked['code_sha256'].items())
    if v=='texture':
        assert not any(k.startswith(('backbone.refinement.','backbone.local.')) for k in smoke)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42904)
            module=ShadowRedistribution(ordinal=False)
        changed={n:not torch.equal(p,smoke['backbone.texture_adapter.'+n]) for n,p in module.named_parameters()}
        assert all(changed.values()),changed
        audit[v]['all_texture_parameters_updated']=changed
        audit[v]['texture_parameters']=sum(p.numel() for p in module.parameters())
assert common[0]['common_sha']==common[1]['common_sha']
assert common[0]['frozen_sha']==common[1]['frozen_sha']
traces=[]
for v in ['control','texture']:
    traces.append({x['step']:x['batch_sha'] for x in map(json.loads,(R/v/'INPUT_TRACE.jsonl').read_text().splitlines())})
steps=sorted(set(traces[0])&set(traces[1]))
assert all(traces[0][s]==traces[1][s] for s in steps)
metric=ExactIoU496(iou_metrics=['mIoU'])
metric.dataset_meta={'classes':['surface','cloud','shadow']}
sample=tuple(torch.tensor(x,dtype=torch.float64) for x in ([2,3,4],[4,5,8],[3,4,6],[3,4,6]))
values=metric.compute_metrics([sample])
assert values['IoU_surface']==50 and values['IoU_shadow']==50 and values['IoU_cloud']==60
audit.update(passed=True,common_initialization_equal=True,input_trace_matched_steps=steps,exact_metric_test=True)
(R/'INDEPENDENT_AUDIT.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))

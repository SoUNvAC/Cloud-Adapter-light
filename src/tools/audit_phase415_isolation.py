"""CPU-only audit of residual optimizer coupling and checkpoint differences."""
from pathlib import Path
import json,torch
r=Path('/home/scv/shared/phase415_20261101')
assert (r/'EXIT_CODE').read_text()=='0'
a=torch.load(r/'aux_detached/work_dirs/formal/M/iter_1000.pth',map_location='cpu')['state_dict']
reference=torch.load('/home/scv/shared/phase411_20261101/layerwise/work_dirs/formal/M/best_mIoU_iter_1000.pth',map_location='cpu')
assert reference['phase400']['draw_cursor']==1000
b=reference['state_dict']
keys=[k for k in b if k.startswith('backbone.target_msre.')]
deltas={k:float((a[k]-b[k]).abs().max()) for k in keys}
cfg=json.loads((r/'aux_detached/configs/formal_M.json').read_text())
result={'common_parameter_tensors':len(keys),'exactly_equal_tensors':sum(v==0 for v in deltas.values()),'max_absolute_difference':max(deltas.values()),'per_tensor_max_difference':deltas,'optimizer':cfg['optim_wrapper'],'limitation':'detach blocks direct auxiliary gradient; global clipping still includes semantic head gradients, so optimizer isolation is incomplete'}
(r/'ISOLATION_AUDIT.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='per_tensor_max_difference'}))

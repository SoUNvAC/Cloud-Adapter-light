"""Read-only CPU checkpoint inspection, zero additional model forwards."""
from pathlib import Path
import json
from collect_phase241 import ssh
L=Path(__file__).resolve().parents[2]/'outputs/phase410/phase410_20261101'
SCRIPT=r"""
from pathlib import Path
import json,hashlib,torch
r=Path('/home/scv/shared/phase410_20261101')
assert all(json.loads((r/m/'COMPLETE.json').read_text())['passed'] for m in ['baseline','adaptive'])
import sys
sys.path[:0]=[str(r),'/home/scv/Cloud-Adapter-light/src','/home/scv/shared/phase400_20261031/code']
from phase410_msre import AdaptiveMsRE, SparseMsREResidual
from phase400_common import state_sha
torch.set_num_threads(4)
initial_checks={}
for mode,cls in [('baseline',SparseMsREResidual),('adaptive',AdaptiveMsRE)]:
 torch.manual_seed(4090101);model=cls()
 scope=json.loads((r/mode/'work_dirs/formal/M/ACTIVE_SCOPE.json').read_text())
 assert state_sha(model.state_dict())==scope['MsRE_initial_SHA']
 if mode=='adaptive':assert model.scale_gate.weight.count_nonzero()==0 and model.scale_gate.bias.count_nonzero()==0
 initial_checks[mode]=True
p=r/'adaptive';selection=json.loads((p/'SELECTION.json').read_text())
records=[]
for role,file in [('best',Path(selection['checkpoint'])),('final',p/'work_dirs/formal/M/iter_4000.pth')]:
 checkpoint=torch.load(file,map_location='cpu');assert checkpoint['phase400']['draw_cursor']==(selection['best']['step'] if role=='best' else 4000)
 state=checkpoint['state_dict']
 tensors={n:t for n,t in state.items() if n.startswith('backbone.target_msre.scale_gate.')}
 assert len(tensors)==2 and sum(t.numel() for t in tensors.values())==1155
 stats={}
 for n,t in tensors.items():
  assert torch.isfinite(t).all() and t.abs().sum()>0
  stats[n]={'shape':list(t.shape),'L2':t.norm().item(),'abs_max':t.abs().max().item(),'SHA':hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()}
 records.append({'role':role,'checkpoint':str(file),'parameters':1155,'tensors':stats})
print(json.dumps({'passed':True,'additional_model_forwards':0,'initial_state_reconstructed':initial_checks,'records':records}))
"""
if __name__=='__main__':
 data=ssh(SCRIPT).splitlines()[-1];v=json.loads(data);assert v['passed'];(L/'GATE_CHECKPOINT_AUDIT.json').write_bytes(data);print(data.decode())

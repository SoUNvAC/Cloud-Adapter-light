from pathlib import Path
import json,hashlib,torch
R=Path('/home/scv/shared/phase462_20261106');checks={}
for a in ['A','B']:
 d=R/a/'network_run';final=json.loads((d/'FINAL_RESULTS.json').read_text());step=final['choices']['best_shadow'];origin=torch.load(d/'branch_0.pth',map_location='cpu');state=torch.load(d/f'branch_{step}.pth',map_location='cpu')
 changed={n:not torch.equal(t,state['state_dict'][n]) for n,t in origin['state_dict'].items()}
 new={n:v for n,v in changed.items() if n.startswith('decomposition.')}
 assert any(changed.values())
 assert any(not torch.equal(origin['student_patch'][n],state['student_patch'][n]) for n in origin['student_patch'] if n.startswith('decode_head.'))
 fixed=[n for n in origin['student_patch'] if n.startswith('backbone.')]
 assert all(torch.equal(origin['student_patch'][n],state['student_patch'][n]) for n in fixed)
 files=[d/'branch_0.pth',d/f'branch_{step}.pth']
 checks[a]=dict(step=step,original_backbone_patch_unchanged=True,new_parameter_tensors=len(new),new_tensors_changed=sum(new.values()),sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(R/'TRAINED_WEIGHT_ACCEPTANCE.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))

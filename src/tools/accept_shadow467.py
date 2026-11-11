from pathlib import Path
import json,hashlib,torch
R=Path('/home/scv/shared/phase467_20261107');checks={}
source=Path('/home/scv/shared/phase431_20261102/fused/best_mIoU_iter_32000.pth')
source_sha=hashlib.sha256(source.read_bytes()).hexdigest();assert source_sha=='ffc0a9d6de9a25ae29fb0eb7b158c6d6d483d84fff6704309816336b34852205'
source_state=torch.load(source,map_location='cpu')['state_dict']
branch_path=Path('/home/scv/shared/phase450_20261106/network_run/branch_6000.pth')
branch_state=torch.load(branch_path,map_location='cpu')['state_dict']
checks['source_checkpoint_sha']=source_sha;checks['origin450_sha']=hashlib.sha256(branch_path.read_bytes()).hexdigest()
for a in ['A','B']:
 d=R/a/'network_run';final=json.loads((d/'FINAL_RESULTS.json').read_text());step=final['choices']['best_shadow'];origin=torch.load(d/'branch_0.pth',map_location='cpu');state=torch.load(d/f'branch_{step}.pth',map_location='cpu')
 assert all(torch.equal(t,source_state[n]) for n,t in origin['student_patch'].items())
 assert all(torch.equal(t,origin['state_dict'][n]) for n,t in branch_state.items())
 changed={n:not torch.equal(t,state['state_dict'][n]) for n,t in origin['state_dict'].items()}
 new={n:v for n,v in changed.items() if n.startswith('projection.')}
 assert all(not v for n,v in changed.items() if not n.startswith('projection.'))
 if a=='B':assert new and all(new.values())
 else:assert not new
 fixed=[n for n in origin['student_patch'] if not n.startswith('backbone.refinement.projector.')]
 assert all(torch.equal(origin['student_patch'][n],state['student_patch'][n]) for n in fixed)
 pc={n:not torch.equal(t,state['student_patch'][n]) for n,t in origin['student_patch'].items() if n.startswith('backbone.refinement.projector.')}
 assert len(pc)==2 and all(pc.values())
 checks[a]=dict(step=step,all_other_original_parameters_fixed=True,original450_branch_fixed=True,projector_changed=pc,new_changed=new,sha={f'branch_{s}.pth':hashlib.sha256((d/f'branch_{s}.pth').read_bytes()).hexdigest() for s in [0,step]})
traces={a:(R/a/'network_run/INPUT_TRACE.jsonl').read_text() for a in ['A','B']}
assert traces['A']==traces['B'];checks['identical_training_batches']=True
(R/'TRAINED_WEIGHT_ACCEPTANCE.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))

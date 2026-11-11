from pathlib import Path
import json,hashlib,torch
R=Path('/home/scv/shared/phase474_20261107');checks={};origins={}
source=Path('/home/scv/shared/phase431_20261102/fused/best_mIoU_iter_32000.pth')
source_sha=hashlib.sha256(source.read_bytes()).hexdigest();assert source_sha=='ffc0a9d6de9a25ae29fb0eb7b158c6d6d483d84fff6704309816336b34852205'
source_state=torch.load(source,map_location='cpu')['state_dict']
branch_path=Path('/home/scv/shared/phase450_20261106/network_run/branch_6000.pth')
branch_state=torch.load(branch_path,map_location='cpu')['state_dict']
checks['source_checkpoint_sha']=source_sha;checks['origin450_sha']=hashlib.sha256(branch_path.read_bytes()).hexdigest()
assert checks['origin450_sha']=='3051e08ea9733c82202e1425ffc52046132b149bd62a157e863ed156c95fc533'
for a in ['A','B']:
 d=R/a/'network_run';final=json.loads((d/'FINAL_RESULTS.json').read_text());step=final['choices']['best_shadow'];origin=torch.load(d/'branch_0.pth',map_location='cpu');state=torch.load(d/f'branch_{step}.pth',map_location='cpu')
 lock=json.loads((d/'LOCK.json').read_text());assert lock['active_task_names']==[] and lock['task_parameters']==0
 origins[a]=origin
 assert all(torch.equal(t,source_state[n]) for n,t in origin['student_patch'].items())
 assert all(torch.equal(t,origin['state_dict'][n]) for n,t in branch_state.items())
 changed={n:not torch.equal(t,state['state_dict'][n]) for n,t in origin['state_dict'].items()}
 new={n:v for n,v in changed.items() if n.startswith('decoder.') and not n.endswith('conditional_mode')}
 assert all(not v for n,v in changed.items() if not n.startswith('decoder.'))
 assert new and all(new.values())
 fixed=list(origin['student_patch'])
 assert all(torch.equal(origin['student_patch'][n],state['student_patch'][n]) for n in fixed)
 checks[a]=dict(step=step,original_trainable_parameters=0,new_trainable_parameters=sum(t.numel() for n,t in origin['state_dict'].items() if n.startswith('decoder.') and not n.endswith('conditional_mode')),all_original_parameters_fixed=True,original450_branch_fixed=True,new_changed=new,sha={f'branch_{s}.pth':hashlib.sha256((d/f'branch_{s}.pth').read_bytes()).hexdigest() for s in [0,step]})
shared=[n for n in origins['A']['state_dict'] if n.startswith('decoder.') and not n.endswith('conditional_mode')]
assert all(torch.equal(origins['A']['state_dict'][n],origins['B']['state_dict'][n]) for n in shared)
checks['actual_common_initialization_identical']=True
traces={a:(R/a/'network_run/INPUT_TRACE.jsonl').read_text() for a in ['A','B']}
assert traces['A']==traces['B'];checks['identical_training_batches']=True
checks['batch_trace_steps']=[json.loads(line)['step'] for line in traces['A'].splitlines()]
checks['batch_check_scope']='Fixed seed and sampler; identical recorded batch signatures at listed steps'
(R/'TRAINED_WEIGHT_ACCEPTANCE.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))

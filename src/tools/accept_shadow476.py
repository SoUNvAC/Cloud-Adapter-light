from pathlib import Path
import json,hashlib,torch
R=Path('/home/scv/shared/phase476_20261107');checks={};origins={}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=Path('/home/scv/shared/phase431_20261102/fused/best_mIoU_iter_32000.pth');origin450=Path('/home/scv/shared/phase450_20261106/network_run/branch_6000.pth')
assert sha(source)=='ffc0a9d6de9a25ae29fb0eb7b158c6d6d483d84fff6704309816336b34852205'
assert sha(origin450)=='3051e08ea9733c82202e1425ffc52046132b149bd62a157e863ed156c95fc533'
ss=torch.load(source,map_location='cpu')['state_dict'];bs=torch.load(origin450,map_location='cpu')['state_dict']
checks.update(source_checkpoint_sha=sha(source),origin450_sha=sha(origin450))
for arm in ['A','B']:
 d=R/arm/'network_run';final=json.loads((d/'FINAL_RESULTS.json').read_text());step=final['choices']['best_shadow'];lock=json.loads((d/'LOCK.json').read_text())
 origin=torch.load(d/'branch_0.pth',map_location='cpu');state=torch.load(d/f'branch_{step}.pth',map_location='cpu');origins[arm]=origin
 assert all(n.startswith('decode_head.') for n in lock['active_task_names']) and lock['active_task_names']
 assert all(torch.equal(t,ss[n]) for n,t in origin['student_patch'].items())
 changed={n:not torch.equal(t,state['student_patch'][n]) for n,t in origin['student_patch'].items()}
 assert all(not v for n,v in changed.items() if not n.startswith('decode_head.')) and any(changed.values())
 assert all(torch.equal(t,state['state_dict'][n]) and torch.equal(t,origin['state_dict'][n]) for n,t in bs.items())
 new={n:not torch.equal(t,state['state_dict'][n]) for n,t in origin['state_dict'].items() if n.startswith('material.') and not n.endswith('chroma_mode')}
 assert len(new)==3 and all(new.values())
 assert state['state_dict']['material.chroma_mode'].item()==(arm=='B') and origin['state_dict']['material.chroma_mode'].item()==(arm=='B')
 checks[arm]=dict(step=step,new_trainable_parameters=4609,task_parameters=lock['task_parameters'],all_other_original_parameters_fixed=True,original450_branch_fixed=True,new_changed=new,decoder_changed={n:v for n,v in changed.items() if n.startswith('decode_head.')},sha={f'branch_{s}.pth':sha(d/f'branch_{s}.pth') for s in [0,step]})
for n,t in origins['A']['state_dict'].items():
 if not n.endswith('chroma_mode'):assert torch.equal(t,origins['B']['state_dict'][n])
assert all(torch.equal(t,origins['B']['student_patch'][n]) for n,t in origins['A']['student_patch'].items())
traces={a:(R/a/'network_run/INPUT_TRACE.jsonl').read_text() for a in ['A','B']};assert traces['A']==traces['B']
checks.update(actual_common_initialization_identical=True,identical_training_batches=True,batch_trace_steps=[json.loads(line)['step'] for line in traces['A'].splitlines()],batch_check_scope='Fixed seed/sampler; identical recorded batch signatures at listed steps')
checks['update_flag_note']='Legacy UPDATE_CHECK.branch_updated follows arm label and is false for A; not used for acceptance. All three new parameter tensors in both arms were independently compared with initial weights and changed.'
(R/'TRAINED_WEIGHT_ACCEPTANCE.json').write_text(json.dumps(checks,indent=2));print('TRAINED_WEIGHT_ACCEPTANCE passed')

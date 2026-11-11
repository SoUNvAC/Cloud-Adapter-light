from pathlib import Path
import json,hashlib,torch
R=Path('/home/scv/shared/phase473_20261107');checks={};origins={}
source=Path('/home/scv/shared/phase431_20261102/fused/best_mIoU_iter_32000.pth')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)=='ffc0a9d6de9a25ae29fb0eb7b158c6d6d483d84fff6704309816336b34852205'
ss=torch.load(source,map_location='cpu')['state_dict'];bp=Path('/home/scv/shared/phase450_20261106/network_run/branch_6000.pth');bs=torch.load(bp,map_location='cpu')['state_dict']
assert sha(bp)=='3051e08ea9733c82202e1425ffc52046132b149bd62a157e863ed156c95fc533'
checks.update(source_checkpoint_sha=sha(source),origin450_sha=sha(bp))
allowed=('decode_head.','backbone.refinement.projector.')
for arm in ['A','B']:
 d=(Path('/home/scv/shared/phase472_20261107') if arm=='A' else R)/arm/'network_run';final=json.loads((d/'FINAL_RESULTS.json').read_text());step=final['choices']['best_shadow'];assert step>=2000
 origin=torch.load(d/'branch_0.pth',map_location='cpu');state=torch.load(d/f'branch_{step}.pth',map_location='cpu');origins[arm]=origin
 lock=json.loads((d/'LOCK.json').read_text());assert lock['active_task_names'] and all(n.startswith(allowed) for n in lock['active_task_names'])
 assert all(torch.equal(t,ss[n]) for n,t in origin['student_patch'].items())
 assert all(torch.equal(t,origin['state_dict'][n]) and torch.equal(t,state['state_dict'][n]) for n,t in bs.items())
 changed={n:not torch.equal(t,state['student_patch'][n]) for n,t in origin['student_patch'].items()}
 assert all(not v for n,v in changed.items() if not n.startswith(allowed))
 projector={n:v for n,v in changed.items() if n.startswith(allowed[1])};decoder={n:v for n,v in changed.items() if n.startswith(allowed[0])}
 assert len(projector)==2 and all(projector.values()) and any(decoder.values())
 new={n:not torch.equal(t,state['state_dict'][n]) for n,t in origin['state_dict'].items() if n.startswith('contrast.') and n!='contrast.strength'}
 if arm=='B':
  assert len(new)==6 and all(new.values())
  assert origin['state_dict']['contrast.strength'].item()==0 and state['state_dict']['contrast.strength'].item()==.25
 else:assert not new and set(state['state_dict'])==set(bs)
 checks[arm]=dict(step=step,task_parameters=lock['task_parameters'],all_other_original_parameters_fixed=True,original450_branch_fixed=True,projector_changed=projector,decoder_changed=decoder,new_changed=new,new_trainable_parameters=sum(t.numel() for n,t in origin['state_dict'].items() if n in new),sha={f'branch_{s}.pth':sha(d/f'branch_{s}.pth') for s in [0,step]})
assert all(torch.equal(t,origins['B']['student_patch'][n]) for n,t in origins['A']['student_patch'].items())
checks['actual_common_initialization_identical']=True
assert Path('/home/scv/shared/phase472_20261107/A/network_run/INPUT_TRACE.jsonl').read_bytes()==(R/'B/network_run/INPUT_TRACE.jsonl').read_bytes()
checks['identical_training_batches']=True
checks['batch_trace_steps']=[json.loads(line)['step'] for line in (R/'B/network_run/INPUT_TRACE.jsonl').read_text().splitlines()]
checks['batch_check_scope']='Fixed seed and sampler; byte-identical recorded batch signatures at listed steps'
(R/'TRAINED_WEIGHT_ACCEPTANCE.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))

from pathlib import Path
import json,hashlib,torch
R=Path('/home/scv/shared/phase495_20261109');P=Path('/home/scv/shared/phase493_20261109/B/network_run')
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in read(R/'INITIAL_CODE_SHA.json').items():assert sha(R/n)==h
source=Path('/home/scv/shared/phase431_20261102/fused/best_mIoU_iter_32000.pth');bp=Path('/home/scv/shared/phase450_20261106/network_run/branch_6000.pth')
assert sha(source)=='ffc0a9d6de9a25ae29fb0eb7b158c6d6d483d84fff6704309816336b34852205'
assert sha(bp)=='3051e08ea9733c82202e1425ffc52046132b149bd62a157e863ed156c95fc533'
ss=torch.load(source,map_location='cpu')['state_dict'];bs=torch.load(bp,map_location='cpu')['state_dict'];control=torch.load(P/'branch_0.pth',map_location='cpu');out={}
ia=torch.load(R/'A/network_run/branch_0.pth',map_location='cpu')['state_dict'];ib=torch.load(R/'B/network_run/branch_0.pth',map_location='cpu')['state_dict']
assert set(ia)==set(ib) and all(torch.equal(t,ib[n]) for n,t in ia.items() if n!='redistribution.feedback_slot')
for arm in ['A','B']:
 d=R/arm/'network_run';assert (R/f'{arm}_EXIT_CODE').read_text().strip()=='0' and (d.parent/'DONE').exists()
 lock=read(d/'LOCK.json');origin=torch.load(d/'branch_0.pth',map_location='cpu');active=set(lock['active_task_names'])
 assert lock['seed']==45201 and lock['steps']==6000 and lock['batch_size']==4 and lock['target_label_supervision_percent']==0
 assert lock['new_module']['init_seed']==48001 and lock['new_module']['shadow_fraction_auxiliary_weight']==0.
 assert not lock['new_module']['ordinal'] and lock['new_module']['feedback_slot']==(0 if arm=='A' else 3) and lock['new_module']['normalized']==False and lock['new_module']['texture_scales_pixels']==[12,28]
 assert all(n.startswith('decode_head.') for n in active)
 assert all(torch.equal(t,ss[n]) and torch.equal(t,control['student_patch'][n]) for n,t in origin['student_patch'].items())
 assert all(torch.equal(t,origin['state_dict'][n]) for n,t in control['state_dict'].items())
 assert len(set(origin['state_dict'])-set(control['state_dict']))==2
 assert not bool(origin['state_dict']['redistribution.contrast_mode']) and not bool(origin['state_dict']['redistribution.ordinal_mode']) and int(origin['state_dict']['redistribution.feedback_slot'])==(0 if arm=='A' else 3)
 extra={n:t for n,t in origin['state_dict'].items() if n not in bs};weights={}
 for step in sorted(set(read(d/'FINAL_RESULTS.json')['choices'].values())|{0}):
  p=d/f'branch_{step}.pth';s=torch.load(p,map_location='cpu')
  assert set(s['state_dict'])==set(origin['state_dict']) and set(s['student_patch'])==set(origin['student_patch'])
  assert all(torch.equal(t,s['state_dict'][n]) for n,t in bs.items())
  changed={n for n,t in origin['student_patch'].items() if not torch.equal(t,s['student_patch'][n])};assert changed<=active
  new_changed=[n for n,t in extra.items() if not torch.equal(t,s['state_dict'][n])]
  assert bool(s['state_dict']['redistribution.conserve_mode']) and not bool(s['state_dict']['redistribution.contrast_mode']) and not bool(s['state_dict']['redistribution.ordinal_mode']) and int(s['state_dict']['redistribution.feedback_slot'])==(0 if arm=='A' else 3)
  if step:assert changed and len(new_changed)==22
  weights[p.name]=dict(sha256=sha(p),changed_decoder_tensors=len(changed),changed_new_tensors=len(new_changed))
 assert (d/'INPUT_TRACE.jsonl').read_bytes()==(P/'INPUT_TRACE.jsonl').read_bytes()
 u=read(d/'UPDATE_CHECK.json');assert u['step']==6000 and all(u['new_changed'].values()) and u['feedback_calls']==1 and u['feedback_relative']>0
 assert all(u[k] for k in ['reference_unchanged','encoder_unchanged','BN_unchanged','frozen_parameters_unchanged'])
 out[arm]=dict(passed=True,weights=weights)
result=dict(passed=True,source_sha=sha(source),origin450_sha=sha(bp),initial_parameters_identical_to_493B=True,recorded_batches_identical_to_493B=True,training_code_unchanged=True,arms=out)
(R/'TRAINED_WEIGHT_ACCEPTANCE.json').write_text(json.dumps(result,indent=2));print('TRAINED_WEIGHT_ACCEPTANCE passed')

from pathlib import Path
import json,hashlib,torch
R=Path('/home/scv/shared/phase483_20261108');read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in read(R/'INITIAL_CODE_SHA.json').items():assert sha(R/n)==h,(n,'code changed')
refs=read(R/'PREFLIGHT.json')['refs']
for v in refs.values():
 for n,h in v['sha256'].items():assert sha(Path(v['directory'])/n)==h
source=Path('/home/scv/shared/phase431_20261102/fused/best_mIoU_iter_32000.pth');base=Path('/home/scv/shared/phase450_20261106/network_run/branch_6000.pth')
assert sha(source)=='ffc0a9d6de9a25ae29fb0eb7b158c6d6d483d84fff6704309816336b34852205'
assert sha(base)=='3051e08ea9733c82202e1425ffc52046132b149bd62a157e863ed156c95fc533'
ss=torch.load(source,map_location='cpu')['state_dict'];bs=torch.load(base,map_location='cpu')['state_dict'];out={};traces=[]
for seed in [45202,45203]:
 d=R/f'seed_{seed}'/'B/network_run';p=Path(refs[f'{seed}/S']['directory']);control=torch.load(p/'branch_0.pth',map_location='cpu')
 assert (d.parent/'DONE').exists() and (R/f'seed_{seed}_EXIT_CODE').read_text().strip()=='0'
 lock=read(d/'LOCK.json');assert lock['seed']==seed and lock['steps']==6000 and lock['batch_size']==4 and lock['target_label_supervision_percent']==0
 assert lock['new_module']['init_seed']==48000+seed-45200 and lock['new_module']['shadow_fraction_auxiliary_weight']==0
 origin=torch.load(d/'branch_0.pth',map_location='cpu');active=set(lock['active_task_names']);assert all(n.startswith('decode_head.') for n in active)
 assert all(torch.equal(t,ss[n]) and torch.equal(t,control['student_patch'][n]) for n,t in origin['student_patch'].items())
 assert all(torch.equal(t,origin['state_dict'][n]) for n,t in control['state_dict'].items())
 assert set(origin['state_dict'])==set(control['state_dict'])|{'redistribution.contrast_mode'}
 assert not bool(origin['state_dict']['redistribution.contrast_mode'])
 extra={n:t for n,t in origin['state_dict'].items() if n not in bs};weights={}
 for step in sorted(set(read(d/'FINAL_RESULTS.json')['choices'].values())|{0}):
  f=d/f'branch_{step}.pth';s=torch.load(f,map_location='cpu')
  assert set(s['state_dict'])==set(origin['state_dict']) and set(s['student_patch'])==set(origin['student_patch'])
  assert all(torch.equal(t,s['state_dict'][n]) for n,t in bs.items())
  changed={n for n,t in origin['student_patch'].items() if not torch.equal(t,s['student_patch'][n])};assert changed<=active
  new=[n for n,t in extra.items() if not torch.equal(t,s['state_dict'][n])]
  assert bool(s['state_dict']['redistribution.conserve_mode']) and not bool(s['state_dict']['redistribution.contrast_mode'])
  if step:assert changed and len(new)==13
  weights[f.name]=dict(sha256=sha(f),changed_decoder_tensors=len(changed),changed_new_tensors=len(new))
 trace=(d/'INPUT_TRACE.jsonl').read_bytes();assert trace==(p/'INPUT_TRACE.jsonl').read_bytes();traces.append(trace)
 u=read(d/'UPDATE_CHECK.json');assert u['step']==6000 and all(u['new_changed'].values())
 assert all(u[k] for k in ['reference_unchanged','encoder_unchanged','BN_unchanged','frozen_parameters_unchanged'])
 out[str(seed)]=dict(passed=True,weights=weights)
assert traces[0]!=traces[1]
(R/'TRAINED_WEIGHT_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,source_sha=sha(source),origin450_sha=sha(base),matched_seed_initial_parameters=True,matched_recorded_batches=True,frozen_states_verified=True,training_code_unchanged=True,historical_references_unchanged=True,runs=out),indent=2));print('TRAINED_WEIGHT_ACCEPTANCE passed')

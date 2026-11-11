from pathlib import Path,PureWindowsPath
import json, hashlib, torch
R=Path('/home/scv/shared/phase481_20261108')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=Path('/home/scv/shared/phase431_20261102/fused/best_mIoU_iter_32000.pth')
base=Path('/home/scv/shared/phase450_20261106/network_run/branch_6000.pth')
assert sha(source)=='ffc0a9d6de9a25ae29fb0eb7b158c6d6d483d84fff6704309816336b34852205'
assert sha(base)=='3051e08ea9733c82202e1425ffc52046132b149bd62a157e863ed156c95fc533'
for n,h in read(R/'INITIAL_CODE_SHA.json').items():assert sha(R/PureWindowsPath(n).name)==h,(n,'code changed')
for a,v in read(R/'PREFLIGHT.json')['sources'].items():
 for n,h in v['sha256'].items():assert sha(Path(v['directory'])/n)==h
ss=torch.load(source,map_location='cpu')['state_dict'];bs=torch.load(base,map_location='cpu')['state_dict']
out={};seed_initial=[];traces=[]
for seed in [45202,45203]:
 common=None;trace=None
 for arm in ['C','A','B']:
  d=R/f'seed_{seed}'/arm/'network_run';assert (d.parent/'DONE').exists()
  assert (R/f'{arm}_EXIT_CODE').read_text().strip()=='0'
  assert not (d/'FAILURE.json').exists()
  done=read(d/'COMPLETE.json');assert done['passed'] and done['steps']==6000
  lock=read(d/'LOCK.json');assert lock['seed']==seed and lock['target_label_supervision_percent']==0
  assert lock['batch_size']==4 and lock['steps']==6000 and lock['task_lr']==1e-5
  active=set(lock['active_task_names']);assert all(n.startswith('decode_head.') for n in active)
  initial=torch.load(d/'branch_0.pth',map_location='cpu')
  assert all(torch.equal(t,ss[n]) for n,t in initial['student_patch'].items())
  assert all(torch.equal(t,initial['state_dict'][n]) for n,t in bs.items())
  extra={n:t for n,t in initial['state_dict'].items() if n not in bs}
  if arm=='C':assert not extra
  else:
   assert all(n.startswith('redistribution.') for n in extra)
   assert bool(extra['redistribution.conserve_mode'])==(arm=='B')
   if common is not None:assert all(torch.equal(t,common[n]) for n,t in extra.items() if n!='redistribution.conserve_mode')
   else:common=extra;seed_initial.append(extra)
  t=(d/'INPUT_TRACE.jsonl').read_bytes()
  if trace is not None:assert trace==t
  trace=t
  final=read(d/'FINAL_RESULTS.json');assert final['passed'] and final['target_label_supervision_percent']==0
  weights={}
  for step in sorted(set(final['choices'].values())|{0}):
   p=d/f'branch_{step}.pth';s=torch.load(p,map_location='cpu')
   assert set(s['state_dict'])==set(initial['state_dict'])
   assert all(torch.equal(t,s['state_dict'][n]) for n,t in bs.items())
   assert set(s['student_patch'])==set(initial['student_patch'])
   changed={n for n,t in initial['student_patch'].items() if not torch.equal(t,s['student_patch'][n])}
   assert changed<=active
   new=[n for n,t in extra.items() if not torch.equal(t,s['state_dict'][n])]
   if step:assert changed and (len(new)==13 if arm!='C' else not new)
   weights[p.name]=dict(sha256=sha(p),changed_decoder_tensors=len(changed),changed_new_tensors=len(new))
  u=read(d/'UPDATE_CHECK.json');assert u['step']==6000
  assert all(u[k] for k in ['reference_unchanged','encoder_unchanged','BN_unchanged','frozen_parameters_unchanged'])
  if arm!='C':assert len(u['new_changed'])==13 and all(u['new_changed'].values())
  out[f'{seed}/{arm}']=dict(passed=True,weights=weights,trace_sha=hashlib.sha256(t).hexdigest())
 traces.append(trace)
assert traces[0]!=traces[1]
assert any(not torch.equal(seed_initial[0][n],seed_initial[1][n]) for n in seed_initial[0] if n!='redistribution.conserve_mode')
result=dict(passed=True,source_sha=sha(source),origin450_sha=sha(base),seed0_provenance_unchanged=True,training_code_unchanged=True,paired_batches_equal=True,initial_branches_equal_within_seed=True,initial_branches_and_batches_differ_between_seeds=True,runs=out)
(R/'TRAINED_WEIGHT_ACCEPTANCE.json').write_text(json.dumps(result,indent=2))
print('TRAINED_WEIGHT_ACCEPTANCE passed')

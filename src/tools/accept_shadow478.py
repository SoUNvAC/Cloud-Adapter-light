from pathlib import Path
import json,hashlib,torch
R=Path('/home/scv/shared/phase478_20261108');P=Path('/home/scv/shared/phase477_20261107/A/network_run')
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=Path('/home/scv/shared/phase431_20261102/fused/best_mIoU_iter_32000.pth');bp=Path('/home/scv/shared/phase450_20261106/network_run/branch_6000.pth')
assert sha(source)=='ffc0a9d6de9a25ae29fb0eb7b158c6d6d483d84fff6704309816336b34852205'
assert sha(bp)=='3051e08ea9733c82202e1425ffc52046132b149bd62a157e863ed156c95fc533'
ss=torch.load(source,map_location='cpu')['state_dict'];bs=torch.load(bp,map_location='cpu')['state_dict'];control=torch.load(P/'branch_0.pth',map_location='cpu')
out={}
for arm in ['A','B']:
 d=R/arm/'network_run';lock=read(d/'LOCK.json');origin=torch.load(d/'branch_0.pth',map_location='cpu');active=set(lock['active_task_names'])
 assert lock['new_module'] is None and lock['target_label_supervision_percent']==0
 assert all(torch.equal(t,ss[n]) and torch.equal(t,control['student_patch'][n]) for n,t in origin['student_patch'].items())
 assert set(bs)==set(origin['state_dict']) and all(torch.equal(t,origin['state_dict'][n]) for n,t in bs.items())
 allowed=['decode_head.']+lock['trainable_path'];assert all(n.startswith(tuple(allowed)) for n in active)
 weights={}
 for step in sorted(set(read(d/'FINAL_RESULTS.json')['choices'].values())|{0}):
  p=d/f'branch_{step}.pth';s=torch.load(p,map_location='cpu')
  assert set(s['state_dict'])==set(bs) and all(torch.equal(t,s['state_dict'][n]) for n,t in bs.items())
  assert set(s['student_patch'])==set(origin['student_patch'])
  changed={n for n,t in origin['student_patch'].items() if not torch.equal(t,s['student_patch'][n])}
  assert changed<=active
  counts={prefix:sum(n.startswith(prefix) for n in changed) for prefix in allowed}
  if step:assert all(counts.values())
  norms={}
  for prefix in allowed:
   names=[n for n in active if n.startswith(prefix)]
   initial_sq=sum(float(origin['student_patch'][n].double().square().sum()) for n in names)
   delta_sq=sum(float((s['student_patch'][n].double()-origin['student_patch'][n].double()).square().sum()) for n in names)
   norms[prefix]=dict(initial_L2=initial_sq**.5,change_L2=delta_sq**.5,relative_L2=(delta_sq/max(initial_sq,1e-30))**.5)
  weights[p.name]=dict(sha256=sha(p),changed_tensors=counts,parameter_changes=norms)
 assert (d/'INPUT_TRACE.jsonl').read_bytes()==(P/'INPUT_TRACE.jsonl').read_bytes()
 u=read(d/'UPDATE_CHECK.json');assert u['step']==6000
 assert all(u[k] for k in ['reference_unchanged','encoder_unchanged','BN_unchanged','frozen_parameters_unchanged'])
 out[arm]=dict(passed=True,task_parameters=lock['task_parameters'],allowed=allowed,weights=weights)
result=dict(passed=True,source_sha=sha(source),origin450_sha=sha(bp),new_parameters=0,original450_branch_fixed=True,matches_477_common_initialization_and_recorded_batches=True,arms=out)
(R/'TRAINED_WEIGHT_ACCEPTANCE.json').write_text(json.dumps(result,indent=2));print('TRAINED_WEIGHT_ACCEPTANCE passed')

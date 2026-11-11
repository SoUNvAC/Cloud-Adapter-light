from pathlib import Path
import json,torch
R=Path('/home/scv/shared/phase495_20261109');old=torch.load('/home/scv/shared/phase493_20261109/B/network_run/branch_0.pth',map_location='cpu');rows={}
for arm in ['A','B']:
 d=R/arm/'network_run'
 if (d/'FAILURE.json').exists():raise RuntimeError((d/'FAILURE.json').read_text())
 x=torch.load(d/'branch_0.pth',map_location='cpu')
 assert all(torch.equal(t,x['state_dict'][n]) for n,t in old['state_dict'].items())
 assert all(torch.equal(t,x['student_patch'][n]) for n,t in old['student_patch'].items())
 assert len(set(x['state_dict'])-set(old['state_dict']))==2
 u=json.loads((d/'UPDATE_CHECK.json').read_text())
 assert u['step']>=10 and u['feedback_calls']==1 and u['feedback_relative']>0
 assert all(u['new_changed'].values()) and all(u[k] for k in ['reference_unchanged','encoder_unchanged','BN_unchanged','frozen_parameters_unchanged'])
 rows[arm]={k:u[k] for k in ['step','feedback_calls','feedback_relative','peak_memory_bytes']}
result=dict(passed=True,initial_parameters_identical_to_493B=True,real_feedback_gradient_reached=True,original_weights_frozen=True,runs=rows)
(R/'EARLY_ROUTING_ACCEPTANCE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

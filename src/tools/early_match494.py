from pathlib import Path
import json,torch
R=Path('/home/scv/shared/phase494_20261109');records=[]
for seed in [45202,45203]:
 a=torch.load(R/f'seed_{seed}/B/network_run/branch_0.pth',map_location='cpu')
 b=torch.load(Path('/home/scv/shared/phase489_20261108')/f'seed_{seed}/B/network_run/branch_0.pth',map_location='cpu')
 assert all(torch.equal(t,a['state_dict'][n]) for n,t in b['state_dict'].items())
 assert set(a['student_patch'])==set(b['student_patch'])
 assert all(torch.equal(t,a['student_patch'][n]) for n,t in b['student_patch'].items())
 assert len(set(a['state_dict'])-set(b['state_dict']))==5
 records.append(seed)
(R/'EARLY_WEIGHT_MATCH.json').write_text(json.dumps(dict(passed=True,seeds=records,common_initial_weights_identical=True),indent=2));print('EARLY_WEIGHT_MATCH passed')

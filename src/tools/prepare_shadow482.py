from pathlib import Path
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
names=['train_shadow480.py','infer_shadow480.py','verify_shadow480.py','ablate_shadow480.py','boundary_shadow480.py','boundary_effect480.py','collect_shadow480.py','verify_archive480.py','status_shadow480.py','audit_shadow480.py']
for n in names:
 s=(T/n).read_text(encoding='utf-8').replace('480','482')
 if n=='train_shadow480.py':
  s=s.replace('torch.manual_seed(48201);branch.redistribution=ShadowRedistribution(conserve=ARM==\'B\').cuda()',"torch.manual_seed(48001);branch.redistribution=ShadowRedistribution(conserve=True,contrast=ARM=='A').cuda()")
  s=s.replace("+.1*branch.redistribution.auxiliary(y)","+(.1 if ARM=='A' else 0.)*branch.redistribution.auxiliary(y)")
  s=s.replace("conserve=ARM=='B',", "conserve=True,contrast=ARM=='A',")
  s=s.replace("shadow_fraction_auxiliary_weight=.1", "shadow_fraction_auxiliary_weight=(.1 if ARM=='A' else 0.)")
  s=s.replace("init_seed=48201", "init_seed=48001")
  s=s.replace('new_module=dict(conserve=', 'new_module=dict(init_seed=48001,conserve=')
  assert 'torch.manual_seed(48001);' in s
 if n=='infer_shadow480.py':
  s=s.replace("conserve=bool(branch_state['redistribution.conserve_mode']))", "conserve=bool(branch_state['redistribution.conserve_mode']),contrast=bool(branch_state.get('redistribution.contrast_mode',False)))")
 if n=='audit_shadow480.py':continue
 if n=='verify_archive480.py':
  s=s.replace("['RESULT.md','MAIN.csv']", "['RESULT.md','MAIN.csv','LOCAL_ACCEPTANCE.json']")
 put(n.replace('480','482'),s)
s=(T/'shadow_redistribution480.py').read_text(encoding='utf-8')
s=s.replace('def __init__(self,conserve=True):','def __init__(self,conserve=True,contrast=False):')
s=s.replace("self.detail=nn.Conv2d", "self.register_buffer('contrast_mode',torch.tensor(contrast,dtype=torch.bool));self.contrast_enabled=True\n  self.detail=nn.Conv2d")
s=s.replace('detail_code=z*self.shadow_logits.sigmoid()',"p=self.shadow_logits.sigmoid()\n  if bool(self.contrast_mode) and self.contrast_enabled:p=p-F.interpolate(F.avg_pool2d(p,4,4),scale_factor=4,mode='nearest')\n  detail_code=z*p")
put('shadow_redistribution482.py',s)
s=(T/'ablate_shadow482.py').read_text(encoding='utf-8')
s=s.replace('setattr(network.branch.redistribution,"enabled",False)', 'setattr(network.branch.redistribution,"contrast_enabled",False)')
assert '"contrast_enabled",False' in s
s=s.replace('OFF_', 'CONTRAST_OFF_').replace("'ablation_off'", "'contrast_off'").replace('ABLATION_OFF.json','CONTRAST_OFF.json')
put('ablate_contrast482.py',s)
put('run_shadow482.sh', '''#!/bin/bash
set -eu
cd /home/scv/shared/phase482_20261108
ARM="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "${ARM}_EXIT_CODE"' EXIT
$PY train_shadow482.py "$ARM"
$PY verify_shadow482.py "$ARM"
$PY boundary_shadow482.py "$ARM"
$PY ablate_shadow482.py "$ARM"
$PY boundary_effect482.py "$ARM"
if [ "$ARM" = A ]; then $PY ablate_contrast482.py "$ARM"; fi
touch "${ARM}/DONE"
''')

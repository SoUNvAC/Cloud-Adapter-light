from pathlib import Path
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
for n in ['train_shadow482.py','infer_shadow482.py','verify_shadow482.py','ablate_shadow482.py','boundary_shadow482.py','boundary_effect482.py','verify_archive482.py','status_shadow482.py']:
 s=(T/n).read_text(encoding='utf-8').replace('482','484')
 if n=='train_shadow482.py':
  s=s.replace("ShadowRedistribution(conserve=True,contrast=ARM=='A')", "ShadowRedistribution(conserve=True,contrast=False,regional_conserve=ARM=='A')")
  s=s.replace("+(.1 if ARM=='A' else 0.)*branch.redistribution.auxiliary(y)", '')
  s=s.replace("conserve=True,contrast=ARM=='A',", "conserve=True,contrast=False,regional_conserve=ARM=='A',regional_support_pixels=64,")
  s=s.replace("shadow_fraction_auxiliary_weight=(.1 if ARM=='A' else 0.)", "shadow_fraction_auxiliary_weight=0.")
 if n=='infer_shadow482.py':
  s=s.replace("contrast=bool(branch_state.get('redistribution.contrast_mode',False)))", "contrast=bool(branch_state.get('redistribution.contrast_mode',False)),regional_conserve=bool(branch_state.get('redistribution.regional_conserve_mode',True)))")
 put(n.replace('482','484'),s)
s=(T/'shadow_redistribution482.py').read_text(encoding='utf-8')
s=s.replace('def __init__(self,conserve=True,contrast=False):', 'def __init__(self,conserve=True,contrast=False,regional_conserve=True):')
s=s.replace('self.enabled=True;', "self.register_buffer('regional_conserve_mode',torch.tensor(regional_conserve,dtype=torch.bool))\n  self.regional=nn.Sequential(nn.Conv2d(16,16,5,padding=2,groups=16),nn.GELU(),nn.Conv2d(16,16,1),nn.GELU())\n  self.regional_inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.regional_inject.weight)\n  self.regional_enabled=True;self.regional_relative_change=None;self.regional_mean_residual=None\n  self.enabled=True;")
s=s.replace('detail_code=z*p', 'detail_code=z*p\n  pooled=F.avg_pool2d(detail_code,4,4)')
s=s.replace('delta=self.inject(detail_code)', '''delta=self.inject(detail_code)
  if self.regional_enabled:
   field=self.regional(pooled)
   field=F.interpolate(field,size=detail.shape[-2:],mode='bilinear',align_corners=False)
   if bool(self.regional_conserve_mode):field=field-F.interpolate(F.avg_pool2d(field,16,16),scale_factor=16,mode='nearest')
   regional_delta=self.regional_inject(field)
   self.regional_relative_change=(regional_delta.square().mean()/(base.square().mean()+1e-12)).sqrt().detach()
   self.regional_mean_residual=F.avg_pool2d(regional_delta.detach(),16,16).abs().max()
   delta=delta+regional_delta''')
put('shadow_redistribution484.py',s)
s=(T/'ablate_shadow484.py').read_text(encoding='utf-8').replace('setattr(network.branch.redistribution,"enabled",False)', 'setattr(network.branch.redistribution,"regional_enabled",False)')
assert '"regional_enabled",False' in s
s=s.replace('OFF_', 'REGION_OFF_').replace("'ablation_off'", "'region_off'").replace('ABLATION_OFF.json','REGION_OFF.json')
put('ablate_region484.py',s)
put('run_shadow484.sh','''#!/bin/bash
set -eu
cd /home/scv/shared/phase484_20261108
ARM="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "${ARM}_EXIT_CODE"' EXIT
$PY train_shadow484.py "$ARM"
$PY verify_shadow484.py "$ARM"
$PY boundary_shadow484.py "$ARM"
$PY ablate_shadow484.py "$ARM"
$PY boundary_effect484.py "$ARM"
$PY ablate_region484.py "$ARM"
touch "${ARM}/DONE"
''')

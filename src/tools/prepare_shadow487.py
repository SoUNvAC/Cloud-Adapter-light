from pathlib import Path
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
for n in ['train_shadow484.py','infer_shadow484.py','verify_shadow484.py','ablate_shadow484.py','boundary_shadow484.py','boundary_effect484.py','verify_archive484.py','status_shadow484.py','run_shadow484.sh','collect_all484.py','compare_shadow484.py','finish_shadow484.py','boundary_region484.py','ablate_region484.py']:
 s=(T/n).read_text(encoding='utf-8').replace('484','487').replace('regional_enabled','evidence_enabled')
 if n=='train_shadow484.py':
  s=s.replace("regional_conserve=ARM=='A'","shift_pixels=8 if ARM=='A' else 16").replace('regional_support_pixels=64,','extra_frozen_backbone_pass=True,common_border_pixels=16,')
  s=s.replace("calls+=len(rgb);logits", "calls+=2*len(rgb);logits")
 if n=='infer_shadow484.py':s=s.replace("regional_conserve=bool(branch_state.get('redistribution.regional_conserve_mode',True))","shift_pixels=int(branch_state.get('redistribution.shift_mode',8))")
 put(n.replace('484','487'),s)
s=(T/'shadow_redistribution482.py').read_text(encoding='utf-8')
s=s.replace('def __init__(self,conserve=True,contrast=False):','def __init__(self,conserve=True,contrast=False,shift_pixels=8):')
s=s.replace('self.enabled=True;', '''assert shift_pixels in [8,16]
  self.register_buffer('shift_mode',torch.tensor(shift_pixels,dtype=torch.int64))
  self.phase_reduce=nn.Conv2d(384,16,1,bias=False)
  self.phase_mix=nn.Sequential(nn.Conv2d(16,16,3,padding=1,groups=16,bias=False),nn.GELU(),nn.Conv2d(16,16,1,bias=False),nn.GELU())
  self.phase_inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.phase_inject.weight)
  self.evidence_enabled=True;self.auxiliary=None
  self.enabled=True;''')
s=s.replace('delta=self.inject(detail_code)', '''delta=self.inject(detail_code)
  if self.evidence_enabled:
   assert self.auxiliary is not None and self.auxiliary.shape==base.shape
   difference=self.auxiliary-base
   field=self.phase_mix(self.phase_reduce(difference))
   extra=self.phase_inject(field)
   # Same 16px exclusion in both arms; padded view edges never inject.
   valid=torch.ones_like(extra[:,:1]);valid[:,:,:4]=0;valid[:,:,-4:]=0;valid[:,:,:,:4]=0;valid[:,:,:,-4:]=0
   delta=delta+extra*valid''')
s=s.replace('def install(backbone,module):','''def align_shifted(feature,shift_cells):
 return F.pad(feature[:,:,shift_cells:,shift_cells:],(0,shift_cells,0,shift_cells),mode='replicate')

def install(backbone,module):
 def second_view(m,args):
  if not module.enabled or not module.evidence_enabled:return
  image=args[0];s=int(module.shift_mode);h,w=image.shape[-2:]
  shifted=F.pad(image,(s,0,s,0),mode='replicate')[:,:,:h,:w]
  # Direct forward bypasses this module's pre/post hooks. Submodule captures
  # are overwritten by the subsequent original-view forward before fusion.
  with torch.no_grad():other=m.forward(shifted)[0]
  module.auxiliary=align_shifted(other,s//4).detach()
 backbone.register_forward_pre_hook(second_view)''')
put('shadow_redistribution487.py',s)
s=(T/'accept_shadow484.py').read_text(encoding='utf-8').replace('484','487').replace('regional_conserve_mode','shift_mode').replace("['regional_conserve']==(arm=='A') and lock['new_module']['regional_support_pixels']==64","['shift_pixels']==(8 if arm=='A' else 16) and lock['new_module']['common_border_pixels']==16")
s=s.replace("bool(origin['state_dict']['redistribution.shift_mode'])==(arm=='A')", "int(origin['state_dict']['redistribution.shift_mode'])==(8 if arm=='A' else 16)")
s=s.replace("bool(s['state_dict']['redistribution.shift_mode'])==(arm=='A')", "int(s['state_dict']['redistribution.shift_mode'])==(8 if arm=='A' else 16)")
s=s.replace('))==6','))==5').replace('len(new_changed)==18','len(new_changed)==17')
put('accept_shadow487.py',s)

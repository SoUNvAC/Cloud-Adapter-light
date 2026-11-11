from pathlib import Path
import re
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
for n in ['train_shadow490.py','infer_shadow490.py','verify_shadow490.py','ablate_shadow490.py','boundary_shadow490.py','boundary_effect490.py','verify_archive490.py','status_shadow490.py','run_shadow490.sh','collect_all490.py','compare_shadow490.py','finish_shadow490.py','boundary_region490.py','ablate_region490.py','accept_shadow490.py','cost_shadow490.py']:
 s=re.sub(r'(?<!\d)490(?!\d)','491',(T/n).read_text(encoding='utf-8')).replace('directional','cross_depth').replace('default_rng(49002)','default_rng(49102)')
 if n=='train_shadow490.py':s=s.replace('neighbor_offsets_pixels=[16,32],','depth_slots=[0,3],')
 if n=='accept_shadow490.py':s=s.replace('))==5','))==6').replace('len(new_changed)==21','len(new_changed)==22')
 put(n.replace('490','491'),s)
s=(T/'shadow_redistribution488.py').read_text(encoding='utf-8')
s=s.replace('normalized=True):','normalized=False,cross_depth=True):')
s=s.replace('self.enabled=True;', '''self.register_buffer('cross_depth_mode',torch.tensor(cross_depth,dtype=torch.bool))
  self.shallow_project=nn.Conv2d(384,16,1,bias=False)
  self.deep_project=nn.Conv2d(384,16,1,bias=False)
  self.depth_mix=nn.Sequential(nn.Conv2d(32,16,1,bias=False),nn.GELU(),nn.Conv2d(16,16,3,padding=1,groups=16,bias=False),nn.GELU())
  self.depth_inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.depth_inject.weight)
  self.deep=None;self.cross_depth_enabled=True
  self.enabled=True;''')
s=s.replace('delta=delta+self.evidence_inject(field)', '''delta=delta+self.evidence_inject(field)
   if self.cross_depth_enabled:
    assert self.deep is not None and self.deep.shape==native.shape
    normalize=lambda x:F.layer_norm(x.permute(0,2,3,1),(384,)).permute(0,3,1,2)
    shallow=self.shallow_project(normalize(native));deep=self.deep_project(normalize(self.deep))
    if bool(self.cross_depth_mode):cue=torch.cat([shallow-deep,shallow*deep],1)
    else:cue=torch.cat([shallow,deep],1)
    field=F.interpolate(self.depth_mix(cue),size=detail.shape[-2:],mode='bilinear',align_corners=False)
    delta=delta+self.depth_inject(field)''')
s=s.replace("if args[1]==0:module.native=out", "if args[1]==0:module.native=out\n  if args[1]==3:module.deep=out")
put('shadow_redistribution491.py',s)

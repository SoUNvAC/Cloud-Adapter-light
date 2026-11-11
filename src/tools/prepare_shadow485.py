from pathlib import Path
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
for n in ['train_shadow484.py','infer_shadow484.py','verify_shadow484.py','ablate_shadow484.py','boundary_shadow484.py','boundary_effect484.py','verify_archive484.py','status_shadow484.py','run_shadow484.sh','collect_all484.py','compare_shadow484.py','finish_shadow484.py','boundary_region484.py','ablate_region484.py']:
 s=(T/n).read_text(encoding='utf-8').replace('484','485').replace('regional_enabled','evidence_enabled')
 if n=='train_shadow484.py':
  s=s.replace("regional_conserve=ARM=='A'","ordered=ARM=='A'").replace("regional_support_pixels=64,","profile_offsets_pixels=[-32,-24,-16,-8,0,8,16,24,32],")
 if n=='infer_shadow484.py':s=s.replace("regional_conserve=bool(branch_state.get('redistribution.regional_conserve_mode',True))","ordered=bool(branch_state.get('redistribution.ordered_mode',True))")
 put(n.replace('484','485'),s)
s=(T/'shadow_redistribution482.py').read_text(encoding='utf-8')
s=s.replace('def __init__(self,conserve=True,contrast=False):','def __init__(self,conserve=True,contrast=False,ordered=True):')
s=s.replace('self.enabled=True;', '''self.register_buffer('ordered_mode',torch.tensor(ordered,dtype=torch.bool))
  self.profile=nn.Sequential(nn.Conv3d(6,8,(3,1,1),padding=(1,0,0),bias=False),nn.GELU(),nn.Conv3d(8,8,(3,1,1),padding=(1,0,0),bias=False),nn.GELU())
  self.profile_fuse=nn.Sequential(nn.Conv2d(72,16,1,bias=False),nn.GELU())
  self.evidence_inject=nn.Conv2d(32,384,1,bias=False);nn.init.zeros_(self.evidence_inject.weight)
  self.evidence_enabled=True;self.rgb=None
  self.enabled=True;''')
s=s.replace('delta=self.inject(detail_code)', '''delta=self.inject(detail_code)
  if self.evidence_enabled:
   field=self.profile_evidence(self.rgb)
   field=F.interpolate(field,size=detail.shape[-2:],mode='bilinear',align_corners=False)
   delta=delta+self.evidence_inject(field)''')
pos=s.index(' def auxiliary(')
s=s[:pos]+''' def profile_evidence(self,rgb):
  assert rgb is not None
  rgb=F.avg_pool2d(rgb,8,8);logrgb=(rgb+.02).log()
  lum=(rgb*rgb.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1,keepdim=True)
  cue=torch.cat([logrgb,(lum+.02).log(),logrgb[:,0:1]-logrgb[:,1:2],logrgb[:,2:3]-logrgb[:,1:2]],1)
  b,c,h,w=cue.shape;padded=F.pad(cue,(4,4,4,4),mode='replicate');fields=[]
  for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
   samples=torch.stack([padded[:,:,4+s*dy:4+s*dy+h,4+s*dx:4+s*dx+w] for s in range(-4,5)],2)
   samples=samples-cue[:,:,None]
   if not bool(self.ordered_mode):samples=samples[:,:,[4,0,8,2,6,1,7,3,5]]
   enc=self.profile(samples).reshape(b,72,h,w);fields.append(self.profile_fuse(enc))
  fields=torch.stack(fields,1)
  return torch.cat([fields.mean(1),fields.max(1).values],1)
''' +s[pos:]
s=s.replace('def install(backbone,module):','''def install(backbone,module):
 def rgb_hook(m,args):
  x=args[0];mean=x.new_tensor([123.675,116.28,103.53])[None,:,None,None];std=x.new_tensor([58.395,57.12,57.375])[None,:,None,None]
  module.rgb=((x*std+mean)/255).clamp(0,1)
 backbone.register_forward_pre_hook(rgb_hook)''')
put('shadow_redistribution485.py',s)

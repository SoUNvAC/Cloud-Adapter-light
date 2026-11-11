from pathlib import Path
import re
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
for n in ['train_shadow490.py','infer_shadow490.py','verify_shadow490.py','ablate_shadow490.py','boundary_shadow490.py','boundary_effect490.py','verify_archive490.py','status_shadow490.py','run_shadow490.sh','collect_all490.py','compare_shadow490.py','finish_shadow490.py','boundary_region490.py','ablate_region490.py','accept_shadow490.py','cost_shadow490.py']:
 s=re.sub(r'(?<!\d)490(?!\d)','492',(T/n).read_text(encoding='utf-8')).replace('phase492_20261108','phase492_20261109').replace('directional','chromatic').replace('default_rng(49002)','default_rng(49202)')
 if n=='train_shadow490.py':s=s.replace('neighbor_offsets_pixels=[16,32],','color_axis_scales_pixels=[12,28],')
 put(n.replace('490','492'),s)
s=(T/'shadow_redistribution488.py').read_text(encoding='utf-8')
s=s.replace('normalized=True):','normalized=False,chromatic=True):')
s=s.replace('self.enabled=True;', '''self.register_buffer('chromatic_mode',torch.tensor(chromatic,dtype=torch.bool))
  self.chromatic=nn.Sequential(nn.Conv2d(12,16,3,padding=1,bias=False),nn.GELU(),nn.Conv2d(16,16,3,padding=1,groups=16,bias=False),nn.GELU(),nn.Conv2d(16,16,1,bias=False),nn.GELU())
  self.chromatic_inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.chromatic_inject.weight)
  self.chromatic_enabled=True
  self.enabled=True;''')
s=s.replace('delta=delta+self.evidence_inject(field)', '''delta=delta+self.evidence_inject(field)
   if self.chromatic_enabled:
    delta=delta+self.chromatic_inject(self.chromatic(self.chromatic_evidence(self.rgb)))''')
pos=s.index(' def auxiliary(')
s=s[:pos]+''' def chromatic_evidence(self,rgb):
  # Display RGB only: this is a color-gradient cue, not recovered reflectance.
  with torch.no_grad():
   x=F.avg_pool2d(rgb,4,4);p=F.pad(x,(1,1,1,1),mode='replicate')
   gx=(p[:,:,1:-1,2:]-p[:,:,1:-1,:-2])*.5;gy=(p[:,:,2:,1:-1]-p[:,:,:-2,1:-1])*.5
   total=gx.square().sum(1,keepdim=True)+gy.square().sum(1,keepdim=True)
   cues=[]
   for k in [3,7]:
    color=F.avg_pool2d(F.pad(x,(k//2,)*4,mode='replicate'),k,1)
    axis=F.normalize(color,dim=1,eps=1e-4) if bool(self.chromatic_mode) else torch.ones_like(color)/(3**.5)
    px=(gx*axis).sum(1,keepdim=True);py=(gy*axis).sum(1,keepdim=True)
    ox=(gx-px*axis).square().sum(1,keepdim=True).clamp_min(0).sqrt()
    oy=(gy-py*axis).square().sum(1,keepdim=True).clamp_min(0).sqrt()
    fraction=(px.square()+py.square())/(total+1e-5)
    cues.extend([px,py,ox,oy,fraction,torch.log1p(100*total)])
   return torch.cat(cues,1)
''' +s[pos:]
put('shadow_redistribution492.py',s)

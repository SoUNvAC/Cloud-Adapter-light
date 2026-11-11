from pathlib import Path
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
for n in ['train_shadow488.py','infer_shadow488.py','verify_shadow488.py','ablate_shadow488.py','boundary_shadow488.py','boundary_effect488.py','verify_archive488.py','status_shadow488.py','run_shadow488.sh','collect_all488.py','compare_shadow488.py','finish_shadow488.py','boundary_region488.py','ablate_region488.py','accept_shadow488.py','cost_shadow488.py']:
 s=(T/n).read_text(encoding='utf-8').replace('488','490')
 if n=='train_shadow488.py':s=s.replace("normalized=ARM=='A'","normalized=False,directional=ARM=='A'").replace('texture_scales_pixels=[12,28],','texture_scales_pixels=[12,28],neighbor_offsets_pixels=[16,32],')
 if n=='infer_shadow488.py':s=s.replace("normalized=bool(branch_state.get('redistribution.normalized_mode',True))","normalized=False,directional=bool(branch_state.get('redistribution.directional_mode',True))")
 if n in ['ablate_region488.py']:s=s.replace('evidence_enabled','directional_enabled')
 if n in ['compare_shadow488.py','status_shadow488.py','accept_shadow488.py','cost_shadow488.py']:
  s=s.replace('phase482_20261108','phase488_20261108').replace('482B','488B')
 if n=='compare_shadow488.py':s=s.replace("['totals']['482']","['totals']['488']")
 if n=='cost_shadow488.py':s=s.replace('from infer_shadow482 import','from infer_shadow488 import')
 if n=='accept_shadow488.py':
  s=s.replace('redistribution.normalized_mode','redistribution.directional_mode')
  s=s.replace("['normalized']==(arm=='A')","['directional']==(arm=='A') and lock['new_module']['normalized']==False")
  s=s.replace('len(new_changed)==17','len(new_changed)==21')
 put(n.replace('488','490'),s)
s=(T/'shadow_redistribution488.py').read_text(encoding='utf-8')
s=s.replace('normalized=True):','normalized=False,directional=True):')
s=s.replace('self.enabled=True;', '''self.register_buffer('directional_mode',torch.tensor(directional,dtype=torch.bool))
  self.directional=nn.Sequential(nn.Conv2d(8,16,3,padding=1,bias=False),nn.GELU(),nn.Conv2d(16,16,3,padding=1,groups=16,bias=False),nn.GELU(),nn.Conv2d(16,16,1,bias=False),nn.GELU())
  self.directional_inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.directional_inject.weight)
  self.directional_enabled=True
  self.enabled=True;''')
s=s.replace('delta=delta+self.evidence_inject(field)', '''delta=delta+self.evidence_inject(field)
   if self.directional_enabled:
    direction=self.directional(self.directional_evidence(self.rgb,native))
    delta=delta+self.directional_inject(direction)''')
pos=s.index(' def auxiliary(')
s=s[:pos]+''' def directional_evidence(self,rgb,native):
  with torch.no_grad():
   y=F.avg_pool2d((rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True),4,4)
   p=F.pad(y,(1,1,1,1),mode='replicate');gx=(p[:,:,1:-1,2:]-p[:,:,1:-1,:-2])*.5;gy=(p[:,:,2:,1:-1]-p[:,:,:-2,1:-1])*.5
   avg=lambda t:F.avg_pool2d(F.pad(t,(1,1,1,1),mode='replicate'),3,1)
   xx=avg(gx.square());yy=avg(gy.square());xy=avg(gx*gy);energy=xx+yy
   uv=torch.cat([(xx-yy)/(energy+1e-5),2*xy/(energy+1e-5)],1);e=torch.log1p(100*energy)
   semantic=F.normalize(F.interpolate(native,size=y.shape[-2:],mode='bilinear',align_corners=False),dim=1)
   b,c,h,w=y.shape;outputs=[]
   for radius in [4,8]:
    pack=torch.cat([y,e,uv,semantic],1);pad=F.pad(pack,(radius,)*4,mode='replicate');valid=F.pad(torch.ones_like(y),(radius,)*4)
    cues=[];affinities=[]
    for dy,dx in [(-radius,-radius),(-radius,0),(-radius,radius),(0,-radius),(0,radius),(radius,-radius),(radius,0),(radius,radius)]:
     neighbor=pad[:,:,radius+dy:radius+dy+h,radius+dx:radius+dx+w]
     ok=valid[:,:,radius+dy:radius+dy+h,radius+dx:radius+dx+w]
     texture_difference=(neighbor[:,2:4]-uv).square().sum(1,keepdim=True)
     sim=(neighbor[:,4:]*semantic).sum(1,keepdim=True).clamp(-1,1)
     brightness=(neighbor[:,:1]-y)/(neighbor[:,:1]+y+.02)
     cues.append(torch.cat([brightness,neighbor[:,1:2]-e,texture_difference,1-sim],1))
     score=4*sim-2*texture_difference if bool(self.directional_mode) else torch.zeros_like(sim)
     affinities.append(score.masked_fill(ok==0,-1e4))
    weights=torch.stack(affinities,1).softmax(1)
    outputs.append((torch.stack(cues,1)*weights).sum(1))
   return torch.cat(outputs,1)
''' +s[pos:]
put('shadow_redistribution490.py',s)

from pathlib import Path
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
for n in ['train_shadow484.py','infer_shadow484.py','verify_shadow484.py','ablate_shadow484.py','boundary_shadow484.py','boundary_effect484.py','verify_archive484.py','status_shadow484.py','run_shadow484.sh','collect_all484.py','compare_shadow484.py','finish_shadow484.py','boundary_region484.py','ablate_region484.py']:
 s=(T/n).read_text(encoding='utf-8').replace('484','486').replace('regional_enabled','evidence_enabled')
 if n=='train_shadow484.py':s=s.replace("regional_conserve=ARM=='A'","adaptive=ARM=='A'").replace('regional_support_pixels=64,','neighbor_offset_pixels=32,')
 if n=='infer_shadow484.py':s=s.replace("regional_conserve=bool(branch_state.get('redistribution.regional_conserve_mode',True))","adaptive=bool(branch_state.get('redistribution.adaptive_mode',True))")
 if n=='verify_shadow484.py':s=s.replace('per.append(dict(index=', 'per.append(dict(consensus_direction=network.branch.redistribution.last_direction[0].cpu().tolist(),index=')
 put(n.replace('484','486'),s)
s=(T/'shadow_redistribution482.py').read_text(encoding='utf-8')
s=s.replace('def __init__(self,conserve=True,contrast=False):','def __init__(self,conserve=True,contrast=False,adaptive=True):')
s=s.replace('self.enabled=True;', '''self.register_buffer('adaptive_mode',torch.tensor(adaptive,dtype=torch.bool))
  self.evidence=nn.Sequential(nn.Conv2d(12,16,3,padding=1,bias=False),nn.GELU(),nn.Conv2d(16,16,3,padding=1,groups=16,bias=False),nn.GELU(),nn.Conv2d(16,16,1,bias=False),nn.GELU())
  self.evidence_inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.evidence_inject.weight)
  self.evidence_enabled=True;self.rgb=None;self.last_direction=None
  self.enabled=True;''')
s=s.replace('delta=self.inject(detail_code)', '''delta=self.inject(detail_code)
  if self.evidence_enabled:
   cue=self.consensus_evidence(self.rgb,native)
   field=F.interpolate(self.evidence(cue),size=detail.shape[-2:],mode='bilinear',align_corners=False)
   delta=delta+self.evidence_inject(field)''')
pos=s.index(' def auxiliary(')
s=s[:pos]+''' def consensus_evidence(self,rgb,native):
  with torch.no_grad():
   rgb=F.avg_pool2d(rgb,8,8);x=(rgb+.02).log();b,c,h,w=x.shape
   semantic=F.normalize(F.interpolate(native,size=(h,w),mode='bilinear',align_corners=False),dim=1)
   xp=F.pad(x,(4,4,4,4),mode='replicate');sp=F.pad(semantic,(4,4,4,4),mode='replicate')
   vp=F.pad(torch.ones_like(x[:,:1]),(4,4,4,4));deltas=[];similarities=[];weights=[]
   for dy,dx in [(-4,-4),(-4,0),(-4,4),(0,-4),(0,4),(4,-4),(4,0),(4,4)]:
    d=xp[:,:,4+dy:4+dy+h,4+dx:4+dx+w]-x
    sim=(sp[:,:,4+dy:4+dy+h,4+dx:4+dx+w]*semantic).sum(1,keepdim=True).clamp(0,1)
    valid=vp[:,:,4+dy:4+dy+h,4+dx:4+dx+w]
    norm=d.square().sum(1,keepdim=True).sqrt();chroma=d-d.mean(1,keepdim=True)
    weight=valid*sim.pow(4)*torch.sigmoid(8*d.mean(1,keepdim=True))*torch.exp(-3*chroma.square().mean(1,keepdim=True))*norm/(norm+.05)
    deltas.append(d*valid);similarities.append(sim);weights.append(weight)
   d=torch.stack(deltas,1);sim=torch.stack(similarities,1);weight=torch.stack(weights,1)
   default=F.normalize(x.new_ones((b,3)),dim=1)
   direction=(d*weight).sum((1,3,4))/weight.sum((1,3,4)).clamp_min(1e-8)
   length=direction.norm(dim=1,keepdim=True);direction=torch.where(length>1e-5,F.normalize(direction,dim=1),default)
   direction=torch.where(direction.sum(1,keepdim=True)>=0,direction,-direction)
   if not bool(self.adaptive_mode):direction=default
   self.last_direction=direction.detach()
   parallel=(d*direction[:,None,:,None,None]).sum(2,keepdim=True)
   residual=(d-parallel*direction[:,None,:,None,None]).square().sum(2,keepdim=True).clamp_min(0).sqrt()
   norm=d.square().sum(2,keepdim=True).sqrt();chroma=(d-d.mean(2,keepdim=True)).square().mean(2,keepdim=True).sqrt()
   cue=torch.cat([parallel,residual,d.mean(2,keepdim=True),chroma,sim*norm,weight*norm],2)
   return torch.cat([cue.mean(1),cue.max(1).values],1)
''' +s[pos:]
s=s.replace('def install(backbone,module):','''def install(backbone,module):
 def rgb_hook(m,args):
  x=args[0];mean=x.new_tensor([123.675,116.28,103.53])[None,:,None,None];std=x.new_tensor([58.395,57.12,57.375])[None,:,None,None]
  module.rgb=((x*std+mean)/255).clamp(0,1)
 backbone.register_forward_pre_hook(rgb_hook)''')
put('shadow_redistribution486.py',s)
s=(T/'accept_shadow484.py').read_text(encoding='utf-8').replace('484','486').replace('regional_conserve_mode','adaptive_mode').replace("['regional_conserve']","['adaptive']").replace("['regional_support_pixels']==64","['neighbor_offset_pixels']==32").replace('))==6','))==5').replace('len(new_changed)==18','len(new_changed)==17')
put('accept_shadow486.py',s)

from pathlib import Path
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
for n in ['train_shadow486.py','infer_shadow486.py','ablate_shadow486.py','boundary_shadow486.py','boundary_effect486.py','verify_archive486.py','status_shadow486.py','run_shadow486.sh','collect_all486.py','compare_shadow486.py','boundary_region486.py','ablate_region486.py','accept_shadow486.py']:
 s=(T/n).read_text(encoding='utf-8').replace('486','488').replace('adaptive','normalized').replace('neighbor_offset_pixels=32','texture_scales_pixels=[12,28]').replace("['neighbor_offset_pixels']==32","['texture_scales_pixels']==[12,28]")
 put(n.replace('486','488'),s)
for n in ['verify_shadow487.py','finish_shadow487.py','cost_shadow487.py']:
 put(n.replace('487','488'),(T/n).read_text(encoding='utf-8').replace('487','488'))
s=(T/'shadow_redistribution486.py').read_text(encoding='utf-8').replace('adaptive','normalized').replace('consensus_evidence','texture_evidence')
start=s.index(' def texture_evidence(');end=s.index(' def auxiliary(',start)
s=s[:start]+''' def texture_evidence(self,rgb,native):
  # Display-RGB texture cue, not an assumed linear-reflectance inversion.
  with torch.no_grad():
   y=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
   y=F.avg_pool2d(y,4,4)
   p=F.pad(y,(1,1,1,1),mode='replicate')
   gx=(p[:,:,1:-1,2:]-p[:,:,1:-1,:-2])*.5
   gy=(p[:,:,2:,1:-1]-p[:,:,:-2,1:-1])*.5
   def avg(t,k):return F.avg_pool2d(F.pad(t,(k//2,)*4,mode='replicate'),k,1)
   if bool(self.normalized_mode):
    denominator=avg(y,3)+.02;gx=gx/denominator;gy=gy/denominator
   cues=[]
   for k in [3,7]:
    xx=avg(gx.square(),k);yy=avg(gy.square(),k);xy=avg(gx*gy,k)
    energy=xx+yy;u=(xx-yy)/(energy+1e-5);v=2*xy/(energy+1e-5)
    # Compare the local texture tensor to its surrounding annulus.
    outer=2*k+1
    ring=lambda t:(avg(t,outer)*outer**2-avg(t,k)*k**2)/(outer**2-k**2)
    e=torch.log1p(100*energy)
    brightness=(y-ring(y))/(ring(y)+.02)
    cues.extend([u,v,e,u-ring(u),v-ring(v),brightness])
   return torch.cat(cues,1)
''' +s[end:]
put('shadow_redistribution488.py',s)

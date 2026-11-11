from pathlib import Path
import re
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
names=['train_shadow','infer_shadow','verify_shadow','ablate_shadow','boundary_shadow','boundary_effect','verify_archive','status_shadow','collect_all','compare_shadow','finish_shadow','boundary_region','ablate_region','accept_shadow','cost_shadow']
for stem in names:
 s=re.sub(r'(?<!\d)492(?!\d)','493',(T/f'{stem}492.py').read_text(encoding='utf-8')).replace('chromatic','ordinal').replace('default_rng(49202)','default_rng(49302)')
 if stem=='train_shadow':s=s.replace('color_axis_scales_pixels=[12,28],','ordinal_offsets_pixels=[1,2,4],ordinal_channels=48,')
 if stem=='collect_all':
  s=s[:s.index("a=json.loads((R/'TRAINED_WEIGHT_ACCEPTANCE.json')")]
  a=s.index('files={str(p.relative_to(r))');b=s.index("files['DELIVERY_SHA.json']",a)
  s=s[:a]+'''import os
files={}
for root,dirs,names in os.walk(r):
 dirs[:]=[n for n in dirs if n not in {'independent','ablation_off','region_off','direction_off'}]
 for name in names:
  p=Path(root)/name
  if p.suffix in ['.json','.csv','.md','.npz','.jsonl']:files[str(p.relative_to(r))]=p.read_bytes()
'''+s[b:]
 put(f'{stem}493.py',s)
put('run_shadow493.sh',re.sub(r'(?<!\d)492(?!\d)','493',(T/'run_shadow492.sh').read_text()))
s=(T/'shadow_redistribution492.py').read_text().replace('chromatic','ordinal')
s=s.replace('self.ordinal=nn.Sequential(nn.Conv2d(12,16,3','self.ordinal=nn.Sequential(nn.Conv2d(48,16,3')
a=s.index(' def ordinal_evidence(');b=s.index(' def auxiliary(',a)
s=s[:a]+''' def ordinal_evidence(self,rgb):
  # Compute microtexture BEFORE 4x pooling. A retains order; B retains bounded amplitude.
  # No segmentation threshold or learned decision rule is changed.
  with torch.no_grad():
   y=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
   h,w=y.shape[-2:];cues=[]
   for radius in [1,2,4]:
    p=F.pad(y,(radius,)*4,mode='replicate')
    for dy,dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
     neighbor=p[:,:,radius+dy*radius:radius+dy*radius+h,radius+dx*radius:radius+dx*radius+w]
     diff=neighbor-y
     cue=diff.sign() if bool(self.ordinal_mode) else torch.tanh(20*diff)
     cues.append(F.avg_pool2d(cue,4,4))
   local=torch.cat(cues,1)
   def avg(x,k):return F.avg_pool2d(F.pad(x,(k//2,)*4,mode='replicate'),k,1)
   ring=(49*avg(local,7)-9*avg(local,3))/40
   return torch.cat([local,local-ring],1)
''' +s[b:]
put('shadow_redistribution493.py',s)

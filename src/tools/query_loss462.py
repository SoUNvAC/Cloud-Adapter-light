"""Valid-pixel-aware matched query supervision; optional shadow interior/edge targets."""
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

def targets(y,size,split_shadow=False):
 valid=(y!=255).float();labels=[];masks=[]
 for c in range(3):
  m=(y==c).float()
  if c==2 and split_shadow:
   # 4-pixel geometric edge band, not a physical penumbra label.
   eroded=1-F.max_pool2d(F.pad((1-m)[None,None],(4,4,4,4),value=1),9,stride=1)[0,0]
   candidates=[eroded*m,m-eroded*m]
  else:candidates=[m]
  for candidate in candidates:
   if candidate.sum()>0:labels.append(c);masks.append(candidate)
 coverage=F.adaptive_avg_pool2d(valid[None,None],size)[0,0]
 if not masks:return torch.empty(0,dtype=torch.long,device=y.device),y.new_empty((0,*size),dtype=torch.float32),coverage
 # Occupancy is conditional on valid support, so ignore pixels never become negatives.
 occupancy=F.adaptive_avg_pool2d(torch.stack(masks)[:,None],size)[:,0]/coverage.clamp_min(1e-8)
 return torch.tensor(labels,device=y.device),occupancy,coverage

def matched_query_loss(all_cls,all_masks,y,split_shadow=False):
 total=all_cls[0].sum()*0;parts={'classification':0.,'mask':0.,'dice':0.};matched=0;images=0;target_cache={}
 for cls,masks in zip(all_cls,all_masks):
  for b in range(len(y)):
   key=(b,tuple(masks.shape[-2:]))
   if key not in target_cache:target_cache[key]=targets(y[b],masks.shape[-2:],split_shadow)
   labels,target,coverage=target_cache[key]
   if not len(labels):continue
   images+=1;pred=masks[b].flatten(1);t=target.flatten(1);w=coverage.flatten();den=w.sum().clamp_min(1);prob=pred.sigmoid()
   with torch.no_grad():
    bce=(F.softplus(pred)*w).sum(1)[:,None]/den-(pred*w)@t.T/den
    dice=1-(2*(prob*w)@t.T+1)/((prob*w).sum(1)[:,None]+(t*w).sum(1)[None]+1)
    cost=-2*cls[b].softmax(-1)[:,labels]+5*bce+5*dice
    qi,ti=linear_sum_assignment(cost.detach().cpu().numpy());qi=torch.as_tensor(qi,device=y.device);ti=torch.as_tensor(ti,device=y.device)
   query_labels=torch.full((cls.shape[1],),3,device=y.device,dtype=torch.long);query_labels[qi]=labels[ti]
   classloss=F.cross_entropy(cls[b],query_labels,weight=cls.new_tensor([1,1,1,.1]))
   chosen=pred[qi];truth=t[ti];p=chosen.sigmoid()
   maskloss=(F.binary_cross_entropy_with_logits(chosen,truth,reduction='none')*w).sum()/(len(qi)*den)
   diceloss=(1-(2*(p*truth*w).sum(1)+1)/((p*w).sum(1)+(truth*w).sum(1)+1)).mean()
   total=total+2*classloss+5*maskloss+5*diceloss;matched+=len(qi)
   for key,value in [('classification',classloss),('mask',maskloss),('dice',diceloss)]:parts[key]+=float(value.detach())
 return total/max(images,1),dict(**{k:v/max(images,1) for k,v in parts.items()},matched_masks=matched,valid_image_layers=images)

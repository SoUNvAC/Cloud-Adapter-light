"""Synthetic actual Mask2Former loss/gradient and bilinear-support gate, no fit."""
from phase236_common import *
import sys,traceback
def main():
 global budget
 tag=sys.argv[1] if len(sys.argv)>1 else 'FILL_GATE'
 assert not (R/f'{tag}_STARTED.json').exists();save(f'{tag}_STARTED.json',{'PID':os.getpid(),'epoch':time.time()})
 os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1'
 import numpy as np,torch
 from mmengine.structures import InstanceData,PixelData
 from mmseg.structures import SegDataSample
 from phase236_valid_mask import install
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.manual_seed(236)
 assert sha(SOURCE)==js(R/'LOCK.json')['Source_SHA256'];budget=Budget(tag.lower());wrapper,cfg=build(budget);h=wrapper.segmentor.decode_head
 assert sum(p.numel() for p in h.parameters())==1316516
 patch=install(h);original=patch.pop('original_single_loss');grids=patch.pop('prediction_valid_grid');save(tag+'_IMPLEMENTATION_DIFF.json',patch)
 # Actual production query count/classes/num_points/losses and matching objects;
 # synthetic decoder tensors only, no full-image model forward.
 q=h.num_queries;numclasses=h.num_classes;H=64;P=16
 cls=torch.randn((1,q,numclasses+1),device='cuda');mask=torch.randn((1,q,P,P),device='cuda')
 truth=torch.zeros((1,H,H),dtype=torch.long,device='cuda');truth[:,:,H//2:]=1;truth[:,H//2:,H//2:]=2
 meta={'img_shape':(H,H),'ori_shape':(H,H),'pad_shape':(H,H)}
 def run(truth,cs,ms,old=False):
  sample=SegDataSample(metainfo=meta);sample.gt_sem_seg=PixelData(data=truth)
  inst,metas=h._seg_data_to_instance_data([sample]);assert not (inst[0].labels==255).any()
  cs=cs.detach().clone().requires_grad_();ms=ms.detach().clone().requires_grad_()
  torch.manual_seed(123);torch.cuda.manual_seed_all(123);h._phase236_generator.manual_seed(236)
  losses=(original if old else h._loss_by_feat_single)(cs,ms,inst,metas)
  g=torch.autograd.grad(sum(losses),(cs,ms))
  return [v.detach() for v in losses],[v.detach() for v in g]
 a,ag=run(truth,cls,mask,True);b,bg=run(truth,cls,mask)
 nofill=max(float((x-y).abs().max()) for x,y in zip(a+ag,b+bg));assert nofill<=1e-6,nofill
 fill=truth.clone();fill[:,:,:H//3]=255
 predvalid=grids(fill[0]!=255,mask[0])[1][0,0].bool();assert predvalid.any() and (~predvalid).any()
 changed=mask.clone();changed[:,:,:,~predvalid.any(0)]=100  # all rows of wholly invalid columns
 # Every invalid prediction node is changed, including partial-support edge bins.
 changed=torch.where(predvalid[None,None],mask,torch.full_like(mask,100))
 c,cg=run(fill,cls,mask);d,dg=run(fill,cls,changed)
 filldiff=max(float((x-y).abs().max()) for x,y in zip(c+cg,d+dg));assert filldiff<=1e-6,filldiff
 assert float(cg[1][:,:,~predvalid].abs().max())<=1e-6
 varied=torch.where(predvalid[None,None],mask+2,mask);e,eg=run(fill,cls,varied)
 active=max(float((x-y).abs().max()) for x,y in zip(c,e));assert active>1e-6
 # Multiple decoder-layer dispatch uses the same patched single-layer routine.
 # Replay all loss terms twice with fixed RNG, validating auxiliary layers too.
 def aux(ms):
  sample=SegDataSample(metainfo=meta);sample.gt_sem_seg=PixelData(data=fill)
  inst,metas=h._seg_data_to_instance_data([sample]);torch.manual_seed(123);h._phase236_generator.manual_seed(236)
  cs=cls.clone().requires_grad_();mm=ms.clone().requires_grad_()
  out=h.loss_by_feat([cs,cs,cs],[mm,mm,mm],inst,metas)
  grad=torch.autograd.grad(sum(out.values()),(cs,mm));return {k:float(v.detach()) for k,v in out.items()},grad
 ax,axg=aux(mask);ay,ayg=aux(changed)
 assert ax==ay and max(float((x-y).abs().max()) for x,y in zip(axg,ayg))<=1e-6
 assert any(k.startswith('d0.') for k in ax)
 sparse=[]
 for yy,xx in [(0,0),(17,17),(63,63)]:
  t=torch.full_like(truth,255);t[0,yy,xx]=1;pv=grids(t[0]!=255,mask[0])[1][0,0].bool();changed=torch.where(pv[None,None],mask,torch.full_like(mask,100))
  u,ug=run(t,cls,mask);w,wg=run(t,cls,changed);difference=max(float((x-y).abs().max()) for x,y in zip(u+ug,w+wg));assert difference<=1e-6
  sparse.append({'valid_GT_pixels':1,'yx':[yy,xx],'invalid_node_loss_gradient_difference':difference})
 save(tag+'_ACCEPTED.json',{'status':'passed','default_ignore_only_removes_class_but_supervises_invalid_negative_pixels':True,'noFill_max_loss_gradient_difference':nofill,'Fill_invalid_node_change_max_loss_gradient_difference':filldiff,'valid_node_loss_change':active,'sparse_singleton_cases':sparse,'auxiliary_terms':ax,'sampling_checks':h._phase236_sampling,'num_points':h.num_points,'queries':q,'prediction_grid':[P,P],'GT_grid':[H,H],'no_model_fits':True,'production_class_mask_dice_weights':{'cls':h.loss_cls.loss_weight,'mask':h.loss_mask.loss_weight,'dice':h.loss_dice.loss_weight},'module_SHA':sha(Path(__file__).with_name('phase236_valid_mask.py')),'dependencies':{'torch':torch.__version__,'numpy':np.__version__}})
 budget.finish('passed')
if __name__=='__main__':
 try:main()
 except BaseException:
  save((sys.argv[1] if len(sys.argv)>1 else 'FILL_GATE')+'_FAILURE.json',{'traceback':traceback.format_exc()})
  if 'budget' in globals():budget.finish('failed')
  raise

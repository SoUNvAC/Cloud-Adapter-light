"""Five fixed strengths for the internal material residual, zero training."""
from pathlib import Path
import os,sys,json,hashlib,time,traceback,csv
R=Path('/home/scv/shared/phase424_20261101')
P=Path('/home/scv/shared/phase421_20261101')
os.environ['PHASE400_ROOT']=str(R);os.environ['PHASE413_MODE']='material'
sys.path.insert(0,str(R/'code'))
from phase400_common import setup,build,state_sha,sha,save,Budget

def js(p):return json.loads(p.read_text())
def writecsv(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def inspect_context(module,feature,rgb,z,conf):
 import torch
 import torch.nn.functional as F
 lum=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True).clamp_min(.01)
 chrom=rgb/rgb.sum(1,keepdim=True).clamp_min(.03);loglum=lum.log()
 avg=lambda x:F.avg_pool2d(F.pad(x,(1,1,1,1),mode='replicate'),3,stride=1)
 texture=(avg(loglum.square())-avg(loglum).square()).clamp_min(0).add(1e-6).sqrt()
 neighbor,valid=module.neighbors(torch.cat([feature,lum,chrom,texture],1))
 nf,nl,nc,nt=neighbor[:,:16],neighbor[:,16:17],neighbor[:,17:20],neighbor[:,20:21]
 sim=(F.normalize(feature,dim=1).unsqueeze(2)*F.normalize(nf,dim=1)).sum(1,keepdim=True)*4
 brighter=torch.sigmoid((nl.clamp_min(.01).log()-loglum.unsqueeze(2)-.05)/.1)
 compatibility=torch.exp(-((nc-chrom.unsqueeze(2)).square().sum(1,keepdim=True)/.02+(nt-texture.unsqueeze(2)).abs()/.2))
 sim=sim+(brighter*compatibility).clamp_min(1e-8).log()
 weight=sim.masked_fill(valid==0,-1e4).softmax(2)*valid;weight=weight/weight.sum(2,keepdim=True).clamp_min(1e-8)
 ref=(weight*neighbor).sum(2);c=(weight*brighter*compatibility).sum(2)
 reproduced=torch.cat([feature,ref[:,:16],ref[:,:16]-feature,(ref[:,16:17].clamp_min(.01).log()-loglum).clamp(-3,3),ref[:,17:20]-chrom,ref[:,20:21]-texture,c],1)
 torch.testing.assert_close(reproduced,z,rtol=0,atol=0);torch.testing.assert_close(c,conf,rtol=0,atol=0)
 return {'confidence':c,'luminance':lum,'texture':texture,'weights':weight,'brighter_mass':(weight*(nl>lum.unsqueeze(2)*1.05)).sum(2),'compatibility':(weight*compatibility).sum(2),'reference_luminance':ref[:,16:17],'descriptor':z}

def self_descriptor(z):
 d=z.clone();d[:,16:32]=d[:,:16];d[:,32:53]=0
 return d

def main():
 start=time.monotonic()
 for n,h in js(R/'LOCK.json')['files'].items():assert sha(R/n)==h,n
 torch=setup()
 import numpy as np
 import phase400_runtime
 from worker_phase400 import configure
 from mmseg.registry import DATASETS
 from mmengine.dataset import pseudo_collate
 from phase411_msre import LayerwiseMsRE
 sys.path.insert(0,'/home/scv/shared/phase400_20261031/code')
 from report_phase400 import describe,ap_exact
 os.environ['PHASE65_PREPARED_MANIFEST']=str(R/'PREPARED_MANIFEST.json')
 os.environ['PHASE65_SOURCE_CHECKPOINT']='/home/scv/Cloud-Adapter-light/src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'
 for n,h in js(P/'material/CODE_LOCK.json').items():assert sha(R/'code'/n)==h
 ckpath=P/'material/work_dirs/formal/M/iter_1000.pth'
 checkpoint=torch.load(ckpath,map_location='cpu');baseck=torch.load(P/'baseline/work_dirs/formal/M/iter_1000.pth',map_location='cpu')
 assert checkpoint['phase400']['draw_cursor']==baseck['phase400']['draw_cursor']==1000
 oldreceipt=js(P/'INDEPENDENT_ACCEPTANCE.json')
 for mode,path in [('material',ckpath),('baseline',P/'baseline/work_dirs/formal/M/iter_1000.pth')]:
  assert sha(path)==next(c['checkpoint_SHA'] for c in oldreceipt['checks'] if c['mode']==mode)
 diffs=[]
 for n,t in baseck['state_dict'].items():
  other=checkpoint['state_dict'][n]
  if not n.startswith('backbone.target_msre.'):assert torch.equal(t,other),n
  else:
   delta=(t.float()-other.float())
   diffs.append({'parameter':n,'L2_difference':float(delta.norm()),'max_absolute_difference':float(delta.abs().max())})
 writecsv(R/'COMMON_WEIGHT_CHANGES.csv',diffs)
 budget=Budget('branch_ablation')
 model,_,_=build('M',budget);model.load_state_dict(checkpoint['state_dict'],strict=True)
 model.eval()
 before=state_sha(model.state_dict());module=model.backbone.target_msre
 dataset=DATASETS.build(configure('M','formal').val_dataloader.dataset);assert len(dataset)==32
 original=js(P/'material/evaluations/M/step_1000/METRICS.json');baseline=js(P/'baseline/evaluations/M/step_1000/METRICS.json')
 oldgroups={g['product']:g for g in original['per_group']}
 import torch.nn.functional as F
 original_context=module.context;cache={};current_mode=['observed']
 def context_probe(feature,rgb):
  z,c=original_context(feature,rgb)
  if current_mode[0]=='observed':cache.update(inspect_context(module,feature,rgb,z,c))
  return (self_descriptor(z) if current_mode[0]=='self_reference' else z),c
 module.context=context_probe
 def before_msre(m,args):
  if args[1]==m.injection_indices[-1] and current_mode[0]=='observed':cache['input']=args[0].detach();cache['height']=args[2];cache['width']=args[3]
 def after_msre(m,args,out):
  if args[1]==m.injection_indices[-1] and current_mode[0]=='observed':cache['output']=out.detach()
 pre=module.register_forward_pre_hook(before_msre);post=module.register_forward_hook(after_msre)
 hook=model.register_forward_pre_hook(lambda m,a,k:budget.count('development_full_input'),with_kwargs=True)
 rows=[];groups=[];patches=[];file_records=[];metrics={};labels=0;full_replays=0
 diagnostics=R/'diagnostics';diagnostics.mkdir()
 def pool_array(a,h,w):return F.adaptive_avg_pool2d(torch.as_tensor(a.astype(np.float32),device='cuda')[None,None],(h,w))
 for mode in ['observed','self_reference']:
  current_mode[0]=mode;output=R/mode;output.mkdir();mat=np.zeros((3,3),np.int64);aps=[];alls=[];ally=[];records=[]
  for i in range(len(dataset)):
   cache.clear();sample=dataset[i];product=Path(sample['data_samples'].img_path).name.removesuffix('_rgb.npy')
   budget.current={'mode':mode,'index':i,'product':product}
   with torch.no_grad():result=model.test_step(pseudo_collate([sample]))[0]
   pred=result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8);truth=result.gt_sem_seg.data[0].cpu().numpy().astype(np.uint8);valid=truth!=255
   logits=result.seg_logits.data.float().clamp_min(0);score=(logits/logits.sum(0,keepdim=True).clamp_min(1e-12))[2].cpu().numpy()
   with np.load(P/'material'/oldgroups[product]['score_file'],allow_pickle=False) as old:
    assert np.array_equal(truth,old['truth']);labels+=1
    if mode=='observed':assert np.array_equal(pred,old['pred']) and np.array_equal(score,old['shadow']);full_replays+=1
   cm=np.bincount(3*truth[valid].astype(np.int64)+pred[valid],minlength=9).reshape(3,3);mat+=cm
   target=output/(product+'.npz');np.savez(target,pred=pred,truth=truth,shadow=score)
   with np.load(target,allow_pickle=False) as z:
    assert np.array_equal(np.bincount(3*z['truth'][valid].astype(np.int64)+z['pred'][valid],minlength=9).reshape(3,3),cm)
    ss=z['shadow'][valid].astype(np.float64);yy=z['truth'][valid]==2
   ap=ap_exact(ss,yy);aps.append(ap);alls.append(ss);ally.append(yy)
   records.append({'product':product,'confusion':cm.tolist(),'file':str(target.relative_to(R)),'SHA':sha(target)})
   groups.append({'mode':mode,'product':product,**describe(cm)})
   if mode=='observed':
    with torch.no_grad():
     h,w=cache['height'],cache['width'];z=cache['descriptor'];c=cache['confidence']
     baseout=LayerwiseMsRE.forward(module,cache['input'],module.injection_indices[-1],h,w)
     full=cache['output'][:,1:]-baseout[:,1:]
     scale=module.scale[-1];cf=c.flatten(2).transpose(1,2)
     head=lambda d:scale*module.local_output(d.flatten(2).transpose(1,2))*cf
     true_delta=head(z);self_delta=head(self_descriptor(z));zero=z.clone();zero[:,:53]=0;bias_delta=head(zero)
     torch.testing.assert_close(full,true_delta,rtol=1e-4,atol=1e-5)
     norm=lambda t:t.norm(dim=-1).reshape(1,1,h,w)
     names={k:cache[k] for k in ['confidence','luminance','texture','brighter_mass','compatibility','reference_luminance']}
     names.update(residual_norm=norm(true_delta),residual_to_input=norm(true_delta)/norm(cache['input'][:,1:]).clamp_min(1e-8),residual_to_msre=norm(true_delta)/norm(baseout[:,1:]-cache['input'][:,1:]).clamp_min(1e-8),self_norm_ratio=norm(self_delta)/norm(true_delta).clamp_min(1e-8),bias_norm_ratio=norm(bias_delta)/norm(true_delta).clamp_min(1e-8),self_cosine=F.cosine_similarity(true_delta,self_delta,dim=-1).reshape(1,1,h,w))
     names['valid_fraction']=pool_array(valid,h,w)
     gt=torch.cat([pool_array((truth==k)&valid,h,w) for k in range(3)],1)
     neighbors,_=module.neighbors(gt);refgt=(neighbors*cache['weights']).sum(2)
     for k,n in enumerate(['surface','cloud','shadow']):names['GT_'+n]=gt[:,k:k+1];names['reference_'+n]=refgt[:,k:k+1]
     with np.load(Path('/home/scv/shared/phase422_20261101/off')/(product+'.npz'),allow_pickle=False) as off:
      assert np.array_equal(off['truth'],truth)
      names['new_cloud_error_fraction']=pool_array(valid&(truth==0)&(pred==1)&(off['pred']!=1),h,w)
      names['fix_fraction']=pool_array(valid&(pred==truth)&(off['pred']!=truth),h,w)
      names['regression_fraction']=pool_array(valid&(pred!=truth)&(off['pred']==truth),h,w)
     arrays={n:t[0,0].cpu().numpy() for n,t in names.items()}
     arrays.update(weights=cache['weights'][0,0].cpu().numpy(),descriptor=z[0].cpu().numpy())
     path=diagnostics/(product+'.npz');np.savez(path,**arrays);file_records.append({'product':product,'file':str(path.relative_to(R)),'SHA':sha(path),'height':h,'width':w})
     for y in range(h):
      for x in range(w):patches.append({'product':product,'y':y,'x':x,**{n:float(a[y,x]) for n,a in arrays.items() if n not in ['weights','descriptor']}})
   if i%8==0:budget.check()
  assert sum(a is not None for a in aps)==7
  row={'mode':mode,**describe(mat),'macro_AP':float(np.mean([a for a in aps if a is not None])*100),'pooled_AP':ap_exact(np.concatenate(alls),np.concatenate(ally))*100}
  rows.append(row);metrics[mode]={**row,'per_group':records};save(output/'METRICS.json',metrics[mode]);assert state_sha(model.state_dict())==before
 hook.remove();pre.remove();post.remove();module.context=original_context
 assert budget.calls==64 and budget.updates==0 and full_replays==32 and labels==64
 budget.finish('passed')
 writecsv(R/'TARGET_SUMMARY.csv',rows);writecsv(R/'TARGET_GROUPS.csv',groups);writecsv(R/'PATCH_DIAGNOSTICS.csv',patches)
 save('DIAGNOSTIC_MANIFEST.json',file_records)
 strata={'all':lambda p:True,'surface_pure':lambda p:p['GT_surface']>=.9,'cloud_pure':lambda p:p['GT_cloud']>=.9,'shadow_pure':lambda p:p['GT_shadow']>=.9,'bright_flat_surface':lambda p:p['GT_surface']>=.9 and p['luminance']>=.7 and p['texture']<=.05,'new_cloud_error_surface':lambda p:p['GT_surface']>=.9 and p['new_cloud_error_fraction']>=.1,'T25XEG':lambda p:'T25XEG' in p['product']}
 for lo,hi in [(0,.1),(.1,.3),(.3,.5),(.5,1.00001)]:strata[f'confidence_{lo}_{hi}']=lambda p,lo=lo,hi=hi:lo<=p['confidence']<hi
 summary=[];keys=[n for n in patches[0] if n not in ['product','x','y']]
 for name,select in strata.items():
  pp=[p for p in patches if p['valid_fraction']>=.9 and select(p)]
  row={'stratum':name,'patches':len(pp),'scenes':len({p['product'] for p in pp})}
  for k in keys:row[k+'_mean']=float(np.mean([p[k] for p in pp])) if pp else None
  summary.append(row)
 writecsv(R/'STRATA_SUMMARY.csv',summary)
 audit={'passed':True,'checkpoint_SHA':sha(ckpath),'state_SHA':before,'calls':64,'updates':0,'replay_exact_scenes':32,'truth_arrays_exact':64,'reconstructed_context_exact':32,'tiny_MsRE_calls':32,'extra_small_head_calls':96,'label_usage':'posthoc diagnostics only; no inference routing','elapsed_seconds':time.monotonic()-start,'self_reference_scope':'reference features equal query, difference channels zero, original neighbor-based confidence retained'}
 save('ACCEPTANCE.json',audit)
 lines=['# Phase424：参考选择与残差响应诊断','','完整32组、零训练。观测组与Phase421预测/分数逐位一致；self_reference将参考特征替换为当前位置并清零差异项，保留原门控，因此不是完全删除邻域信息。','','|状态|mIoU|阴影IoU|阴影宏AP|','|---|---:|---:|---:|']
 for r in rows:lines.append(f"|{r['mode']}|{r['mIoU']:.4f}|{r['Shadow_IoU']:.4f}|{r['macro_AP']:.4f}|")
 lines+=['','邻域与当前像素亮度相等时，brighter=sigmoid(-0.5)=0.37754；相似且等亮的区域不会自动关闭分支。描述符保留query与reference两个绝对特征块，因此差异项为零也不保证修正为零。','','STRATA_SUMMARY.csv按特征网格统计。GT类别比例仅用于事后诊断；参考来自地表类不等于确为同材质，也不等于未受阴影影响。PATCH_DIAGNOSTICS.csv包含逐格证据，不能把patch当成独立样本推断显著性。','','本轮正式64次完整输入前向、零更新；另32次小张量MsRE及96次小修正头前向。没有修改标签，没有GT引导的推理。']
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 save('COMPLETE.json',{'passed':True,'new_calls':64,'new_updates':0,'files':{n:sha(R/n) for n in ['RESULT.md','ACCEPTANCE.json','TARGET_SUMMARY.csv','TARGET_GROUPS.csv','STRATA_SUMMARY.csv','PATCH_DIAGNOSTICS.csv','DIAGNOSTIC_MANIFEST.json']}})
 (R/'EXIT_CODE').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:save('FAILURE.json',{'traceback':traceback.format_exc()});(R/'EXIT_CODE').write_text('1');raise

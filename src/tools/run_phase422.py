"""Same-checkpoint branch ablation, no training or checkpoint mutation."""
from pathlib import Path
import os,sys,json,hashlib,time,traceback,csv
R=Path('/home/scv/shared/phase422_20261101')
P=Path('/home/scv/shared/phase421_20261101')
os.environ['PHASE400_ROOT']=str(R);os.environ['PHASE413_MODE']='material'
sys.path.insert(0,str(R/'code'))
from phase400_common import setup,build,state_sha,sha,save,Budget

def js(p):return json.loads(p.read_text())
def writecsv(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

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
 # Verify mode switch bypasses only the new branch, retaining learned MsRE.
 module.mode='baseline';toy=torch.randn(1,10,384,device='cuda')
 with torch.no_grad():
  torch.testing.assert_close(module(toy,11,3,3),LayerwiseMsRE.forward(module,toy,11,3,3),rtol=0,atol=0)
 module.mode='material'
 rows=[];groups=[];metrics={};maxscore=0.;label_checks=0
 hook=model.register_forward_pre_hook(lambda m,a,k:budget.count('development_full_input'),with_kwargs=True)
 for mode in ['on','off']:
  module.mode='material' if mode=='on' else 'baseline'
  output=R/mode;output.mkdir();aggregate=np.zeros((3,3),np.int64);aps=[];all_scores=[];all_y=[];records=[]
  for i in range(len(dataset)):
   sample=dataset[i];product=Path(sample['data_samples'].img_path).name.removesuffix('_rgb.npy')
   budget.current={'mode':mode,'index':i,'product':product}
   with torch.no_grad():result=model.test_step(pseudo_collate([sample]))[0]
   pred=result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
   truth=result.gt_sem_seg.data[0].cpu().numpy().astype(np.uint8);valid=truth!=255
   logits=result.seg_logits.data.float().clamp_min(0);prob=logits/logits.sum(0,keepdim=True).clamp_min(1e-12)
   score=prob[2].cpu().numpy();assert np.isfinite(score).all()
   old=np.load(P/'material'/oldgroups[product]['score_file'],allow_pickle=False)
   assert np.array_equal(truth,old['truth']);label_checks+=1
   if mode=='on':
    assert np.array_equal(pred,old['pred']),('replay prediction mismatch',product)
    error=float(np.max(np.abs(score-old['shadow'])));maxscore=max(maxscore,error);save('REPLAY_LAST.json',{'product':product,'max_score_error':error});assert error<=1e-5
   cm=np.bincount(3*truth[valid].astype(np.int64)+pred[valid],minlength=9).reshape(3,3)
   reference=old['pred'];fixes=int(((pred==truth)&(reference!=truth)&valid).sum());regressions=int(((pred!=truth)&(reference==truth)&valid).sum())
   old.close();aggregate+=cm
   target=output/(product+'.npz');np.savez(target,pred=pred,truth=truth,shadow=score)
   # Reload the written artifact for independent metric computation.
   with np.load(target,allow_pickle=False) as check:
    recomputed=np.bincount(3*check['truth'][valid].astype(np.int64)+check['pred'][valid],minlength=9).reshape(3,3)
    assert np.array_equal(recomputed,cm)
    s=check['shadow'][valid].astype(np.float64);y=check['truth'][valid]==2
   ap=ap_exact(s,y);aps.append(ap);all_scores.append(s);all_y.append(y)
   rec={'product':product,'confusion':cm.tolist(),'score_file':str(target.relative_to(R)),'SHA':sha(target),'fixes_vs_on':fixes,'regressions_vs_on':regressions}
   records.append(rec);groups.append({'mode':mode,'product':product,**describe(cm),'fixes_vs_on':fixes,'regressions_vs_on':regressions})
   if i%8==0:budget.check()
  assert sum(x is not None for x in aps)==7
  row={'mode':mode,**describe(aggregate),'macro_AP':100*float(np.mean([x for x in aps if x is not None])),'pooled_AP':100*ap_exact(np.concatenate(all_scores),np.concatenate(all_y))}
  rows.append(row);metrics[mode]={'confusion':aggregate.tolist(),'per_group':records,**row};save(output/'METRICS.json',metrics[mode])
  assert state_sha(model.state_dict())==before
 hook.remove();module.mode='material';assert budget.calls==64 and budget.updates==0
 assert metrics['on']['confusion']==original['confusion']
 budget.finish('passed')
 writecsv(R/'TARGET_SUMMARY.csv',rows);writecsv(R/'TARGET_GROUPS.csv',groups)
 off=metrics['off']['mIoU'];on=metrics['on']['mIoU'];base=baseline['mIoU']
 decomposition={'baseline_mIoU':base,'on_mIoU':on,'off_mIoU':off,'direct_branch_on_minus_off_pp':on-off,'jointly_trained_MsRE_off_minus_baseline_pp':off-base,'total_on_minus_baseline_pp':on-base,'interpretation':'additive endpoint contrasts, not a full causal decomposition of training'}
 save('DECOMPOSITION.json',decomposition)
 audit={'passed':True,'checkpoint_SHA':sha(ckpath),'weights_unchanged_SHA':before,'on_prediction_replay_exact_scenes':32,'max_shadow_score_error':maxscore,'label_arrays_exact':label_checks,'new_full_input_forwards':64,'updates':0,'off_equals_LayerwiseMsRE_unit_test':True,'CPU_toy_MsRE_calls':0,'GPU_toy_MsRE_calls':2,'all_non_MsRE_checkpoint_weights_equal':True,'elapsed_seconds':time.monotonic()-start}
 save('ACCEPTANCE.json',audit)
 lines=['# Phase422：材质参考损伤诊断','', '|状态|目标mIoU|','|---|---:|',f'|Phase421独立训练基线|{base:.4f}|',f'|材质模型：分支开启|{on:.4f}|',f'|同一材质模型：仅关闭新增分支|{off:.4f}|','',f'分支直接作用（开启−关闭）：{on-off:+.4f}个百分点。',f'共同训练后的MsRE状态（关闭−基线）：{off-base:+.4f}个百分点。','','关闭仅绕过local_projection/local_output的新增残差路径，保留同一检查点中的全部分层MsRE权重与scale。两个差值可相加，但不能据此认定训练过程中的完整因果关系。','','64次完整输入前向、零训练；另有2次小张量MsRE前向检查开关。完整32组开启预测复现，标签、权重不变，输出文件重算混淆矩阵及AP通过。']
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 save('COMPLETE.json',{'passed':True,'new_calls':64,'new_updates':0,'files':{n:sha(R/n) for n in ['RESULT.md','ACCEPTANCE.json','DECOMPOSITION.json','TARGET_SUMMARY.csv','TARGET_GROUPS.csv','COMMON_WEIGHT_CHANGES.csv']}})
 (R/'EXIT_CODE').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:save('FAILURE.json',{'traceback':traceback.format_exc()});(R/'EXIT_CODE').write_text('1');raise

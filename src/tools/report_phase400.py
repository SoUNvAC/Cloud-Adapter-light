"""CPU AP, explicit source-on/off reporting, and small formal-only delivery."""
from phase400_common import *
import numpy as np,shutil,csv,math
def writecsv(path,rows):
 keys=list(dict.fromkeys(k for v in rows for k in v))
 with Path(path).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in v.items()} for v in rows])
def ap_exact(scores,labels):
 P=int(labels.sum());N=len(labels)-P
 if P==0 or N==0:return None
 order=np.argsort(-scores,kind='stable');s=scores[order];y=labels[order];ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1];tp=np.cumsum(y,dtype=np.int64)[ends];fp=ends+1-tp;increments=np.diff(np.r_[0,tp]);return float(np.sum(increments*(tp/(tp+fp)))/P)
def describe(cm):
 c=np.asarray(cm,np.int64);d=c.diagonal();r=c.sum(1);p=c.sum(0);u=r+p-d;iou=[float(d[i]/u[i]*100) if u[i] else None for i in range(3)];return {'mIoU':float(np.mean([v for v in iou if v is not None])) if any(v is not None for v in iou) else None,'IoU':iou,'Shadow_IoU':iou[2],'Shadow_recall':float(d[2]/r[2]*100) if r[2] else None,'nonShadow_FPR':float((c[0,2]+c[1,2])/(r[0]+r[1])*100) if r[0]+r[1] else None,'reference':r.tolist(),'confusion':c.tolist()}
def main():
 started=time.monotonic();assert not (R/'delivery').exists();summaries=[];pergroup=[];curve=[];checks=[]
 paths=[('S',None,R/'evaluations/Source/METRICS.json')]+[(a,i,R/'evaluations'/a/('step_'+str(i))/'METRICS.json') for a in ('MH','M','H') for i in range(500,4001,500)]
 computed={}
 for arm,step,path in paths:
  report=js(path);pooled_s=[];pooled_y=[];aps=[];cm=np.zeros((3,3),np.int64);Shadowgroups=0
  for g in report['per_group']:
   assert sha(R/g['score_file'])==g['score_SHA']
   with np.load(R/g['score_file'],allow_pickle=False) as z:s=z['shadow'];truth=z['truth'];valid=truth!=255;s=s[valid].astype(np.float64);y=truth[valid]==2
   P=int(y.sum());Shadowgroups+=P>0;ap=ap_exact(s,y);aps.append(ap);pooled_s.append(s);pooled_y.append(y);cm+=np.array(g['confusion']);pergroup.append({'arm':arm,'step':step,'group_id':g['group_id'],'product':g['product'],'AP':None if ap is None else ap*100,'positive_shadow':P,'valid':len(y),**describe(g['confusion'])})
  assert cm.tolist()==report['confusion'];assert abs(describe(cm)['mIoU']-report['mIoU'])<1e-10;assert len(report['per_group'])==32
  scores=np.concatenate(pooled_s);labels=np.concatenate(pooled_y);poolap=ap_exact(scores,labels);macro=float(np.mean([a for a in aps if a is not None])*100);summary={'arm':arm,'step':step,'domain':'Catalogue_development','scope':'exposed_exploratory_32_groups','source_branch':'baseline' if arm=='S' else 'on',**describe(cm),'macro_Shadow_AP':macro,'pooled_Shadow_AP':poolap*100 if poolap is not None else None,'groups_with_Shadow_reference':Shadowgroups,'AP_evaluable_groups':sum(a is not None for a in aps)};computed[arm,step]=summary;curve.append(summary);del scores,labels,pooled_s,pooled_y;checks.append('32 group count/support/mIoU/AP '+str((arm,step)))
 endpoints=js(R/'SELECTION_LOCK.json')['endpoints'];aliases=js(R/'SOURCE_EVALUATION_ALIASES.json');source=js(R/'source_evaluations/Source/METRICS.json');summaries.append({**computed['S',None],'role':'baseline'});summaries.append({'arm':'S','role':'baseline','domain':'CloudSEN_val535','source_branch':'baseline',**describe(source['confusion'])})
 for e in endpoints:
  arm=e['arm'];step=e['step'];r=next(v for v in aliases if v['arm']==arm and v['role']==e['role']);summaries.append({**computed[arm,step],'role':e['role'],'checkpoint_SHA':e['checkpoint_SHA']})
  for branch in ('on','off'):
   v=js(R/'source_evaluations'/(r['alias_of']+'_'+branch)/'METRICS.json');assert len(v['per_input'])==535;aggregate=np.sum([x['confusion'] for x in v['per_input']],axis=0);assert aggregate.tolist()==v['confusion'];summaries.append({'arm':arm,'step':step,'role':e['role'],'domain':'CloudSEN_val535','source_branch':branch,'evaluation_alias':r['alias_of']+'_'+branch,**describe(v['confusion']),'delta_Source_mIoU':v['mIoU']-source['mIoU'],'off_max_abs':v['branch_off_max_abs'],'off_disagreement_images':v['branch_off_prediction_disagreement_images']});checks.append('source535 count/parent support '+str((arm,e['role'],branch)))
  ck=__import__('torch').load(e['checkpoint'],map_location='cpu');assert ck['phase400']['Source_SHA']==EXPECTED_SOURCE and ck['phase400']['input_lock_SHA']==sha(R/'INPUT_LOCK.json');assert all(k in ck for k in ('optimizer','param_schedulers'));checks.append('checkpoint state optimizer scheduler input identity '+str((arm,e['role'])));del ck
 deltas=[]
 for role in ('best','final'):
  a={v['arm']:v for v in summaries if v['domain']=='Catalogue_development' and v.get('role')==role}
  for left,right in [('MH','M'),('MH','H')]:deltas.append({'role':role,'comparison':left+'-'+right,'pooled_mIoU_delta':a[left]['mIoU']-a[right]['mIoU'],'macro_AP_delta':a[left]['macro_Shadow_AP']-a[right]['macro_Shadow_AP'],'pooled_AP_delta':a[left]['pooled_Shadow_AP']-a[right]['pooled_Shadow_AP']})
  for arm,v in a.items():deltas.append({'role':role,'comparison':arm+'-S','pooled_mIoU_delta':v['mIoU']-computed['S',None]['mIoU'],'macro_AP_delta':v['macro_Shadow_AP']-computed['S',None]['macro_Shadow_AP'],'pooled_AP_delta':v['pooled_Shadow_AP']-computed['S',None]['pooled_Shadow_AP']})
 # Independent toy AP definition and input-pairing verification; no scene bootstrap.
 z=np.array([.8,.5,.5,.1]);y=np.array([1,1,0,0],np.uint8);assert abs(ap_exact(z,y)-5/6)<1e-12
 for kind in ('short','formal'):
  traces={a:[json.loads(l) for l in (R/'work_dirs'/kind/a/'STEP_TRACE.jsonl').read_text().splitlines()] for a in ('MH','M','H')};expected=120 if kind=='short' else 4000;assert all(len(v)==expected for v in traces.values())
  for i in range(expected):assert all(all(traces[a][i][k]==traces['MH'][i][k] for k in ('draw','product','image_SHA','label_SHA','augmentation_seed')) for a in ('M','H'))
  checks.append('full paired data/augmentation trace '+kind)
 writecsv(R/'SUMMARY.csv',summaries);writecsv(R/'PER_GROUP.csv',pergroup);writecsv(R/'DEVELOPMENT_CURVE.csv',curve);writecsv(R/'COMPARISONS.csv',deltas);save('INDEPENDENT_ACCEPTANCE.json',{'passed':True,'checks':checks,'checkpoint_identity_verified':True,'same_input_and_augmentation_trace':True,'AP_ties_unit':True,'no_independent_confirmation':True})
 costfiles={p.name:js(p) for p in (R/'costs').glob('*.json')};save('COST.json',{'GPU_stage_seconds':prior_gpu_seconds(),'CPU_statistics_seconds':time.monotonic()-started,'stages':costfiles,'new_bytes_before_delivery':sum(p.stat().st_size for p in R.rglob('*') if p.is_file()),'checkpoint_bytes':{str(p.relative_to(R)):p.stat().st_size for p in (R/'work_dirs').rglob('*.pth')},'formal_updates':12000,'short_updates':360,'gate_updates':3,'peak_note':'samples/allocator peaks, not a separate deployment process peak','limits_GPU_seconds':43200})
 lines=['# Phase400：MsRE与输出残差头最小拆分','', '## 1．数据结果','', 'Catalogue153 fit代表、固定32 development组；development已暴露且用于选模。Source保留评价只用CloudSEN val535。H/M参数量不同，单seed40001；不作等容量或独立确认结论。','', '|臂|角色|步数|目标mIoU|Shadow IoU|宏AP|池化AP|Shadow召回|非Shadow FPR|','|---|---|---|---|---|---|---|---|---|']
 for v in summaries:
  if v['domain']=='Catalogue_development':lines.append('|'+ '|'.join([v['arm'],v['role'],str(v.get('step'))]+['NA' if v.get(k) is None else f'{v[k]:.4f}' for k in ('mIoU','Shadow_IoU','macro_Shadow_AP','pooled_Shadow_AP','Shadow_recall','nonShadow_FPR')])+'|')
 lines+=['','|预定比较|角色|mIoU差／百分点|宏AP差|池化AP差|','|---|---|---|---|---|']
 for v in deltas:lines.append(f"|{v['comparison']}|{v['role']}|{v['pooled_mIoU_delta']:.4f}|{v['macro_AP_delta']:.4f}|{v['pooled_AP_delta']:.4f}|")
 lines+=['','|臂|角色|源域分支|val535 mIoU|相对S|类IoU|','|---|---|---|---|---|---|']
 for v in summaries:
  if v['domain']=='CloudSEN_val535':lines.append(f"|{v['arm']}|{v['role']}|{v['source_branch']}|{v['mIoU']:.4f}|{v.get('delta_Source_mIoU',0.):.4f}|{v['IoU']}|")
 lines+=['','## 2．观察与未确定事项','','branch-off仅为Source回退一致性检查，不能当作开启目标适配路径的无遗忘证据。branch-on代价、M+H相对M/H增量、mIoU与AP取舍应分别解释；预算与短跑实测在SHORT_RUN.json，输入及共同初始化已锁定。单seed、组件参数量差异、development选模与曝光使结果仅支持探索诊断。完整数据到齐后补充具体数值解释与下一小步建议，不自动执行401。']
 (R/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');d=R/'delivery';d.mkdir();files=['README.md','SUMMARY.csv','PER_GROUP.csv','DEVELOPMENT_CURVE.csv','COMPARISONS.csv','INPUT_LOCK.json','CODE_LOCK.json','EXPERIMENT_MATRIX.json','PREFLIGHT_READONLY.json','RGB_RESTORATION.json','CORRECTNESS_GATE.json','SHORT_RUN.json','FORMAL_ADMISSION.json','SELECTION_LOCK.json','SOURCE_EVALUATION_ALIASES.json','INDEPENDENT_ACCEPTANCE.json','COST.json']
 for n in files:shutil.copy2(R/n,d/n)
 shutil.copytree(R/'code',d/'code');save(d/'SHA256SUMS.json',{p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()});save('DELIVERY_COMPLETE.json',{'passed':True,'epoch':time.time(),'manifest_SHA':sha(d/'SHA256SUMS.json'),'formal_bytes':sum(p.stat().st_size for p in d.rglob('*') if p.is_file()),'no_work_dirs_or_weights_transferred':True})
if __name__=='__main__':main()

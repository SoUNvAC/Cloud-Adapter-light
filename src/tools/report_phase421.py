"""Independent CPU audit for Phase412/413, including exact AP from score arrays."""
from pathlib import Path
import sys,os,json,hashlib,time,csv
import numpy as np
PHASE=int(sys.argv[1]);R=Path(f'/home/scv/shared/phase{PHASE}_20261101')
sys.path.insert(0,'/home/scv/shared/phase400_20261031/code')
from report_phase400 import ap_exact,describe,writecsv
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(R/n).write_text(json.dumps(v,indent=2))
def main():
 start=time.monotonic();complete=js(R/'COMPLETE.json');assert complete['passed']
 lock=js(R/'LOCK.json')
 for n,h in lock['files'].items():assert sha(R/n)==h
 reference=Path('/home/scv/shared/phase411_20261101/layerwise')
 refscope=js(reference/'work_dirs/formal/M/ACTIVE_SCOPE.json')
 reftrace=[json.loads(x) for x in (reference/'work_dirs/formal/M/STEP_TRACE.jsonl').read_text().splitlines()]
 rows=[];groups=[];costs=[];checks=[]
 for mode,receipt in complete['reports'].items():
  sr=R/mode;steps=receipt['steps'];scope=js(sr/'work_dirs/formal/M/ACTIVE_SCOPE.json')
  for n,h in js(sr/'CODE_LOCK.json').items():assert sha(sr/'code'/n)==h
  assert scope['frozen_SHA']==refscope['frozen_SHA'] and scope['head_initial_SHA']==refscope['head_initial_SHA']
  assert scope['common_MsRE_SHA']==refscope['MsRE_initial_SHA']
  assert scope['count']==sum(scope['names'].values())==receipt['parameters']
  trace=[json.loads(x) for x in (sr/'work_dirs/formal/M/STEP_TRACE.jsonl').read_text().splitlines()]
  assert len(trace)==steps
  for a,b in zip(trace,reftrace):
   assert all(a[k]==b[k] for k in ['step','draw','product','image_SHA','label_SHA','augmentation_seed'])
   assert len(set(a['LR']))==1 and a['LR'][0]==b['LR'][0]
  ledger=[json.loads(x) for x in (sr/'CALL_LEDGER.jsonl').read_text().splitlines()]
  assert len(ledger)==steps+32*(steps//500)==receipt['calls']
  assert sum(x['kind']=='train_full_input' for x in ledger)==steps
  for step in range(100,steps+1,100):assert js(sr/f'work_dirs/formal/M/FREEZE_{step}.json')['passed']
  cost=js(sr/'costs/formal_M.json');assert cost['updates']==steps and cost['calls']==len(ledger)
  costs.append({'mode':mode,'steps':steps,'calls':len(ledger),'parameters':scope['count'],'seconds':cost['seconds'],'mean_step_seconds':receipt['mean_step_seconds'],'GPU_reserved_peak':max(x['GPU_peak_reserved'] for x in cost['resources'])})
  for step in range(500,steps+1,500):
   report=js(sr/f'evaluations/M/step_{step}/METRICS.json');mat=np.zeros((3,3),dtype=np.int64);aps=[];pooleds=[];pooledy=[]
   assert len(report['per_group'])==32 and len({g['product'] for g in report['per_group']})==32
   for g in report['per_group']:
    path=sr/g['score_file'];assert sha(path)==g['score_SHA']
    with np.load(path,allow_pickle=False) as z:
     truth=z['truth'];pred=z['pred'];valid=truth!=255;s=z['shadow'][valid].astype(np.float64);y=truth[valid]==2
     assert hashlib.sha256(pred.tobytes()).hexdigest()==g['prediction_SHA']
     recomputed=np.bincount(3*truth[valid].astype(np.int64)+pred[valid],minlength=9).reshape(3,3)
     assert recomputed.tolist()==g['confusion']
     old=np.load(reference/f'evaluations/M/step_1000/{g["product"]}.npz',allow_pickle=False)
     assert np.array_equal(truth,old['truth']);old.close()
    assert np.isfinite(s).all() and s.min()>=0 and s.max()<=1
    cm=np.asarray(g['confusion'],dtype=np.int64)
    assert np.bincount(truth[valid],minlength=3).tolist()==cm.sum(1).tolist()
    ap=ap_exact(s,y);aps.append(ap);pooleds.append(s);pooledy.append(y);mat+=cm
    groups.append({'mode':mode,'step':step,'product':g['product'],'group_id':g['group_id'],**describe(cm),'AP':None if ap is None else ap*100})
   assert mat.tolist()==report['confusion'];metrics=describe(mat);assert abs(metrics['mIoU']-report['mIoU'])<1e-9
   assert sum(a is not None for a in aps)==7
   rows.append({'mode':mode,'step':step,**metrics,'macro_AP':float(np.mean([a for a in aps if a is not None])*100),'pooled_AP':ap_exact(np.concatenate(pooleds),np.concatenate(pooledy))*100})
   save('REPORT_STATUS.json',{'mode':mode,'step':step,'seconds':time.monotonic()-start})
  # Verify saved model endpoint identity, scope and finite weights independently.
  import torch
  ck=torch.load(sr/f'work_dirs/formal/M/iter_{steps}.pth',map_location='cpu')
  assert ck['phase400']['draw_cursor']==steps and ck['phase400']['Source_SHA']==lock['Source_SHA']
  assert all(torch.isfinite(v).all() for k,v in ck['state_dict'].items() if k.startswith('backbone.target_msre.'))
  checks.append({'mode':mode,'checkpoint_SHA':sha(sr/f'work_dirs/formal/M/iter_{steps}.pth'),'paired_trace':True,'frozen_source':True,'AP_scores_hash_verified':True})
 writecsv(R/'TARGET_SUMMARY.csv',rows);writecsv(R/'TARGET_GROUPS.csv',groups);writecsv(R/'COST.csv',costs)
 endpoint={r['mode']:r for r in rows if r['step']==1000 and not r['mode'].endswith('_full')}
 baseline=js(reference/'evaluations/M/step_1000/METRICS.json')['mIoU']
 comparisons=[]
 for mode in ['neighborhood','material']:
  for control in (['baseline'] if mode=='neighborhood' else ['baseline','neighborhood']):
   a,b=endpoint[mode],endpoint[control]
   comparisons.append({'mechanism':mode,'control':control,'delta_mIoU':a['mIoU']-b['mIoU'],'delta_Shadow_IoU':a['Shadow_IoU']-b['Shadow_IoU'],'delta_macro_AP':a['macro_AP']-b['macro_AP']})
 margin=endpoint['material']['mIoU']-max(endpoint[c]['mIoU'] for c in ['baseline','neighborhood'])
 save('MECHANISM_SCREEN.json',{'margin_pp':margin,'beats_both':margin>0,'clears_0_30pp':margin>=.30,'automatic_promotion':False})
 writecsv(R/'COMPARISONS.csv',comparisons)
 # Check equal extension initialization and scope, baseline reproduction separately.
 assert complete['reports']['material']['parameters']==complete['reports']['neighborhood']['parameters']
 a=js(R/'material/work_dirs/formal/M/ACTIVE_SCOPE.json');b=js(R/'neighborhood/work_dirs/formal/M/ACTIVE_SCOPE.json')
 assert a['MsRE_initial_SHA']==b['MsRE_initial_SHA']
 base_delta=endpoint['baseline']['mIoU']-baseline
 base_exact=js(R/'baseline/evaluations/M/step_1000/METRICS.json')['confusion']==js(reference/'evaluations/M/step_1000/METRICS.json')['confusion']
 accepted={'passed':True,'checks':checks,'calls':sum(c['calls'] for c in costs),'updates':sum(c['steps'] for c in costs),'baseline_delta_vs411_pp':base_delta,'baseline_CM_exact_vs411':base_exact,'label_arrays_exact':192,'new_head_same_initialization':True,'CPU_report_seconds':time.monotonic()-start,'scope':'single seed development screening; no confirmation use; no auxiliary loss'}
 save('INDEPENDENT_ACCEPTANCE.json',accepted)
 lines=[f'# Phase{PHASE} 结果','', '153 fit、完整32组development，seed40901，各1000步；表中固定终点，不选峰值。无末端修正、无辅助语义损失。', '', '|方案|mIoU|地表IoU|云IoU|阴影IoU|阴影宏AP|训练参数|','|---|---:|---:|---:|---:|---:|---:|']
 for mode,row in endpoint.items():
  metric=js(R/mode/'evaluations/M/step_1000/METRICS.json')
  lines.append(f"|{mode}|{row['mIoU']:.4f}|{metric['IoU'][0]:.4f}|{metric['IoU'][1]:.4f}|{row['Shadow_IoU']:.4f}|{row['macro_AP']:.4f}|{complete['reports'][mode]['parameters']}|")
 lines+=['','|比较|mIoU差/百分点|阴影IoU差|宏AP差|','|---|---:|---:|---:|']
 for c in comparisons:lines.append(f"|{c['mechanism']} − {c['control']}|{c['delta_mIoU']:+.4f}|{c['delta_Shadow_IoU']:+.4f}|{c['delta_macro_AP']:+.4f}|")
 lines+=['',f'材质版相对最强对照：{margin:+.4f}个百分点；超过两对照：{margin>0}；达到优先复验0.30点标准：{margin>=.30}。',f'基线相对历史Phase411：{base_delta:+.8f}个百分点，混淆矩阵完全复现：{base_exact}。',f"本轮{accepted['calls']}次前向、{accepted['updates']}次更新，GPU阶段累计{sum(c['seconds'] for c in costs)/60:.2f}分钟。",'', '验收：共同初始权重、逐步增强/标签/学习率、冻结参数、192张标签数组与历史一致、完整预测重算混淆矩阵和精确AP通过。','', '解释边界：这是局部颜色比例/纹理/亮度先验的组合测试，没有无云参考真值，不能将检索到的邻域称为真实同材质地表。单seed短程结果不能证明稳定性或发表新颖性。']
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 save('REPORT_COMPLETE.json',{'passed':True,'files':{n:sha(R/n) for n in ['RESULT.md','TARGET_SUMMARY.csv','TARGET_GROUPS.csv','COST.csv','COMPARISONS.csv','INDEPENDENT_ACCEPTANCE.json','MECHANISM_SCREEN.json']}})
if __name__=='__main__':main()

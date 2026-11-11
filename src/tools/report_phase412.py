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
     truth=z['truth'];valid=truth!=255;s=z['shadow'][valid].astype(np.float64);y=truth[valid]==2
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
 families=([('region_dynamic',['region_fixed']),('layer_dynamic',['layer_fixed']),('site_dynamic',['site_fixed'])] if PHASE==412 else [('dual_operator',['dual_generic','dual_shuffled']),('link_cloud',['link_generic']),('field_illumination',['field_spatial','field_shuffled'])])
 comparisons=[];eligible=[]
 for mechanism,controls in families:
  gain=endpoint[mechanism]['mIoU']-max([baseline]+[endpoint[c]['mIoU'] for c in controls])
  if gain>=.30:eligible.append((gain,mechanism))
  for control in controls:
   comparisons.append({'mechanism':mechanism,'control':control,'delta_mIoU':endpoint[mechanism]['mIoU']-endpoint[control]['mIoU'],'delta_Shadow_IoU':endpoint[mechanism]['Shadow_IoU']-endpoint[control]['Shadow_IoU'],'delta_macro_AP':endpoint[mechanism]['macro_AP']-endpoint[control]['macro_AP'],'delta_vs_Phase411':endpoint[mechanism]['mIoU']-baseline,'promotion_margin':gain})
 assert complete['selected']==[m for g,m in sorted(eligible,reverse=True)[:2]]
 writecsv(R/'COMPARISONS.csv',comparisons)
 baseline_rows=list(csv.DictReader((reference.parent/'TARGET_SUMMARY.csv').open()))
 baseline_rows=[r for r in baseline_rows if r['mode']=='layerwise']
 assert len(baseline_rows)==8
 full=[]
 for mode in complete['selected']:
  rr=[r for r in rows if r['mode']==mode+'_full'];best=max(rr,key=lambda r:(r['mIoU'],-r['step']))
  for role,row in [('best',best),('final',next(r for r in rr if r['step']==4000))]:
   ref=max(baseline_rows,key=lambda r:(float(r['mIoU']),-int(r['step']))) if role=='best' else next(r for r in baseline_rows if int(r['step'])==4000)
   full.append({'mode':mode,'role':role,'step':row['step'],'mIoU':row['mIoU'],'delta_vs_Phase411':row['mIoU']-float(ref['mIoU'])})
 writecsv(R/'FULL_COMPARISONS.csv',full)
 retired=[]
 for old in R.parent.glob(R.name+'_attempt*'):
  calls=sum(sum(1 for _ in p.open()) for p in old.glob('*/CALL_LEDGER.jsonl'))
  updates=sum(sum(1 for _ in p.open()) for p in old.glob('*/work_dirs/formal/M/STEP_TRACE.jsonl'))
  retired.append({'root':str(old),'calls':calls,'completed_update_trace_rows':updates,'reason':js(old/'PROTOCOL_RETIRED.json') if (old/'PROTOCOL_RETIRED.json').exists() else None})
 accepted={'passed':True,'checks':checks,'selection_recomputed':True,'calls':sum(c['calls'] for c in costs),'updates':sum(c['steps'] for c in costs),'retired_attempts':retired,'CPU_report_seconds':time.monotonic()-start,'scope':'single seed exposed development screen; no independent confirmation; frozen source weights do not imply preserved source performance'}
 save('INDEPENDENT_ACCEPTANCE.json',accepted)
 lines=[f'# Phase{PHASE} 结果','',f'固定32组Catalogue development，153 fit，seed40901。下表均为第1000步；Phase411同一步 {baseline:.4f}%。源域不参与选择。','', '|方案|mIoU|影IoU|影宏AP|训练参数|','|---|---:|---:|---:|---:|']
 for mode,row in endpoint.items():lines.append(f"|{mode}|{row['mIoU']:.4f}|{row['Shadow_IoU']:.4f}|{row['macro_AP']:.4f}|{complete['reports'][mode]['parameters']}|")
 lines+=['','|机制－对照|mIoU差/百分点|影IoU差|宏AP差|','|---|---:|---:|---:|']
 for c in comparisons:lines.append(f"|{c['mechanism']} − {c['control']}|{c['delta_mIoU']:+.4f}|{c['delta_Shadow_IoU']:+.4f}|{c['delta_macro_AP']:+.4f}|")
 lines+=['', '晋级4000步：'+(', '.join(complete['selected']) or '无。没有方案同时超过对应所有对照及Phase411至少0.30点。')]
 if full:
  lines+=['','|完整训练|角色|步数|mIoU|相对Phase411|','|---|---|---:|---:|---:|']
  for f in full:lines.append(f"|{f['mode']}|{f['role']}|{f['step']}|{f['mIoU']:.4f}|{f['delta_vs_Phase411']:+.4f}|")
 lines+=['',f"有效实验：{accepted['calls']}次前向、{accepted['updates']}次更新；GPU阶段累计{sum(c['seconds'] for c in costs)/60:.2f}分钟。废弃/修正尝试另外记账于INDEPENDENT_ACCEPTANCE.json，不隐去成本。",'', '验收：数据与增强逐步配对、学习率一致、共同初始权重一致、冻结权重检查、32组混淆矩阵汇总、分数文件哈希、精确AP和晋级规则复核通过。', '', '结论范围：这是单seed短程机制筛选，不足以证明稳定性、最终上限或论文创新。500步仅按既有调度验证，不参与筛选。软门控不等于稀疏计算；未做速度提升声明。']
 if PHASE==413:lines+=['','所有组都有同样的额外原始RGB和语义辅助损失；仅组内对照才能区分具体机制。参考检索、位移和照明系数均为学习代理，未验证其物理真值。当前没有云高/太阳角输入，也没有云—影实例配对标签。']
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 save('REPORT_COMPLETE.json',{'passed':True,'seconds':time.monotonic()-start,'files':{n:sha(R/n) for n in ['RESULT.md','TARGET_SUMMARY.csv','TARGET_GROUPS.csv','COST.csv','COMPARISONS.csv','FULL_COMPARISONS.csv','INDEPENDENT_ACCEPTANCE.json']}})
if __name__=='__main__':main()

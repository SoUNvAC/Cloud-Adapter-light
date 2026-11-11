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
  assert all(refscope['frozen_SHA'][k]==v for k,v in scope['frozen_SHA'].items())
  assert all(refscope['frozen_SHA'][k]==v for k,v in scope['source_interaction_initial_SHA'].items())
  assert scope['head_initial_SHA']==refscope['head_initial_SHA']
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
  assert all(torch.isfinite(v).all() for k,v in ck['state_dict'].items() if k.startswith(('backbone.cloud_adapter.msre.','backbone.cloud_adapter.detail.','backbone.cloud_adapter.net.')))
  initial=scope['source_interaction_initial_SHA']
  changed=[]
  for name,old_hash in initial.items():
   tensor=ck['state_dict'][name].detach().cpu().contiguous()
   if hashlib.sha256(tensor.numpy().tobytes()).hexdigest()!=old_hash:changed.append(name)
  assert (len(changed)==0)==(mode=='baseline')
  if mode=='integrated':
   for name in ['out','feedback']:
    assert ck['state_dict'][f'backbone.cloud_adapter.detail.{name}.weight'].abs().sum()>0
  checks.append({'mode':mode,'checkpoint_SHA':sha(sr/f'work_dirs/formal/M/iter_{steps}.pth'),'paired_trace':True,'declared_frozen_scope_verified':True,'changed_source_interaction_tensors':len(changed),'AP_scores_hash_verified':True})
 writecsv(R/'TARGET_SUMMARY.csv',rows);writecsv(R/'TARGET_GROUPS.csv',groups);writecsv(R/'COST.csv',costs)
 endpoint={r['mode']:r for r in rows if r['step']==1000 and not r['mode'].endswith('_full')}
 baseline=js(reference/'evaluations/M/step_1000/METRICS.json')['mIoU']
 comparisons=[]
 for mode,control in [('unfrozen','baseline'),('integrated','baseline'),('integrated','unfrozen')]:
  a,b=endpoint[mode],endpoint[control]
  comparisons.append({'mode':mode,'control':control,'delta_mIoU':a['mIoU']-b['mIoU'],'delta_shadow_IoU':a['Shadow_IoU']-b['Shadow_IoU'],'delta_macro_AP':a['macro_AP']-b['macro_AP']})
 accepted={'passed':True,'checks':checks,'calls':sum(c['calls'] for c in costs),'updates':sum(c['steps'] for c in costs),'CPU_report_seconds':time.monotonic()-start,'scope':'DINO/CNN/head frozen; attention trainable in unfrozen/integrated'}
 save('INDEPENDENT_ACCEPTANCE.json',accepted);writecsv(R/'COMPARISONS.csv',comparisons)
 save('MECHANISM_SCREEN.json',{'comparisons':comparisons,'scope':'single seed fixed endpoint full integrated package'})
 lines=['# Phase427 结果','','153 fit / 完整32 development，seed40901；固定1000步为主。','', '|方案|步数|mIoU|地表IoU|云IoU|阴影IoU|阴影宏AP|','|---|---:|---:|---:|---:|---:|---:|']
 for row in rows:
  m=js(R/row['mode']/f"evaluations/M/step_{row['step']}/METRICS.json")
  lines.append(f"|{row['mode']}|{row['step']}|{row['mIoU']:.4f}|{m['IoU'][0]:.4f}|{m['IoU'][1]:.4f}|{row['Shadow_IoU']:.4f}|{row['macro_AP']:.4f}|")
 for c in comparisons:lines.append(f"\n{c['mode']} 相对 {c['control']}：mIoU {c['delta_mIoU']:+.4f}点；阴影IoU {c['delta_shadow_IoU']:+.4f}点；宏AP {c['delta_macro_AP']:+.4f}点。")
 lines+=['',f"共{accepted['calls']}次前向、{accepted['updates']}次更新，GPU阶段累计{sum(c['seconds'] for c in costs)/60:.2f}分钟。",'', '|方案|可训练参数|平均训练步秒数|','|---|---:|---:|']
 for c in costs:lines.append(f"|{c['mode']}|{c['parameters']}|{c['mean_step_seconds']:.4f}|")
 lines+=['','MsRE已归属Cloud-Adapter，融合组将1/4尺度局部细节送入多尺度残差，并把更新语义反馈空间缓存。原分割损失，无辅助监督。','unfrozen/integrated允许原交互层训练；DINOv2、CNN、分割头冻结。共同初始MsRE、Source交互权重、输入/标签/增强/LR配对与完整预测/AP复算通过。单seed开发集探索，未触及确认集。']
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 save('REPORT_COMPLETE.json',{'passed':True,'files':{n:sha(R/n) for n in ['RESULT.md','TARGET_SUMMARY.csv','TARGET_GROUPS.csv','COST.csv','COMPARISONS.csv','INDEPENDENT_ACCEPTANCE.json','MECHANISM_SCREEN.json']}})
if __name__=='__main__':main()

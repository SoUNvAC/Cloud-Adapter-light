"""Independent paired-training and target-selection audit; source is descriptive."""
from pathlib import Path
import json,csv,hashlib,statistics
from finalize_phase402 import verify,close,metrics
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase410/phase410_20261101'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def table(p,rr):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def main():
 for n,h in js(R/'TRANSFER_ACCEPTED.json').items():assert sha(R/n)==h,n
 end=js(R/'COMPLETE.json');assert end['passed'] and end['calls']==9886 and end['updates']==8240 and end['seconds']<7200
 recovery=js(R/'REPORT_RECOVERY.json');assert recovery['passed'] and recovery['additional_model_forwards']==0 and recovery['additional_training_updates']==0 and recovery['calls_before']==recovery['calls_after']==9886
 assert sha(ROOT/'src/tools/recover_phase410_report.py')==recovery['recovery_code_SHA']
 fix=js(R/'CONTROLLER_FIX.json');assert sha(ROOT/'src/tools/controller_phase410.py')==fix['corrected_local_SHA'] and sha(R/'executed_code/controller_phase410.py')==fix['executed_original_SHA']
 assert js(R/'GATE_SELF_TEST.json')['passed']
 gate=js(R/'GATE_CHECKPOINT_AUDIT.json');assert gate['passed'] and gate['additional_model_forwards']==0 and all(gate['initial_state_reconstructed'].values())
 lock=js(R/'LOCK.json')
 for n,h in lock['new_files'].items():
  p=R/n if n.endswith('.md') else ROOT/'src/tools'/n
  if n=='controller_phase410.py':p=R/'executed_code'/n
  assert sha(p)==h,n
 summary=rows(R/'TARGET_SUMMARY.csv');groups=rows(R/'TARGET_GROUPS.csv');assert len(summary)==16 and len(groups)==512
 inp=js(ROOT/'outputs/phase400/phase400_20261031/delivery/INPUT_LOCK.json')
 expected={x['product']:x['label_counts'] for x in inp['target_rows'] if x['split']=='development_val'}
 initial=[];traces={};selected=[];resources=[]
 for mode in ['baseline','adaptive']:
  p=R/mode;scope=js(p/'formal_ACTIVE_SCOPE.json');short_scope=js(p/'short_ACTIVE_SCOPE.json')
  assert scope['count']==(355588 if mode=='adaptive' else 354433) and all(n.startswith('backbone.target_msre.') for n in scope['names'])
  assert scope['MsRE_initial_SHA']==short_scope['MsRE_initial_SHA'];initial.append(scope['common_MsRE_initial_SHA'])
  receipt=js(p/'formal_FREEZE_4000.json');assert receipt['passed'] and receipt['active_gradient_tensors_seen']==len(scope['names'])
  assert js(p/'short_FREEZE_100.json')['passed']
  trace=[json.loads(x) for x in (p/'formal_STEP_TRACE.jsonl').read_text().splitlines()];short=[json.loads(x) for x in (p/'short_STEP_TRACE.jsonl').read_text().splitlines()]
  assert len(trace)==4000 and len(short)==120
  assert [x['step'] for x in trace]==list(range(1,4001)) and [x['draw'] for x in trace]==list(range(4000))
  assert all(x['augmentation_seed']==int.from_bytes(hashlib.sha256(f"4090103:{x['draw']}".encode()).digest()[:4],'little') for x in trace)
  keys=['draw','product','image_SHA','label_SHA','augmentation_seed','LR']
  assert all(all(a[k]==b[k] for k in keys) for a,b in zip(short,trace));traces[mode]=trace
  ledger=[json.loads(x) for x in (p/'TRAIN_CALL_LEDGER.jsonl').read_text().splitlines()]
  assert len(ledger)==4943
  assert {k:sum(x['kind']==k for x in ledger) for k in ['train_full_input','development_full_input','source_val_full_input']}=={'train_full_input':4120,'development_full_input':288,'source_val_full_input':535}
  for stage in {x['stage'] for x in ledger}:
   part=[x for x in ledger if x['stage']==stage];assert [x['ordinal_within_stage'] for x in part]==list(range(1,len(part)+1))
  curve=js(p/'CURVE.json');assert [x['step'] for x in curve]==list(range(500,4001,500))
  for step in range(500,4001,500):
   row=next(v for v in summary if v['mode']==mode and int(v['step'])==step)
   subset=[v for v in groups if v['mode']==mode and int(v['step'])==step];assert len(subset)==32 and {v['name'] for v in subset}==set(expected)
   cm=[[0]*3 for _ in range(3)]
   for v in subset:
    c=json.loads(v['confusion']);verify(v,c);assert [sum(z) for z in c]==expected[v['name']]
    for i in range(3):
     for j in range(3):cm[i][j]+=c[i][j]
   assert cm==json.loads(row['confusion']);verify(row,cm)
   assert cm==next(v for v in curve if v['step']==step)['confusion']
   aps=[float(v['AP']) for v in subset if v['AP']];assert len(aps)==7;close(sum(aps)/7,row['macro_AP'])
  best=max([v for v in summary if v['mode']==mode],key=lambda v:(float(v['mIoU']),-int(v['step'])))
  selection=js(p/'SELECTION.json');assert int(best['step'])==selection['best']['step'];close(best['mIoU'],selection['best']['mIoU'])
  source=js(p/'SOURCE_METRICS.json');assert source['passed'] and source['source_branch']=='on' and source['checkpoint_SHA']==selection['checkpoint_SHA']==js(p/'COMPLETE.json')['checkpoint_SHA']
  cm=[[0]*3 for _ in range(3)];source_expected={v['name']:v['parent_counts'] for v in inp['source_val']}
  assert len(source['per_input'])==535 and {v['name'] for v in source['per_input']}==set(source_expected)
  for v in source['per_input']:
   c=v['confusion'];assert [sum(z) for z in c]==source_expected[v['name']]
   for i in range(3):
    for j in range(3):cm[i][j]+=c[i][j]
  assert cm==source['confusion'];close(metrics(cm)['mIoU'],source['mIoU'])
  for role,point in [('best',best),('final',next(v for v in summary if v['mode']==mode and int(v['step'])==4000))]:
   selected.append({'mode':mode,'role':role,'step':int(point['step']),'trainable_parameters':scope['count'],'target_mIoU':float(point['mIoU']),'IoU':point['IoU'],'Shadow_IoU':float(point['Shadow_IoU']),'macro_AP':float(point['macro_AP']),'pooled_AP':float(point['pooled_AP']),'source_mIoU':source['mIoU'] if role=='best' else ''})
  timing=js(p/'formal_M_COMPLETE.json')
  costs=[js(f) for f in (p/'costs').glob('*.json')];assert len(costs)==3 and all(c['status']=='passed' for c in costs)
  assert sum(c['calls'] for c in costs)==4943
  peak=max(sample['GPU_peak_reserved'] for c in costs for sample in c['resources'])
  resources.append({'mode':mode,'formal_worker_seconds':timing['wall'],'mean_step_seconds':timing['mean_step_seconds'],'mean_GPU_stream_ms':timing['GPU_stream_ms_mean'],'mean_host_gap_seconds':timing['host_gap_mean'],'peak_reserved_bytes':peak})
 assert initial[0]==initial[1]
 a_scope=js(R/'baseline/formal_ACTIVE_SCOPE.json');b_scope=js(R/'adaptive/formal_ACTIVE_SCOPE.json')
 assert a_scope['frozen_SHA']==b_scope['frozen_SHA'] and a_scope['head_initial_SHA']==b_scope['head_initial_SHA']
 keys=['step','draw','product','image_SHA','label_SHA','augmentation_seed']
 assert all(all(a[k]==b[k] for k in keys) for a,b in zip(traces['baseline'],traces['adaptive']))
 assert all(a['LR']==b['LR'][:len(a['LR'])] and len(set(b['LR']))==1 for a,b in zip(traces['baseline'],traces['adaptive']))
 table(R/'summary.csv',selected);table(R/'timing.csv',resources)
 deltas=[]
 for role in ['best','final']:
  a=next(x for x in selected if x['mode']=='baseline' and x['role']==role);b=next(x for x in selected if x['mode']=='adaptive' and x['role']==role)
  deltas.append({'role':role,**{k:b[k]-a[k] for k in ['target_mIoU','Shadow_IoU','macro_AP','pooled_AP']}})
 table(R/'comparisons.csv',deltas);advance=deltas[0]['target_mIoU']>=.30
 lines=['# Phase410：MsRE自适应尺度融合对照','','## 结果','','seed40901，H关闭；原版354433、改版355588个可训练参数，共有权重同初始化、逐步输入、学习率及4000步预算。每500步验证，按目标mIoU选择best。','','|尺度融合|角色|步数|目标mIoU|Shadow IoU|宏AP|池化AP|','|---|---|---:|---:|---:|---:|---:|']
 for v in selected:lines.append(f"|{v['mode']}|{v['role']}|{v['step']}|{v['target_mIoU']:.4f}|{v['Shadow_IoU']:.4f}|{v['macro_AP']:.4f}|{v['pooled_AP']:.4f}|")
 lines+=['',f"自适应−固定平均：best目标差{deltas[0]['target_mIoU']:+.4f}，final目标差{deltas[1]['target_mIoU']:+.4f}个百分点。",'', '判定：'+('达到预设+0.30点探索晋级线，值得多seed复核。' if advance else '未达到预设+0.30点探索晋级线。'),'', '历史Phase400 M best78.1895、H best77.7668为另一seed参照，不代替本轮配对差。','', '## 成本与验收','',f"9886次完整前向、8240次更新，控制器墙钟{end['seconds']:.1f}秒；尺度门控新增1155参数。两臂耗时见timing.csv（不是独立部署延迟）。",'', '均匀初始化/门控梯度/关闭恒等路径自测、正式初始权重重建与已训练门控检查、输入与参数身份、配对训练轨迹、冻结及全部可训练参数梯度回执、调用账本、逐组和汇总指标、best选择重放通过。云影宏AP算术独立复算；池化AP未传像元分数独立复算。','', '## 源域附表（不参与判定）','']
 for v in selected:
  if v['role']=='best':lines.append(f"- {v['mode']} best：{v['source_mIoU']:.4f}% mIoU。")
 lines+=['','汇总阶段发现旧目录名残留，已保留原始失败记录及执行代码，修正本地控制器，并仅恢复CPU汇总；未增加训练更新或完整前向。','', '限制：单seed、同一已暴露development且参与选模，仅作结构筛选。未证明独立泛化、稳定性或论文创新性。未启动411。']
 if not advance:
  lines+=['','决策：本轮空间尺度门控不晋级，保留原版作为对照。后续结构实验需另行定义并固定预算。']
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (R/'INDEPENDENT_ACCEPTANCE.json').write_text(json.dumps({'passed':True,'paired_initialization_inputs_LR':True,'initial_weights_reconstructed':True,'trained_gate_nonzero':True,'calls':9886,'updates':8240,'target_based_promotion':advance,'source_is_selection_gate':False,'pooled_AP_independent_pixel_recompute':False},indent=2))
 print('\n'.join(lines))
if __name__=='__main__':main()

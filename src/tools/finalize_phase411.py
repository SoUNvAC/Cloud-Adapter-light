"""Independent paired-training and target-selection audit; source is descriptive."""
from pathlib import Path
import json,csv,hashlib,statistics
from finalize_phase402 import verify,close,metrics
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase411/phase411_20261101'
REF=ROOT/'outputs/phase410/phase410_20261101'

def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def table(p,rr):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def main():
 for n,h in js(R/'TRANSFER_ACCEPTED.json').items():assert sha(R/n)==h,n
 end=js(R/'COMPLETE.json');assert end['passed'] and end['calls']==4408 and end['updates']==4120 and end['seconds']<7200
 tests=js(R/'SCALE_SELF_TEST.json');assert tests['passed'] and tests['exact_initial_output_equal'] and tests['isolated_layer_gradients'] and tests['added_parameters']==3
 lock=js(R/'LOCK.json')
 for n,h in lock['new_files'].items():
  p=R/n if n.endswith('.md') else ROOT/'src/tools'/n
  assert sha(p)==h,n
 summary=rows(R/'TARGET_SUMMARY.csv');groups=rows(R/'TARGET_GROUPS.csv');assert len(summary)==16 and len(groups)==512
 inp=js(ROOT/'outputs/phase400/phase400_20261031/delivery/INPUT_LOCK.json')
 expected={x['product']:x['label_counts'] for x in inp['target_rows'] if x['split']=='development_val'}
 initial=[];traces={};selected=[];resources=[]
 for mode in ['baseline','layerwise']:
  p=(REF if mode=='baseline' else R)/mode;scope=js(p/'formal_ACTIVE_SCOPE.json');short_scope=js(p/'short_ACTIVE_SCOPE.json')
  assert scope['count']==(354436 if mode=='layerwise' else 354433) and all(n.startswith('backbone.target_msre.') for n in scope['names'])
  assert scope['MsRE_initial_SHA']==short_scope['MsRE_initial_SHA'];initial.append(scope['MsRE_initial_SHA']);assert scope['MsRE_initial_SHA']==tests[('baseline' if mode=='baseline' else 'layerwise')+'_initial_SHA']
  receipt=js(p/'formal_FREEZE_4000.json');assert receipt['passed'] and receipt['active_gradient_tensors_seen']==len(scope['names'])
  assert js(p/'short_FREEZE_100.json')['passed']
  trace=[json.loads(x) for x in (p/'formal_STEP_TRACE.jsonl').read_text().splitlines()];short=[json.loads(x) for x in (p/'short_STEP_TRACE.jsonl').read_text().splitlines()]
  assert len(trace)==4000 and len(short)==120
  assert [x['step'] for x in trace]==list(range(1,4001)) and [x['draw'] for x in trace]==list(range(4000))
  assert all(x['augmentation_seed']==int.from_bytes(hashlib.sha256(f"4090103:{x['draw']}".encode()).digest()[:4],'little') for x in trace)
  keys=['draw','product','image_SHA','label_SHA','augmentation_seed','LR']
  assert all(all(a[k]==b[k] for k in keys) for a,b in zip(short,trace));traces[mode]=trace
  ledger=[json.loads(x) for x in (p/'TRAIN_CALL_LEDGER.jsonl').read_text().splitlines()]
  expected_calls=4943 if mode=='baseline' else 4408
  assert len(ledger)==expected_calls
  assert {k:sum(x['kind']==k for x in ledger) for k in ['train_full_input','development_full_input','source_val_full_input']}=={'train_full_input':4120,'development_full_input':288,'source_val_full_input':535 if mode=='baseline' else 0}
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
  for role,point in [('best',best),('final',next(v for v in summary if v['mode']==mode and int(v['step'])==4000))]:
   selected.append({'mode':mode,'role':role,'step':int(point['step']),'trainable_parameters':scope['count'],'target_mIoU':float(point['mIoU']),'IoU':point['IoU'],'Shadow_IoU':float(point['Shadow_IoU']),'macro_AP':float(point['macro_AP']),'pooled_AP':float(point['pooled_AP'])})
  timing=js(p/'formal_M_COMPLETE.json')
  costs=[js(f) for f in (p/'costs').glob('*.json')];assert len(costs)==(3 if mode=='baseline' else 2) and all(c['status']=='passed' for c in costs)
  assert sum(c['calls'] for c in costs)==expected_calls
  peak=max(sample['GPU_peak_reserved'] for c in costs for sample in c['resources'])
  resources.append({'mode':mode,'formal_worker_seconds':timing['wall'],'mean_step_seconds':timing['mean_step_seconds'],'mean_GPU_stream_ms':timing['GPU_stream_ms_mean'],'mean_host_gap_seconds':timing['host_gap_mean'],'peak_reserved_bytes':peak})
 assert js(R/'layerwise/formal_ACTIVE_SCOPE.json')['common_MsRE_initial_SHA']==tests['common_initial_SHA']
 a_scope=js(REF/'baseline/formal_ACTIVE_SCOPE.json');b_scope=js(R/'layerwise/formal_ACTIVE_SCOPE.json')
 assert a_scope['frozen_SHA']==b_scope['frozen_SHA'] and a_scope['head_initial_SHA']==b_scope['head_initial_SHA']
 keys=['step','draw','product','image_SHA','label_SHA','augmentation_seed']
 assert all(all(a[k]==b[k] for k in keys) for a,b in zip(traces['baseline'],traces['layerwise']))
 assert all(a['LR']==b['LR'][:len(a['LR'])] and len(set(b['LR']))==1 for a,b in zip(traces['baseline'],traces['layerwise']))
 table(R/'summary.csv',selected);table(R/'timing.csv',resources)
 deltas=[]
 for role in ['best','final']:
  a=next(x for x in selected if x['mode']=='baseline' and x['role']==role);b=next(x for x in selected if x['mode']=='layerwise' and x['role']==role)
  deltas.append({'role':role,**{k:b[k]-a[k] for k in ['target_mIoU','Shadow_IoU','macro_AP','pooled_AP']}})
 table(R/'comparisons.csv',deltas);advance=deltas[0]['target_mIoU']>=.30
 scales=js(R/'layerwise/SCALES.json');assert len(scales)==2
 assert next(x for x in scales if x['role']=='best')['checkpoint_SHA']==js(R/'layerwise/SELECTION.json')['checkpoint_SHA']
 assert next(x for x in scales if x['role']=='final')['values']==js(R/'layerwise/formal_FREEZE_4000.json')['MsRE_scale']
 assert all(len(x['values'])==4 for x in scales)
 original=rows(REF/'TARGET_SUMMARY.csv');assert [v for v in summary if v['mode']=='baseline']==[v for v in original if v['mode']=='baseline']
 # The archived control is verified against its sealed local transfer, not counted as a new run.
 for n,h in js(REF/'TRANSFER_ACCEPTED.json').items():
  if n.startswith('baseline/') or n in ['TARGET_SUMMARY.csv','TARGET_GROUPS.csv']:assert sha(REF/n)==h,n
 lines=['# Phase411：MsRE分层残差控制','','原版复用Phase410已验收seed40901；改版同seed、共有权重、逐步输入、学习率及4000步预算。H关闭，原串联；标量系数改为4层独立系数，净增3参数。源域不作为门槛，本轮未重复评价。','','|模型|角色|步数|目标mIoU|Shadow IoU|宏AP|池化AP|','|---|---|---:|---:|---:|---:|---:|']
 for v in selected:lines.append(f"|{v['mode']}|{v['role']}|{v['step']}|{v['target_mIoU']:.4f}|{v['Shadow_IoU']:.4f}|{v['macro_AP']:.4f}|{v['pooled_AP']:.4f}|")
 lines+=['',f"分层−原版：best {deltas[0]['target_mIoU']:+.4f}，final {deltas[1]['target_mIoU']:+.4f} 个百分点。",'', '判定：'+('达到事前+0.30点晋级线，值得多seed复核。' if advance else '未达到事前+0.30点晋级线，不作为性能改进推广。'),'', '|检查点|block2|block5|block8|block11|','|---|---:|---:|---:|---:|']
 for x in scales:lines.append('|'+x['role']+'|'+ '|'.join(f'{v:.6f}' for v in x['values'])+'|')
 base_best=next(v for v in selected if v['mode']=='baseline' and v['role']=='best');layer_best=next(v for v in selected if v['mode']=='layerwise' and v['role']=='best')
 lines+=['',f"云影解释：best的Shadow IoU从{base_best['Shadow_IoU']:.4f}变为{layer_best['Shadow_IoU']:.4f}，宏AP从{base_best['macro_AP']:.4f}变为{layer_best['macro_AP']:.4f}。总体峰值收益不能称为云影改善；后期回落未解决。历史Phase400 M的78.1895（另一seed）也未被超过。"]
 lines+=['','四层初值均0.001；系数不同不等于某层因果重要性，需要消融验证。', '', f"新增4408次完整前向、4120次更新，墙钟{end['seconds']:.1f}秒。复用对照的调用不计入新增预算。", '', '初始输出严格等价、分层独立梯度、正式初始化重建、配对输入/学习率、训练范围/冻结、调用账本、逐组混淆/宏AP及best选择验收通过。池化AP未传像元分数独立复算。', '', '限制：单seed、已暴露development参与选模；复用历史同seed对照，不是同期新跑双臂；CUDA仍有非确定性算子。Phase409同seed原版best77.0094可作复现波动参照，不能替代多seed。', '', '云影特有结构建议另见CLOUD_SHADOW_ARCHITECTURE.md。本轮未训练这些建议。']
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (R/'INDEPENDENT_ACCEPTANCE.json').write_text(json.dumps({'passed':True,'paired_initialization_inputs_LR':True,'initial_weights_reconstructed':True,'calls':4408,'updates':4120,'reused_baseline':'phase410/baseline','target_based_promotion':advance,'source_is_selection_gate':False,'pooled_AP_independent_pixel_recompute':False},indent=2))
 print('\n'.join(lines))
if __name__=='__main__':main()

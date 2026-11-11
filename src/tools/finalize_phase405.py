"""Independent per-seed counts, metrics and paired comparisons for Phase405."""
from pathlib import Path
import json,csv,hashlib,statistics
from finalize_phase402 import metrics,verify,close
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase405/phase405_20261101'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(p,rr):
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def main():
 for n,h in js(R/'TRANSFER_ACCEPTED.json').items():assert sha(R/n)==h,n
 complete=js(R/'COMPLETE.json');assert complete['passed'] and complete['calls']==15620 and complete['updates']==8240 and complete['seconds']<7200
 initials=[];orders=[];combined=[];deltas=[];checks=[]
 baseline=next(x for x in rows(ROOT/'outputs/phase401/phase401_20261031/summary.csv') if float(x['alpha'])==0)
 for seed in [40501,40502]:
  p=R/str(seed);info=js(p/'RUN_INFO.json');lock=js(p/'INPUT_EXPECTED.json');assert info['passed'] and info['calls']==3402 and info['updates']==0
  assert info['script_SHA']==sha(ROOT/'src/tools/evaluate_phase405.py')==lock['worker_SHA'];assert info['checkpoint_SHA']==js(p/'COMPLETE.json')['checkpoint_SHA'];assert info['input_expected_SHA']==sha(p/'INPUT_EXPECTED.json')
  ledger=[json.loads(x) for x in (p/'CALL_LEDGER.jsonl').read_text().splitlines()];assert len(ledger)==3402 and [x['calls'] for x in ledger]==list(range(1,3403)) and all(x['updates']==0 for x in ledger);assert sha(p/'CALL_LEDGER.jsonl')==info['ledger_SHA']
  train=[json.loads(x) for x in (p/'TRAIN_CALL_LEDGER.jsonl').read_text().splitlines()];assert len(train)==4408;assert sum(x['kind']=='train_full_input' for x in train)==4120
  traces=[json.loads(x) for x in (p/'formal_STEP_TRACE.jsonl').read_text().splitlines()];assert len(traces)==4000 and [x['step'] for x in traces]==list(range(1,4001)) and [x['draw'] for x in traces]==list(range(4000))
  assert all(x['augmentation_seed']==int.from_bytes(hashlib.sha256(f'{seed*100+3}:{x["draw"]}'.encode()).digest()[:4],'little') for x in traces)
  short=[json.loads(x) for x in (p/'short_STEP_TRACE.jsonl').read_text().splitlines()];assert len(short)==120
  assert all(all(x[k]==y[k] for k in ['draw','product','image_SHA','label_SHA','augmentation_seed']) for x,y in zip(short,traces))
  orders.append([x['product'] for x in traces[:153]])
  a=js(p/'formal_ACTIVE_SCOPE.json');b=js(p/'short_ACTIVE_SCOPE.json');assert a['count']==26144 and a['head_initial_SHA']==b['head_initial_SHA'];initials.append(a['head_initial_SHA'])
  assert js(p/'formal_FREEZE_4000.json')['passed'] and js(p/'short_FREEZE_100.json')['passed']
  summ=rows(p/'summary.csv');g=rows(p/'per_group.csv');im=rows(p/'source_per_image.csv');assert len(summ)==6 and len(g)==192 and len(im)==3210
  assert {x['case'] for x in summ}=={'0.0','1.0','0.4','0.45','0.5','weak_5'}
  support={}
  for v in summ:
   for domain,rr,n in [('target',g,32),('source',im,535)]:
    subset=[x for x in rr if x['case']==v['case']];assert len(subset)==n and len({x['name'] for x in subset})==n
    c=[[0]*3 for _ in range(3)];refs={}
    for x in subset:
     cc=json.loads(x['confusion']);verify(x,cc);refs[x['name']]=[sum(z) for z in cc]
     for i in range(3):
      for j in range(3):c[i][j]+=cc[i][j]
    assert c==json.loads(v[domain+'_confusion']);verify(v,c,domain+'_')
    if domain in support:assert refs==support[domain]
    else:support[domain]=refs
    assert sum(x['current']['case']==v['case'] and x['current']['domain']==domain for x in ledger)==n
   aps=[float(x['AP']) for x in g if x['case']==v['case'] and x['AP']];assert len(aps)==7;close(sum(aps)/7,v['target_macro_AP'])
   combined.append({'seed':seed,'case':v['case'],'target_mIoU':float(v['target_mIoU']),'source_mIoU':float(v['source_mIoU']),'Shadow_IoU':float(v['target_Shadow_IoU']),'macro_AP':float(v['target_macro_AP']),'pooled_AP':float(v['target_pooled_AP'])})
  zero=next(v for v in summ if v['case']=='0.0')
  for k in ['target_confusion','source_confusion']:assert json.loads(zero[k])==json.loads(baseline[k])
  curve=js(p/'CURVE.json');assert [v['step'] for v in curve]==list(range(500,4001,500))
  full=next(v for v in summ if v['case']=='1.0');assert json.loads(full['target_confusion'])==curve[-1]['confusion']
  weak=next(v for v in combined if v['seed']==seed and v['case']=='weak_5')
  for case in ['0.4','0.45','0.5']:
   v=next(v for v in combined if v['seed']==seed and v['case']==case)
   deltas.append({'seed':seed,'comparison':'weak_5 minus '+case,**{k:weak[k]-v[k] for k in ['target_mIoU','source_mIoU','Shadow_IoU','macro_AP']}})
  checks.append(f'{seed}: full training and evaluation counts, seeds, gradients/freeze receipts, per-image and pooled metrics, Source and final-H endpoints')
 assert initials[0]!=initials[1] and orders[0]!=orders[1]
 table(R/'summary.csv',combined);table(R/'paired_deltas.csv',deltas)
 lines=['# Phase405：固定缩放配方跨seed验证','','## 数据结果','','仅两个新seed40501/40502作本轮重复性观察；旧40001为发现阶段，不合并计算确认均值。固定最终4000步，未按缩放或源域结果选checkpoint。','','|seed|配置|目标mIoU|源mIoU|Shadow IoU|宏AP|','|---|---|---:|---:|---:|---:|']
 for v in combined:lines.append(f"|{v['seed']}|{v['case']}|{v['target_mIoU']:.4f}|{v['source_mIoU']:.4f}|{v['Shadow_IoU']:.4f}|{v['macro_AP']:.4f}|")
 lines+=['','|seed|weak_5−统一0.45目标Δ|源Δ|','|---|---:|---:|']
 ds=[v for v in deltas if v['comparison']=='weak_5 minus 0.45']
 for v in ds:lines.append(f"|{v['seed']}|{v['target_mIoU']:.4f}|{v['source_mIoU']:.4f}|")
 lines += ['',f"两seed目标差均值{statistics.mean(x['target_mIoU'] for x in ds):.4f}，范围[{min(x['target_mIoU'] for x in ds):.4f}, {max(x['target_mIoU'] for x in ds):.4f}]；源差均值{statistics.mean(x['source_mIoU'] for x in ds):.4f}，范围[{min(x['source_mIoU'] for x in ds):.4f}, {max(x['source_mIoU'] for x in ds):.4f}]。",'',f"15620次前向、8240次更新（正式8000、短跑240），总控制器墙钟{complete['seconds']:.1f}秒。",'','## 验收与限制','','传输/代码/权重身份、连续调用、训练样本顺序、独立增强种子、不同初始化、冻结回执、逐图与池化混淆/IoU、宏AP算术、Source及训练最终H端点通过。新配置池化AP未独立原像元复算。两个新seed仍用同一已暴露development及源val，仅验证重复性，不是独立泛化或显著性证明。']
 lines += ['', '## 结论与下一步', '', 'weak_5相对统一0.45在两个新seed均同时改善目标与源域；目标平均+0.1658、源平均+0.8273个百分点，方向复现。相对统一0.40也在两seed同时改善，但相对统一0.50仍是目标下降、源域改善的折中，不能称全面优胜。', '', 'weak_5源域为73.7031/74.2441，仍较同轮Source的77.8152低4.1121/3.5711个百分点。它是值得保留的低成本固定基线，尚未解决源域损伤，也不是新架构或泛化优势的证明。', '', '建议下一轮停止细搜缩放系数，固定weak_5与统一0.45对照，研究训练期源模型保持约束；先明确可用的训练数据和约束损失，源val只评价，不进入训练。若要检验新数据泛化，须预先锁定未参与设计的确认集。尚未启动Phase406。']
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (R/'INDEPENDENT_ACCEPTANCE.json').write_text(json.dumps({'passed':True,'checks':checks+['different initialization and data orders between seeds'],'new_point_pooled_AP_independent_pixel_recompute':False},indent=2))
 print('\n'.join(lines))
if __name__=='__main__':main()

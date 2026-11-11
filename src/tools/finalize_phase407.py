"""Independent replay of target-only selection and pooled metrics."""
from pathlib import Path
import json,csv,hashlib,statistics
from finalize_phase402 import verify,close
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase407/phase407_20261101'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def table(p,rr):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def main():
 for n,h in js(R/'TRANSFER_ACCEPTED.json').items():assert sha(R/n)==h,n
 lock=js(R/'LOCK.json');complete=js(R/'COMPLETE.json')
 assert complete['passed'] and complete['calls']==3420 and complete['updates']==0 and complete['seconds']<3600
 combined=[];all_probes=[];checks=[]
 for job in lock['jobs']:
  p=R/job['name'];info=js(p/'RUN_INFO.json');sel=js(p/'SELECTION.json');expected=js(p/'INPUT_EXPECTED.json')
  assert info['passed'] and info['calls']==855 and info['updates']==0
  assert info['checkpoint_SHA']==job['checkpoint_SHA']==expected['H_SHA']
  assert info['Source_SHA']==lock['Source_SHA']
  assert info['script_SHA']==expected['worker_SHA']==sha(ROOT/'src/tools/evaluate_phase407.py')
  assert info['input_expected_SHA']==sha(p/'INPUT_EXPECTED.json') and info['ledger_SHA']==sha(p/'CALL_LEDGER.jsonl')
  ledger=[json.loads(x) for x in (p/'CALL_LEDGER.jsonl').read_text().splitlines()]
  assert len(ledger)==855 and [v['calls'] for v in ledger]==list(range(1,856))
  assert all(v['updates']==0 for v in ledger)
  assert all(v['current']['domain']=='target' for v in ledger[:320])
  assert all(v['current']['domain']=='source' and v['current']['case']=='selected' for v in ledger[320:])
  assert sel['calls_before_selection']==288 and sel['source_calls_before_selection']==0
  ss=rows(p/'summary.csv');gg=rows(p/'per_group.csv');im=rows(p/'source_per_image.csv')
  cases=['reference_0.5']+[f'probe_{i:02d}' for i in range(8)]+['selected']
  assert [x['case'] for x in ss]==cases and len(gg)==320 and len(im)==535
  support={}
  for v in ss:
   case=v['case']
   for domain,rr,n in [('target',gg,32)]+([('source',im,535)] if case=='selected' else []):
    subset=[x for x in rr if x['case']==case];assert len(subset)==n and len({x['name'] for x in subset})==n
    cm=[[0]*3 for _ in range(3)];counts={}
    for x in subset:
     cc=json.loads(x['confusion']);verify(x,cc);counts[x['name']]=[sum(row) for row in cc]
     for i in range(3):
      for j in range(3):cm[i][j]+=cc[i][j]
    assert cm==json.loads(v[domain+'_confusion']);verify(v,cm,domain+'_')
    if domain in support:assert counts==support[domain]
    else:support[domain]=counts
    assert sum(x['current']['case']==case and x['current']['domain']==domain for x in ledger)==n
   aps=[float(x['AP']) for x in gg if x['case']==case and x['AP']];assert len(aps)==7;close(sum(aps)/7,v['target_macro_AP'])
   if case!='selected':assert not v['source_mIoU']
  previous=ROOT/f"outputs/phase{job['phase']}/phase{job['phase']}_20261101"
  manifest=js(previous/'LOCAL_MANIFEST.json');prior=previous/str(job['seed'])
  for n in ['per_group.csv','summary.csv','source_per_image.csv']:
   assert sha(prior/n)==manifest[f"{job['seed']}/{n}"]
  assert sha(prior/'per_group.csv')==job['groups_SHA'] and sha(prior/'summary.csv')==job['summary_SHA']
  refs={x['name']:x for x in rows(prior/'per_group.csv') if x['case']=='0.5'}
  for v in gg:
   if v['case']=='reference_0.5':
    assert v['prediction_SHA']==refs[v['name']]['prediction_SHA']
    assert json.loads(v['confusion'])==json.loads(refs[v['name']]['confusion'])
   if v['case']=='selected':
    ref=next(x for x in gg if x['case']==sel['case'] and x['name']==v['name'])
    assert v['prediction_SHA']==ref['prediction_SHA'] and json.loads(v['confusion'])==json.loads(ref['confusion'])
  source_refs={x['name']:[sum(z) for z in json.loads(x['confusion'])] for x in rows(prior/'source_per_image.csv') if x['case']=='0.5'}
  assert source_refs==support['source']
  weakroot=ROOT/'outputs/phase405/phase405_20261101'
  assert sha(weakroot/str(job['seed'])/'summary.csv')==js(weakroot/'LOCAL_MANIFEST.json')[f"{job['seed']}/summary.csv"]
  weak=next(x for x in rows(weakroot/str(job['seed'])/'summary.csv') if x['case']=='weak_5')
  reference=float(weak['target_mIoU']);close(reference,job['target_reference'])
  lo,hi=job['bracket'];old=rows(prior/'summary.csv')
  assert float(next(x for x in old if x['case']!='weak_5' and float(x['case'])==lo)['target_mIoU'])<reference<float(next(x for x in old if x['case']!='weak_5' and float(x['case'])==hi)['target_mIoU'])
  probes=ss[1:9]
  for v in probes:
   mid=(lo+hi)/2;close(mid,v['alpha'])
   if float(v['target_mIoU'])<reference:lo=mid
   else:hi=mid
   all_probes.append({'job':job['name'],'case':v['case'],'alpha':v['alpha'],'target_mIoU':v['target_mIoU'],'target_error':float(v['target_mIoU'])-reference})
  chosen=min(probes,key=lambda v:(abs(float(v['target_mIoU'])-reference),float(v['alpha'])))
  assert chosen['case']==sel['case'];close(chosen['alpha'],sel['alpha']);close(chosen['target_mIoU'],sel['target_mIoU'])
  final=ss[-1];close(final['alpha'],sel['alpha']);close(final['target_mIoU'],sel['target_mIoU'])
  error=float(final['target_mIoU'])-reference;close(abs(error),sel['absolute_error'])
  combined.append({'seed':job['seed'],'phase':job['phase'],'alpha':sel['alpha'],'target_mIoU':float(final['target_mIoU']),'source_mIoU':float(final['source_mIoU']),'target_error':error,'matched':abs(error)<=.05,'target_macro_AP':float(final['target_macro_AP']),'target_Shadow_IoU':float(final['target_Shadow_IoU']),'weak5_target':reference,'weak5_source':float(weak['source_mIoU'])})
  checks.append(job['name']+':855 calls; target-only8 selection replay; prediction parity; pooled metrics')
 table(R/'matched_summary.csv',combined);table(R/'target_probes.csv',all_probes)
 paired=[]
 for seed in [40501,40502]:
  a=next(x for x in combined if x['seed']==seed and x['phase']==405);b=next(x for x in combined if x['seed']==seed and x['phase']==406)
  paired.append({'seed':seed,'both_matched':a['matched'] and b['matched'],'target_delta_406_minus_405':b['target_mIoU']-a['target_mIoU'],'source_delta_406_minus_405':b['source_mIoU']-a['source_mIoU'],'source_delta_406_minus_weak5':b['source_mIoU']-b['weak5_source']})
 table(R/'paired_deltas.csv',paired)
 passed=all(x['both_matched'] and x['source_delta_406_minus_405']>0 for x in paired)
 lines=['# Phase407：目标精度对齐后比较源域','','固定405/406最终权重，零训练。目标参照为各seed原405 weak_5；alpha只用32组development目标mIoU选择，选定后才评价535张源图。','','|seed|模型|alpha|目标mIoU|源mIoU|目标偏差|','|---|---|---:|---:|---:|---:|']
 for seed in [40501,40502]:
  ref=next(x for x in combined if x['seed']==seed)
  lines.append(f"|{seed}|405 weak_5|分层|{ref['weak5_target']:.4f}|{ref['weak5_source']:.4f}|0|")
  for v in [x for x in combined if x['seed']==seed]:lines.append(f"|{seed}|{v['phase']}统一|{v['alpha']:.8f}|{v['target_mIoU']:.4f}|{v['source_mIoU']:.4f}|{v['target_error']:+.4f}|")
 lines+=['','|seed|406−405统一：目标差|源差|406−原weak_5：源差|','|---|---:|---:|---:|']
 for v in paired:lines.append(f"|{v['seed']}|{v['target_delta_406_minus_405']:+.4f}|{v['source_delta_406_minus_405']:+.4f}|{v['source_delta_406_minus_weak5']:+.4f}|")
 lines+=['','主要判据：'+('通过。两个seed均匹配目标，且406源域优于405统一缩放。' if passed else '未通过。未在两个seed均达到目标匹配且406源域改善。'),'',f"406−405统一的平均源差{statistics.mean(x['source_delta_406_minus_405'] for x in paired):+.4f}个百分点；406−原weak_5平均源差{statistics.mean(x['source_delta_406_minus_weak5'] for x in paired):+.4f}个百分点。",'',f"3420次完整前向、0更新，控制器墙钟{complete['seconds']:.1f}秒。",'','验收：四模型参考0.5与选点预测复现、权重/输入/代码/传输身份、8步选点重放、源评价晚于选点、逐图与汇总指标、宏AP算术均通过。新点池化AP未独立像元复算。','','限制：目标development参与选alpha，源val已在前轮暴露。这是同批数据上的局部操作点比较，不证明全局最优、独立泛化或显著性。0.05点为预设匹配容差，仍报告实际差值；未启动408。']
 if not passed and all(x['source_delta_406_minus_weak5']<0 for x in paired):
  lines+=['','决策：暂停当前系数1、四层统一残差能量惩罚路线，不继续细搜其系数或alpha。Phase406全强度双域改善未转化为本轮同目标精度的一致源域收益，原405 weak_5在两个seed均保留更多源域性能。保留weak_5作固定强基线；下一轮若继续，优先检验分层训练设计，另行锁定小规模配对实验。此结论仅针对本轮配置与操作区间，不排除其他约束。']
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (R/'INDEPENDENT_ACCEPTANCE.json').write_text(json.dumps({'passed':True,'checks':checks,'scientific_criterion_passed':passed,'all_targets_matched':all(x['matched'] for x in combined),'new_point_pooled_AP_independent_pixel_recompute':False},indent=2))
 print('\n'.join(lines))
if __name__=='__main__':main()

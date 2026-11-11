"""CPU-only independent count/endpoint acceptance and complete diagnostic delivery."""
from pathlib import Path
import csv,json,hashlib,math,time,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase401/phase401_20261031';OLD=ROOT/'outputs/phase400/phase400_20261031/delivery'
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def save(n,v):(R/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def close(a,b):assert abs(float(a)-float(b))<1e-8,(a,b)
def metric(c):
 c=np.asarray(c,dtype=np.int64);d=np.diag(c);u=c.sum(0)+c.sum(1)-d;i=np.divide(d,u,out=np.full(3,np.nan),where=u>0)*100
 return dict(mIoU=float(np.nanmean(i)),IoU=i,Shadow_IoU=i[2],Shadow_recall=100*d[2]/c[2].sum() if c[2].sum() else None,nonShadow_FPR=100*c[:2,2].sum()/c[:2].sum() if c[:2].sum() else None)
def verify(m,c,prefix=''):
 v=metric(c)
 for k in ('mIoU','Shadow_IoU','Shadow_recall','nonShadow_FPR'):
  if v[k] is None or not math.isfinite(v[k]):assert m[prefix+k] in ('','null','nan')
  else:close(m[prefix+k],v[k])
 for a,b in zip(json.loads(m[prefix+'IoU']),v['IoU']):
  if a is not None:close(a,b)
def main():
 assert not (R/'DELIVERY_COMPLETE.json').exists()
 for n,h in js(R/'TRANSFER_ACCEPTED.json').items():assert sha(R/n)==h,n
 info=js(R/'RUN_INFO.json');expected=js(R/'INPUT_EXPECTED.json');gate=js(R/'ENDPOINT_GATE.json');assert info['passed'] and gate['passed'];assert info['calls']==2835 and info['updates']==0 and info['seconds']<3600;assert info['checkpoint_SHA']==expected['H_SHA'];assert info['Source_SHA']==expected['Source_SHA'];assert info['script_SHA']==sha(ROOT/'src/tools/run_phase401.py')==expected['worker_SHA'];assert sha(R/'INPUT_EXPECTED.json')==info['input_expected_SHA'];assert sha(R/'CALL_LEDGER.jsonl')==info['ledger_SHA']
 ledger=[json.loads(l) for l in (R/'CALL_LEDGER.jsonl').read_text().splitlines()];assert len(ledger)==2835 and [v['calls'] for v in ledger]==list(range(1,2836));assert all(v['updates']==0 for v in ledger)
 s=rows(R/'summary.csv');g=rows(R/'per_group.csv');im=rows(R/'source_per_image.csv');assert len(s)==5 and len(g)==160 and len(im)==2675;assert sorted(float(v['alpha']) for v in s)==[0,.25,.5,.75,1]
 checks=['transfer SHA / worker / H / Source / input identity','2835 contiguous accounted input calls / zero updates / 1h bound'];oldgroups=rows(OLD/'PER_GROUP.csv');oldsummary=rows(OLD/'SUMMARY.csv');byalpha={float(v['alpha']):v for v in s};target_support=None;source_support=None
 for alpha,v in sorted(byalpha.items()):
  gg=[x for x in g if float(x['alpha'])==alpha];ss=[x for x in im if float(x['alpha'])==alpha];assert len(gg)==32 and len(ss)==535;assert len({x['group_id'] for x in gg})==32
  for domain,rr in [('target',gg),('source',ss)]:
   cm=np.zeros((3,3),np.int64)
   for x in rr:c=json.loads(x['confusion']);verify(x,c);cm+=c
   assert cm.tolist()==json.loads(v[domain+'_confusion']);verify(v,cm,domain+'_')
  ts={x['name']:json.loads(x['reference']) for x in gg};xs={x['name']:json.loads(x['reference']) for x in ss}
  if target_support is None:target_support=ts;source_support=xs
  else:assert ts==target_support and xs==source_support
  aps=[float(x['AP']) for x in gg if x['AP']];assert len(aps)==7;close(sum(aps)/7,v['target_macro_AP']);assert int(v['AP_groups'])==7
  for domain,n in [('target',32),('source',535)]:assert sum(x['current']['alpha']==alpha and x['current']['domain']==domain for x in ledger)==n
  checks.append(f'alpha{alpha}: every image/group CM/IoU/recall/FPR and equal supports, macro AP7, calls567')
 for alpha,a,step in [(0,'S',''),(1,'H','4000')]:
  old=next(v for v in oldsummary if v['arm']==a and v['domain']=='Catalogue_development' and v['role']==('baseline' if a=='S' else 'best'));v=byalpha[alpha]
  for key,ok in [('target_mIoU','mIoU'),('target_macro_AP','macro_Shadow_AP'),('target_pooled_AP','pooled_Shadow_AP')]:close(v[key],old[ok])
  ref={x['product']:x for x in oldgroups if x['arm']==a and x['step']==step}
  for x in (x for x in g if float(x['alpha'])==alpha):
   oldx=ref[x['name']];assert json.loads(x['confusion'])==json.loads(oldx['confusion'])
   if x['AP']:close(x['AP'],oldx['AP'])
  src=next(x for x in oldsummary if x['arm']==a and x['domain']=='CloudSEN_val535' and x['role']==('baseline' if a=='S' else 'best') and x['source_branch']==('baseline' if a=='S' else 'on'));assert json.loads(v['source_confusion'])==json.loads(src['confusion']);close(v['source_mIoU'],src['mIoU'])
  checks.append(f'alpha{alpha}: independently matches Phase400 full-pool CM, all32 group CM/AP, macro/pooled AP and val535 CM')
 if (ROOT/'outputs/phase400/phase400_20261031/source_val535_per_image.csv').exists():
  srcold=rows(ROOT/'outputs/phase400/phase400_20261031/source_val535_per_image.csv')
  for alpha,tag in [(0,'Source'),(1,'H_best_on')]:
   ref={x['name']:x for x in srcold if x['evaluation']==tag};assert len(ref)==535
   for x in (x for x in im if float(x['alpha'])==alpha):assert x['prediction_SHA']==ref[x['name']]['prediction_SHA'] and json.loads(x['confusion'])==json.loads(ref[x['name']]['confusion'])
  checks.append('all1070 source endpoint per-image prediction SHA independently identical to Phase400')
 gain=float(byalpha[1]['target_mIoU'])-float(byalpha[0]['target_mIoU']);loss=float(byalpha[0]['source_mIoU'])-float(byalpha[1]['source_mIoU']);assert gain>0 and loss>0;trade=[]
 for alpha,v in sorted(byalpha.items()):trade.append({**v,'alpha':alpha,'target_mIoU_gain_pp':float(v['target_mIoU'])-float(byalpha[0]['target_mIoU']),'target_mIoU_gain_retention_percent':100*(float(v['target_mIoU'])-float(byalpha[0]['target_mIoU']))/gain,'source_mIoU_loss_pp':float(byalpha[0]['source_mIoU'])-float(v['source_mIoU']),'source_loss_reduction_vs_alpha1_percent':100*(float(v['source_mIoU'])-float(byalpha[1]['source_mIoU']))/loss})
 with (R/'tradeoff.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(trade[0]));w.writeheader();w.writerows(trade)
 lines=['# Phase401：H残差强度诊断','','## 1．数据结果','','固定Phase400 H最佳权重，四个输出残差同时乘同一个α；MsRE关闭，未修改权重或输入。Catalogue32 development组／7组有有效参考云影，CloudSEN val535；零训练。按α=0、1、0.25、0.5、0.75执行，两个端点全部567张图的预测SHA、混淆与目标AP先复现，再评价中间点。以下为百分点或百分数。','','|α|目标mIoU|目标Shadow IoU|宏AP|池化AP|目标召回|目标FPR|目标mIoU收益保留%|源mIoU|源mIoU损失|','|---|---|---|---|---|---|---|---|---|---|']
 for v in trade:lines.append('|'+ '|'.join([str(v['alpha'])]+[f'{float(v[k]):.4f}' for k in ('target_mIoU','target_Shadow_IoU','target_macro_AP','target_pooled_AP','target_Shadow_recall','target_nonShadow_FPR','target_mIoU_gain_retention_percent','source_mIoU','source_mIoU_loss_pp')])+'|')
 lines+=['','收益保留比例＝(目标mIoUα−目标mIoU0)/(目标mIoU1−目标mIoU0)×100%；未裁剪超过100%或负值。源损失＝源mIoU0−源mIoUα。FPR为池化参考非云影（Surface+Cloud）被预测Shadow的比例，不是逐组平均预算。无参考云影组的AP为NA，仍参与误报统计。每点三类IoU和完整混淆见summary.csv；全部160行目标逐组、2675行源逐图保留。','','## 2．结论分析','']
 for v in trade:
  if v['alpha'] in (0,1):continue
  lines.append(f"- α={v['alpha']}：保留完整H的目标mIoU收益{v['target_mIoU_gain_retention_percent']:.2f}%，源域仍下降{v['source_mIoU_loss_pp']:.4f}点；相比α=1，源损失减少{v['source_loss_reduction_vs_alpha1_percent']:.2f}%。这说明的是该固定强度下的目标／源域取舍。")
 dominates=[v for v in trade if 0<v['alpha']<1 and float(v['target_mIoU'])>=float(byalpha[1]['target_mIoU']) and float(v['source_mIoU'])>float(byalpha[1]['source_mIoU'])]
 if dominates:lines+=['','在已测网格内，'+ '、'.join('α='+str(v['alpha']) for v in dominates)+'的目标mIoU不低于完整H，同时源mIoU更高；这是固定评价池上的描述性支配关系，不是独立确认或部署强度选择。']
 else:lines+=['','已测三个中间强度未同时达到“目标mIoU不低于完整H且源mIoU更高”。应按表中收益保留与源损失讨论取舍，不把减弱残差描述为免费恢复。']
 lines+=['','α=0恢复Source是关闭适配贡献的回退一致性；不能据此说开启适配后无遗忘。是否“保住”目标收益没有额外设置临时通过门槛，表中比例完整公开。未用源val挑最佳α后称独立验证，不追加α、阈值或训练；Phase402尚未授权。','','本轮只诊断单个H checkpoint、已暴露development与重复使用的源val535，不能归因于稳定机制或独立空间泛化。中间点AP沿用Phase400已验收的exact-ties实现；本地独立复算逐图混淆、全部汇总、宏AP算术与端点AP，未传回像元分数以独立重算中间点池化AP，保留这一验收范围限制。','','## 执行与交付','',f"完整样本前向{info['calls']}次，更新{info['updates']}次，总工作进程耗时{info['seconds']:.3f}秒（含加载、检查、CPU AP与写盘，不是纯GPU核时间）；GPU reserved峰值{info['peak_reserved_bytes']/1024**3:.3f}GiB。调用账本、输入／代码／Source／H SHA、冻结状态及端点门在RUN_INFO和INDEPENDENT_ACCEPTANCE。权重和像元数组全部留远端，未同步work_dirs。"]
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(R/'README.md').write_bytes((R/'RESULT.md').read_bytes());save('INDEPENDENT_ACCEPTANCE.json',{'passed':True,'checks':checks,'model_calls_added':0,'updates_added':0,'local_code_SHA':sha(Path(__file__)),'AP_scope':'Phase400 verified implementation reused; exact endpoint AP + independent macro aggregation; no independent pixel-level intermediate pooled AP replay','transfer_manifest_SHA':sha(R/'TRANSFER_ACCEPTED.json')});save('RUN_INFO_LOCAL.json',{'remote_RUN_INFO_SHA':sha(R/'RUN_INFO.json'),'local_protocol_SHA':sha(ROOT/'src/research_plans/phase401.md'),'endpoint_gate_SHA':sha(R/'ENDPOINT_GATE.json'),'data_membership':'Catalogue dev32, CloudSEN val535 only; no fit/test reads','target_gain_denominator_pp':gain,'source_loss_denominator_pp':loss,'target_support':target_support,'source_support':source_support,'finalizer_SHA':sha(Path(__file__)),'no_model_or_alpha_selected':True})
 dest=ROOT/'src/research_results/phase401_20261031';dest.mkdir(parents=True,exist_ok=True)
 for name in ('RESULT.md','summary.csv','tradeoff.csv','RUN_INFO.json','INDEPENDENT_ACCEPTANCE.json'):shutil.copy2(R/name,dest/name)
 log=ROOT/'src/EXPERIMENT_LOG.md';original=log.read_bytes();title='## Phase401：H残差强度诊断（2026-10-31）';assert title.encode() not in original;block=title+'\n\n固定Phase400 H最佳checkpoint a86d96f3…，四输出残差同时乘α；M关闭，不改权重。只Catalogue development32与CloudSEN val535、原512输入、batch1、FP32、4090D；α0/1先逐图复现Source/H，后0.25/0.5/0.75。\n\n数据结果（α／目标mIoU／源mIoU／目标收益保留%）：'+ '；'.join(f"{v['alpha']}/{float(v['target_mIoU']):.4f}/{float(v['source_mIoU']):.4f}/{v['target_mIoU_gain_retention_percent']:.2f}" for v in trade)+f"。实际2835前向、零更新，工作进程{info['seconds']:.3f}秒，GPU reserved峰值{info['peak_reserved_bytes']/1024**3:.3f}GiB；非纯GPU核时间。端点全567样本预测SHA/CM与目标AP精确复现；本地独立逐图混淆、汇总、宏AP算术、身份与账本验收。中间点池化AP未传像元独立重放，验收范围如实保留。\n\n结论探讨：减弱残差的目标收益保留与源损失存在明确取舍，完整五点见outputs/phase401/phase401_20261031/RESULT.md与tradeoff.csv。仅固定H单seed与已暴露池诊断，不用源val选择α后称独立确认；不追加点、不训练、不自动402。权重／数组／work_dirs未传回。\n\n";marker='## 后续维护规则'.encode();assert marker in original;nl='\r\n' if b'\r\n' in original else '\n';log.write_bytes(original.replace(marker,block.replace('\n',nl).encode()+marker,1))
 with (R/'HANDOFF.md').open('a',encoding='utf-8') as f:f.write('\n## Final delivery\nEXIT0、2835调用/0更新、端点精确复现，全部小报告已取回与本地独立数值验收。RESULT/tradeoff/summary/per_group/source_per_image/RUN_INFO及哈希交付，实验日志已追加；停止自动化，不自动402。\n')
 save('MANIFEST.json',{p.relative_to(R).as_posix():sha(p) for p in sorted(R.rglob('*')) if p.is_file() and p.name not in ('MANIFEST.json','DELIVERY_COMPLETE.json')});save('DELIVERY_COMPLETE.json',{'passed':True,'epoch':time.time(),'manifest_SHA':sha(R/'MANIFEST.json'),'calls':2835,'updates':0,'independent_acceptance_SHA':sha(R/'INDEPENDENT_ACCEPTANCE.json'),'next_phase_started':False});print(json.dumps({'passed':True,'checks':len(checks),'calls':2835,'seconds':info['seconds']}))
if __name__=='__main__':main()

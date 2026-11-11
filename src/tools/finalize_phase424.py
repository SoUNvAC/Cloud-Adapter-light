"""Local audit of diagnostic strata and final evidence interpretation."""
from pathlib import Path
import csv,json,hashlib,math,statistics
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase424/phase424_20261101'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 for n,h in js(R/'TRANSFER_MANIFEST.json').items():assert sha(R/n)==h,n
 for n,h in js(R/'COMPLETE.json')['files'].items():assert sha(R/n)==h,n
 a=js(R/'ACCEPTANCE.json');assert a['passed'] and a['calls']==64 and a['updates']==0 and a['replay_exact_scenes']==32
 assert len((R/'CALL_LEDGER.jsonl').read_text().splitlines())==64
 models={k:js(R/k/'METRICS.json') for k in ['observed','self_reference']}
 for m in models.values():
  assert len(m['per_group'])==32
  cm=[[sum(g['confusion'][i][j] for g in m['per_group']) for j in range(3)] for i in range(3)];assert cm==m['confusion']
  v=sum(100*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)-cm[i][i]) for i in range(3))/3
  assert abs(v-m['mIoU'])<1e-9
 patches=[]
 for r in csv.DictReader((R/'PATCH_DIAGNOSTICS.csv').open()):
  patches.append({k:(v if k=='product' else float(v)) for k,v in r.items()})
 assert len(patches)==32768 and len({p['product'] for p in patches})==32
 assert all(math.isfinite(v) for p in patches for k,v in p.items() if k!='product')
 tests={'all':lambda p:True,'surface_pure':lambda p:p['GT_surface']>=.9,'cloud_pure':lambda p:p['GT_cloud']>=.9,'shadow_pure':lambda p:p['GT_shadow']>=.9,'bright_flat_surface':lambda p:p['GT_surface']>=.9 and p['luminance']>=.7 and p['texture']<=.05,'new_cloud_error_surface':lambda p:p['GT_surface']>=.9 and p['new_cloud_error_fraction']>=.1,'T25XEG':lambda p:'T25XEG' in p['product']}
 for lo,hi in [(0,.1),(.1,.3),(.3,.5),(.5,1.00001)]:tests[f'confidence_{lo}_{hi}']=lambda p,lo=lo,hi=hi:lo<=p['confidence']<hi
 strata={x['stratum']:x for x in csv.DictReader((R/'STRATA_SUMMARY.csv').open())};quantiles=[]
 for name,select in tests.items():
  pp=[p for p in patches if p['valid_fraction']>=.9 and select(p)];r=strata[name];assert len(pp)==int(r['patches'])
  assert len({p['product'] for p in pp})==int(r['scenes'])
  for k in patches[0]:
   if k in ['product','x','y'] or not pp:continue
   assert math.isclose(sum(p[k] for p in pp)/len(pp),float(r[k+'_mean']),rel_tol=1e-9,abs_tol=1e-9),(name,k)
  for k in ['confidence','brighter_mass','residual_to_input','residual_to_msre','self_norm_ratio','self_cosine','bias_norm_ratio']:
   if pp:
    values=sorted(p[k] for p in pp);quantiles.append({'stratum':name,'variable':k,'n':len(pp),'median':statistics.median(values),'p10_order_stat':values[int(.1*(len(values)-1))],'p90_order_stat':values[int(.9*(len(values)-1))]})
 with (R/'ROBUST_SUMMARY.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(quantiles[0]));w.writeheader();w.writerows(quantiles)
 critical=strata['T25XEG'];on=models['observed'];selfref=models['self_reference']
 onbad=next(g for g in on['per_group'] if 'T25XEG' in g['product'])['confusion'][0][1]
 selfbad=next(g for g in selfref['per_group'] if 'T25XEG' in g['product'])['confusion'][0][1]
 lines=['# Phase424：为什么材质分支损伤高亮场景','','本轮证据更支持：现有修正头不依赖显式参考差异，而门控在缺乏明显亮度对比时仍保持较大开度。没有证据表明关键场景是因参考混入云类而出错。','', '## 1. 关键场景没有云类参考污染','',f"T25XEG所有有效标签均为地表；所选参考的地表类权重约100%，云类为0。参考比query亮5%以上的权重平均仅 {float(critical['brighter_mass_mean'])*100:.4f}%，但门控均值仍为 {float(critical['confidence_mean']):.4f}。",'', '这个门控不是经过校准的可靠概率。等亮参考的sigmoid项仍为0.37754；相似且等亮的区域缺少明确的关闭条件。地表类别一致也不能证明材质一致。','', '## 2. 清零显式参考差异几乎不改变修正','',f"原模型mIoU {on['mIoU']:.4f}%，自参考 {selfref['mIoU']:.4f}%，差 {selfref['mIoU']-on['mIoU']:+.4f}点；T25XEG地表误判云从 {onbad} 变为 {selfbad}，仅差 {onbad-selfbad} 像素。",'',f"全体patch自参考残差/原残差范数均值为 {float(strata['all']['self_norm_ratio_mean']):.6f}，方向余弦均值 {float(strata['all']['self_cosine_mean']):.9f}。该对照保留原邻域门控，不能说所有邻域信息无作用；只能说明显式参考特征与差异通道在当前权重下对输出影响很小。",'', '## 3. 无差异不等于无修正','',f"T25XEG新增残差/原MsRE同一注入层残差的范数比均值为 {float(critical['residual_to_msre_mean']):.4f}；新增残差/输入特征范数约 {float(critical['residual_to_input_mean'])*100:.2f}%。清零全部外观和差异通道、仅保留门控值时，残差范数约为原值的 {float(critical['bias_norm_ratio_mean'])*100:.2f}%。",'', '结合自参考对照，响应主要依赖保留下来的绝对外观特征，而不是单纯由偏置项产生。范数不能单独证明分类损伤，但Phase422开关实验已经显示该路径造成整体损失。','', '## 后续结构约束','', '若继续该方向，应优先验证关系驱动的内部残差：参考差异为零时修正必须为零；缺乏足够亮度支持时应能拒绝参考；对照中保持参数与训练条件匹配。当前实现不应继续仅调一个全局系数。该建议是下一步假设，本轮没有训练或验证新结构。','', '验收：64次完整输入前向、零更新；32组原预测和分数逐位复现；32次参考权重/描述符重构完全一致；标签仅作事后分析；本地重新汇总混淆矩阵和全部分层均值。另32次小张量MsRE和96次小修正头前向单独记账。']
 (R/'INTERPRETATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (R/'README.md').write_bytes((R/'RESULT.md').read_bytes())
 title='## Phase424：材质参考与残差机制诊断（2026-11-01）'
 entry=title+f"\n\n- 原模型 {on['mIoU']:.4f}%、自参考 {selfref['mIoU']:.4f}%；参考=query且差异项清零、保留原门控，输出几乎不变。\n- T25XEG参考为地表类，明显更亮参考权重仅{float(critical['brighter_mass_mean'])*100:.4f}%，门控仍{float(critical['confidence_mean']):.4f}；当前显式关系通道作用很小，绝对外观修正缺乏零差异零响应约束。\n- 64次前向零训练，32组预测/分数逐位复现，逐格统计和本地独立验收通过；不声称同材质真值或完整因果证明。\n- outputs/phase424/phase424_20261101/RESULT.md 与 INTERPRETATION.md。\n\n"
 log=ROOT/'src/EXPERIMENT_LOG.md';raw=log.read_bytes();marker='## 后续维护规则'.encode();assert raw.count(marker)==1
 if title.encode() not in raw:log.write_bytes(raw.replace(marker,entry.replace('\n','\r\n').encode()+marker))
 plan=ROOT/'src/research_plans/phase424.md';s=plan.read_text(encoding='utf-8');status='完成：64次前向零训练，原预测/分数逐位复现及本地统计复核通过。结果见 outputs/phase424/phase424_20261101/INTERPRETATION.md。'
 if status not in s:plan.write_text(s+'\n'+status+'\n',encoding='utf-8')
 (R/'DELIVERY_COMPLETE.json').write_text(json.dumps({'passed':True,'calls':64,'updates':0,'local_CM_recomputed':True,'strata_recomputed':len(tests),'patches':len(patches)},indent=2))
 manifest={p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file() and p.name!='LOCAL_MANIFEST.json'}
 (R/'LOCAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('sealed',len(manifest),'files')
if __name__=='__main__':main()

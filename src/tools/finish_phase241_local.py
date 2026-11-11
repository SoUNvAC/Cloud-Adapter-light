"""Independently check small received tables; add evidence-based local interpretation."""
from pathlib import Path
import json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase241/phase241_20261030';D=R/'delivery';S=ROOT/'src/research_results/phase241_20261030'
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def main():
 start=time.monotonic();assert js(R/'TRANSFER_ACCEPTED.json')['passed'];assert not (R/'LOCAL_EDITORIAL_RECEIPT.json').exists();checks=[]
 for n,h in js(D/'SHA256SUMS.json').items():assert sha(D/n)==h,n
 m=js(D/'METRICS.json');records=m['per_scene'];rank=m['ranking'];pooled=m['pooled'];boot=js(D/'BOOTSTRAP.json');indices=js(D/'BOOTSTRAP_INDICES.json')['domains']
 for p in pooled:
  rows=[v for v in records if all(v[k]==p[k] for k in ('domain','split','model','state','decision')) and (v['main'] if p['support']=='main' else (not v['main'] if p['support']=='nominal_no10' else True))]
  c=np.array([v['confusion'] for v in rows],np.int64).sum(0);assert c.tolist()==p['confusion'];d=np.diag(c);u=c.sum(0)+c.sum(1)-d;assert abs(np.mean(d[u>0]/u[u>0])-p['mIoU_union'])<1e-13;checks.append('pooled confusion and mIoU')
 for domain,ix in indices.items():
  scenes=ix['scenes'];assert len(scenes)==(6 if domain=='L8' else 10);idx=np.array(ix['indices']);dist={}
  for model in ('Source','C21701','C21702'):
   for state in ('raw','calibrated'):
    for decision in ('argmax','frozen_threshold'):
     lookup={v['scene']:v for v in records if v['domain']==domain and v['split']=='test' and v['model']==model and v['state']==state and v['decision']==decision};c=np.array([lookup[s]['confusion'] for s in scenes],np.float64)[idx].sum(1);r=c.sum(2);p=c.sum(1);d=np.diagonal(c,axis1=1,axis2=2);u=r+p-d
     dist[model,state,decision]={'mIoU':np.mean(np.divide(d,u,out=np.full_like(d,np.nan),where=u>0),1),'Shadow_recall':np.divide(d[:,2],r[:,2],out=np.full(10000,np.nan),where=r[:,2]>0),'FPR':np.divide(c[:,0,2]+c[:,1,2],r[:,0]+r[:,1],out=np.full(10000,np.nan),where=r[:,0]+r[:,1]>0)}
    lookup={v['scene']:v['AUC'] for v in rank if v['domain']==domain and v['split']=='test' and v['model']==model and v['state']==state and v['task']=='Shadow_Surface' and v['scope']=='scene'};a=np.array([np.nan if lookup[s] is None else lookup[s] for s in scenes])[idx];n=np.isfinite(a).sum(1);dist[model,state,None]={'Shadow_Surface_AUC':np.divide(np.nansum(a,1),n,out=np.full(10000,np.nan),where=n>0)}
  def delta(seed,metric,decision,contrast):
   A=dist['Source','raw',decision][metric];B=dist['Source','calibrated',decision][metric];C=dist['C'+str(seed),'raw',decision][metric];F=dist['C'+str(seed),'calibrated',decision][metric]
   return {'B-A':B-A,'D-C':F-C,'C-A':C-A,'D-B':F-B,'(B-A)-(D-C)':B-A-F+C}[contrast]
  for v in boot:
   if v['domain']!=domain:continue
   a=(delta(21701,v['metric'],v['decision'],v['contrast'])+delta(21702,v['metric'],v['decision'],v['contrast']))/2 if v['seed']=='two_seed_difference_mean' else delta(v['seed'],v['metric'],v['decision'],v['contrast']);good=np.isfinite(a);assert int((~good).sum())==v['NA_replicates'];assert np.max(abs(np.quantile(a[good],[.025,.975])-v['CI95']))<1e-12;checks.append('same-index independent local bootstrap '+str((domain,v['metric'],v['contrast'],v['seed'],v['decision'])))
 for t in js(D/'THRESHOLD_LOCK.json')['thresholds']:assert t['success'] and t['mean_val_FPR']<=.01 and t['inclusive_boundary_FPR']>.01
 assert all(v['success'] for v in js(D/'CALIBRATION_LOCK.json')['fits']);cost=js(D/'COST.json');assert cost['calls']==9456 and cost['training']==0 and cost['downloads']==0
 text=(D/'RESULT.md').read_text(encoding='utf-8');before=text.split('## 2．结论分析')[0]
 conclusion='''## 2．结论分析

**排序改变：目标之间不同，景均与池化也不能混说。** L8主比较的两seed均差为+2.335 AUC百分点，条件95%区间[−2.133,+6.969]，不能确定其景均Shadow/Surface排序收益。与此同时，L8池化Shadow/Surface AUC由Source的82.455%降至73.242%／75.656%，池化Shadow OvR AP也由34.054%降至31.858%／32.101%；景均点估计改善并不代表全池排序改善。SPARCS主比较均差+6.687百分点，区间[+0.024,+18.954]，方向更有利，但下界仅略高于零；seed21702单独区间仍跨零，十景条件区间宽，不能称跨seed稳定的独立确认。

**指定对齐兑现：Source可恢复部分决策收益，不能替代全部适配。** L8 Source偏置对齐将池化mIoU从48.570%提高到49.808%（+1.238百分点），而原C为50.607%／50.990%；两方分别按固定程序对齐后，C−Source仍为+0.766／+1.263百分点，两seed均差约+1.015，条件区间[−2.481,+4.960]。因此L8原始mIoU增益有一部分可由这个指定偏置程序兑现，但剩余收益仍不确定。SPARCS Source对齐仅+0.908百分点；适配前后差从原始+13.148／+12.682收窄为对齐后+11.775／+12.063，两seed均差约+11.919，区间[+7.746,+16.895]。这不支持“SPARCS收益只是类别偏置”的解释。D−B是指定对齐后的剩余差异，不是特征贡献或真正上界；对齐优化的是val NLL，C的test NLL/mIoU并非必然改善。

**实际工作点有代价，raw mIoU结论需要限定。** 十二个阈值均满足val景均名义FPR≤1%，却未保证test满足：raw C在L8两seed的test景均Shadow FPR为1.196%／1.243%，SPARCS为1.037%／1.054%。对应Shadow召回从原argmax的21.640%／20.924%降至17.718%／17.856%（L8），从63.588%／61.120%降至43.669%／44.189%（SPARCS）。在原始argmax上，L8 C相对Source同时降低Cloud召回与Surface→Cloud误报，属于不同类别取舍，不能仅凭mIoU称全面改善。Source、C的排序质量、指定决策、参考标签支持应分别描述。

所有区间使用同一批场景的配对重采样，条件于已选checkpoint、val偏置与阈值；不覆盖训练随机性，不能升级为独立确认。L8 no十景标注完整性未知，仅保留名义参考表；不据此宣称真实云影误报，不据反馈修改像元标签。本轮不追加方法或进入下一phase。

## 执行与交付

9450次主体前向及6次固定重复，共9456次；零网络训练、零下载。GPU阶段449.739秒，远端CPU处理11852.607秒（约3.29小时），采样到的CPU峰值约1.21 GiB；编译/微型自检RSS未采样。远端新增约28.51 GiB，其中完整FP32分数保留远端，本地正式回执约23.9 MiB。远端独立检查56091项通过，本地另以既有索引复算全部主要配对区间和池化计数；没有新增抽样或模型调用。

delivery/RESULT.md与SHA256SUMS.json保留远端原始封存版本；本文件仅补充基于已验收数据的解释，修改回执见LOCAL_EDITORIAL_RECEIPT.json，本地额外文件SHA见LOCAL_DELIVERY_MANIFEST.json。
'''
 # Keep the requested three data tables; fix Markdown separator cell counts only.
 lines=(before+conclusion).splitlines()
 for i,line in enumerate(lines):
  if line.startswith('|---') and i and lines[i-1].startswith('|'):lines[i]='|'+'|'.join(['---']*(lines[i-1].count('|')-1))+'|'
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');S.mkdir(parents=True,exist_ok=False)
 summary='''# Phase241：数据结果与结论

## 1．数据结果

- 主要支持：L8 yes6景、SPARCS发布参考10景，Source及固定C两seed。9456次前向、零网络训练；远端56091项CPU检查及本地计数/区间复算通过。
- 主排序终点C−Source景均Shadow/Surface AUC，两seed均差：L8 +2.335百分点，95%条件区间[−2.133,+6.969]；SPARCS +6.687，[+0.024,+18.954]。SPARCS seed21702单独区间跨零。
- Source指定偏置对齐池化mIoU增量：L8 +1.238、SPARCS +0.908百分点。两方对齐后的C−Source均差：L8约+1.015，[−2.481,+4.960]；SPARCS约+11.919，[+7.746,+16.895]。
- L8 raw C池化Shadow/Surface AUC为73.242%／75.656%，低于Source 82.455%；不能用景均点估计掩盖池化下降。
- val 1%名义FPR阈值冻结后，raw C test景均FPR：L8 1.196%／1.243%，SPARCS 1.037%／1.054%；召回有所下降，预算不保证逐景或跨池满足。

## 2．结论分析

L8尚不能确定景均排序和指定对齐后剩余收益，原mIoU增益需限定为类别决策取舍，部分可由Source偏置对齐兑现。SPARCS排序方向和对齐后mIoU剩余差异更有利，不能用纯偏置解释，但只有十景、区间宽且两个seed不是额外独立场景，仍为探索诊断。L8 no十景仅名义参考，不作可靠无影负样本；不进入独立确认或新增方法。

完整三表、逐景、阈值与校准锁、账本、SHA和验收在outputs/phase241/phase241_20261030/；完整FP32分数留授权shared，不传work_dirs或权重。
'''
 (S/'RESULT.md').write_text(summary,encoding='utf-8');receipt={'passed':True,'remote_RESULT_SHA':sha(D/'RESULT.md'),'local_RESULT_SHA':sha(R/'RESULT.md'),'remote_original_manifest_unchanged':True,'local_checks':len(checks),'checks':checks,'new_neural_calls':0,'new_fits':0,'new_bootstrap_draws':0,'bootstrap_existing_index_recomputations':len(boot),'local_seconds':time.monotonic()-start,'editorial_change':'numerical interpretation and Markdown separators; no metric arrays/counts/locks modified'};save(R/'LOCAL_EDITORIAL_RECEIPT.json',receipt)
 files=[R/'RESULT.md',R/'LOCAL_EDITORIAL_RECEIPT.json',S/'RESULT.md'];save(R/'LOCAL_DELIVERY_MANIFEST.json',{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files});print(json.dumps({k:receipt[k] for k in ['passed','local_checks','bootstrap_existing_index_recomputations','local_seconds','local_RESULT_SHA']}))
if __name__=='__main__':main()

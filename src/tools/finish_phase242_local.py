"""Verify received Phase242 arithmetic and add local evidence-based interpretation."""
from pathlib import Path
import json,hashlib,math,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase242/phase242_20261031';D=R/'delivery';S=ROOT/'src/research_results/phase242_20261031'
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def main():
 start=time.monotonic();assert js(R/'TRANSFER_ACCEPTED.json')['passed'];assert js(D/'INDEPENDENT_ACCEPTANCE.json')['passed'];assert not (R/'LOCAL_EDITORIAL_RECEIPT.json').exists();checks=[]
 def ok(name,value):assert bool(value),name;checks.append(name)
 for n,h in js(D/'SHA256SUMS.json').items():ok('formal SHA '+n,sha(D/n)==h)
 for split in ('val','test'):
  for domain in ('L8','SPARCS'):
   r=js(D/(split.upper()+'_'+domain+'_RESULTS.json'));source=r['matrices']['Source:raw'];scenes=r['scenes'];ok('fixed scene count',len(scenes)==(8 if domain=='L8' and split=='val' else 6 if domain=='L8' else 10));P={v['positive_scene']:v['P'] for v in source['cells']};N={v['negative_scene']:v['N'] for v in source['cells']};den=float(sum(P.values()))*sum(N.values())
   for name,mat in r['matrices'].items():
    cells=mat['cells'];s=mat['summary'];value=math.fsum((v['AUC'] or 0.)*P[v['positive_scene']]*N[v['negative_scene']] for v in cells)/den;ok('pooled matrix '+name,abs(value-s['pooled_AUC'])<1e-10)
    vals=[v['AUC'] for v in cells if v['positive_scene']==v['negative_scene'] and v['AUC'] is not None];ok('macro matrix '+name,abs(sum(vals)/len(vals)-s['scene_mean_AUC'])<1e-10)
   for model in ('C21701','C21702'):
    for scene in scenes:
     rec=js(D/'scene_receipts'/split/domain/(scene+'.json'));c=rec['C'][model];T=np.asarray(c['transition_reference_Source_C']);raw=next(v for v in r['decisions'] if v['model']==model and v['scene']==scene and v['state']=='raw' and v['decision']=='argmax');ok('matched transition marginals',T.sum(2).tolist()==rec['Source']['decision']['argmax']['confusion'] and T.sum(1).tolist()==raw['confusion'])
     for k,change in enumerate(c['changes']):ok('same pixel correct-wrong',change['Source_correct_C_wrong']==int(T[k,k,:].sum()-T[k,k,k]) and change['Source_wrong_C_correct']==int(T[k,:,k].sum()-T[k,k,k]))
   for a,b,c in zip(r['comparisons']['C21701'],r['comparisons']['C21702'],r['comparisons']['two_seed_mean']):
    if c['delta_AUC'] is not None:ok('two seed weighted mean',abs(c['delta_AUC']-(a['delta_AUC']+b['delta_AUC'])/2)<1e-14 and abs(c['weighted_delta']-(a['weighted_delta']+b['weighted_delta'])/2)<1e-14)
   for v in r['decompositions']:ok('symmetric decomposition',abs(v['Shadow_component']+v['Surface_component']-(v['CC']-v['SS']))<1e-12)
   for model,state,dec in {(v['model'],v['state'],v['decision']) for v in r['decisions']}:
    rows=[v for v in r['decisions'] if (v['model'],v['state'],v['decision'])==(model,state,dec)];cm=np.sum([v['confusion'] for v in rows],axis=0);ref=cm.sum(1);pred=cm.sum(0);di=np.diag(cm);union=ref+pred-di;v=next(v for v in js(D/'FINAL_DATA.json')['decisions'] if (v['split'],v['domain'],v['model'],v['state'],v['decision'])==(split,domain,model,state,dec));ok('pooled decision arithmetic',cm.tolist()==v['confusion'] and abs(np.mean(di/union)-v['mIoU'])<1e-12 and abs((cm[0,2]+cm[1,2])/(ref[0]+ref[1])-v['nonShadow_to_Shadow_FPR'])<1e-12)
 data=js(D/'FINAL_DATA.json');l8=js(D/'TEST_L8_RESULTS.json');neg=sum(-v['weighted_delta'] for v in l8['top_pairs'] if v['kind']=='damage');totalneg=np.mean([-v['damage_contribution'] for v in data['ranking'] if v['split']=='test' and v['domain']=='L8' and v['model'].startswith('C') and v['state']=='raw']);fraction=float(neg/totalneg)
 before=(D/'RESULT.md').read_text(encoding='utf-8').split('## 2．结论分析')[0]
 text=f'''## 2．结论分析

**损伤集中，但不能把删景后的数字替代主分析。** L8 raw C相对Source池化AUC下降9.213／6.799个百分点，景均却提高2.350／2.321。两seed平均加权损伤前三对都以LC81321192014054LGN00为Shadow来源，分别对应LC81620582014104LGN00、LC81130632014241LGN00、LC82290572014141LGN00的Surface；三对合计贡献−{100*neg:.3f} AUC百分点，占全部负贡献绝对值约{100*fraction:.1f}%（未扣除其他对收益）。该景作为Shadow来源的净行贡献为−9.751／−7.024百分点。去掉其整行整列后，池化差变为+0.293／−0.242；这说明描述性结果对该景高度敏感，不能据此删去难景或断言该景造成了全部损伤。SPARCS总体池化收益+6.844／+6.637，局部前三损伤也集中于同一Shadow来源LC81540102013205LGN00，但规模较小，并不能外推普遍的跨景受损。

**存在整景偏移成分，同时有明显类相关变化，不能称纯加性偏置。** L8关键景LC81321192014054LGN00整景delta中位数为−2.377／−2.107；其Shadow中位数−2.738／−2.569、Surface中位数−1.888／−1.681。全有效域q10到q90仍达[−3.590,−0.567]／[−3.191,−0.687]，像元变化并不近似一个完全相同的常数。SPARCS LC81020642014004LGN00的Shadow中位变化+7.712／+7.953，而Surface为−1.319／−0.810，更明确地包含类相关分化。这些真类条件只有事后标签才知道。第一条L8损伤对的混合评分分解中，Shadow侧为−36.158／−28.199 AUC百分点，Surface侧为+8.978／+13.072；仅是指定跨模型评分混合下的算术分量，不是特征更新的因果归因，也没有影像依据把变化归因于雪冰或暗地物。

**固定信号在L8跨val/test同向，跨目标／seed仍不稳。** L8 val Spearman为0.675／0.815，test为0.889／0.858；val方向一致76.786%／73.214%，test两seed均73.333%，留一景相关性仍为正。SPARCS seed21701却只有val rho=0.208、方向一致48.750%，test rho=0.256，留一景范围均可跨到负值；seed21702为val 0.495、test 0.673。因而不能宣布这是可靠的跨目标检测器，尤其seed21701的val方向证据不足。SPARCS val可评有向跨景对实际81，test90；L8为56／30。这些对共享场景，不是独立样本，不做p值。信号依赖更新后的C分数，并继续使用发布的有效域，不能称适配前预测或完全无标签部署检测。

**排序可被固定平移改变，实际收益却未稳定兑现。** L8 J1将C21701池化AUC从73.242%升至82.160%，C21702从75.656%升至80.594%，接近但仍低于Source 82.455%；景内排序不变是构造性质。然而原argmax mIoU分别50.607%→51.556%、50.990%→50.380%，并非两seed均改善。沿用raw冻结阈值时，景均Shadow FPR从1.196%／1.243%升至2.822%／2.982%，最差景为6.130%／9.968%，不能把排序恢复当作可用工作点已恢复。SPARCS J1反而把池化AUC从96.640%／96.434%降到96.164%／95.704%，argmax mIoU从71.567%／71.101%降到70.552%／70.201%。J0保持排序不变仍可改变召回与误报，也不能预设保住1%预算。本轮没有理由据此挑J1为已验证算法，也未新增干预的置信区间。

这是已暴露数据的解释审计，原主支持和全部不利结果保留。可以判断L8存在集中且与场景间评分相对位置相关的损伤，也可以判断变化含类相关成分；尚不能确定可部署的检测与恢复机制。独立场景或第二模型需要后续另行授权，本轮不自动继续。

## 交付与成本

网络调用、更新、下载、新阈值及新bootstrap均为0。累计CPU阶段1674.585秒（约27.91分钟），采样峰值RSS约3.48 GiB；小工程阶段峰值未采样，不声称已知绝对峰值。新增工作约5.98 GiB，正式本地交付约9.62 MiB，原28 GiB FP32分数没有复制或传回。远端5679项独立验收通过，本地另核矩阵、同像元转移、分解和决策汇总；检查数不是科研证据。

delivery/RESULT.md及原SHA256SUMS保持远端封存文本；本文件仅补充已验收数据的解释，见LOCAL_EDITORIAL_RECEIPT.json和LOCAL_DELIVERY_MANIFEST.json。
'''
 lines=(before+text).splitlines()
 for i,line in enumerate(lines):
  if line.startswith('|---') and i and lines[i-1].startswith('|'):lines[i]='|'+'|'.join(['---']*(lines[i-1].count('|')-1))+'|'
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');S.mkdir(parents=True,exist_ok=False)
 summary='''# Phase242：数据结果与结论

## 1．数据结果

- 零网络调用、更新、下载、新阈值和bootstrap；CPU约27.91分钟，远端5679项独立验收通过，小结果已SHA同步本地。
- L8 raw C池化AUC相对Source−9.213／−6.799百分点；前三跨景损伤均以LC81321192014054LGN00为Shadow来源。去掉其整行列后池化差+0.293／−0.242，仅作敏感性诊断，不替换原支持。
- 固定T信号：L8 val rho 0.675／0.815，test 0.889／0.858；SPARCS val 0.208／0.495、test 0.256／0.673，seed21701留一可转负。
- L8 J1池化AUC 73.242%／75.656%→82.160%／80.594%，Source 82.455%；冻结阈值景均Shadow FPR却从1.196%／1.243%升至2.822%／2.982%。
- SPARCS J1池化AUC 96.640%／96.434%→96.164%／95.704%，argmax mIoU 71.567%／71.101%→70.552%／70.201%。

## 2．结论分析

L8损伤集中且涉及跨场景相对评分；整景共同偏移只能解释一部分，同像元更新含明显类相关变化，不能称纯加性偏置。固定T在L8同向，跨目标和seed尚不稳定。移除中位变化可改善L8池化排序，却未稳定兑现决策收益并增加误报，SPARCS则出现下降。因此J1只是诊断，不是已验证恢复算法。保留全部场景和不利结果，不自动追加独立场景、第二模型或后续phase。

完整三表/矩阵/27格转移/精确分位数/固定六对分解/留一敏感性和SHA回执：outputs/phase242/phase242_20261031/。
'''
 (S/'RESULT.md').write_text(summary,encoding='utf-8');receipt={'passed':True,'checks':len(checks),'details':checks,'local_seconds':time.monotonic()-start,'remote_RESULT_SHA':sha(D/'RESULT.md'),'local_RESULT_SHA':sha(R/'RESULT.md'),'remote_original_manifest_unchanged':True,'new_neural_calls':0,'new_bootstrap':0,'new_thresholds':0,'L8_top3_negative_contribution_pp':100*neg,'L8_top3_share_of_gross_negative':fraction};save(R/'LOCAL_EDITORIAL_RECEIPT.json',receipt);files=[R/'RESULT.md',R/'LOCAL_EDITORIAL_RECEIPT.json',S/'RESULT.md'];save(R/'LOCAL_DELIVERY_MANIFEST.json',{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files});print(json.dumps({'passed':True,'checks':len(checks),'top3_share':fraction,'local_seconds':receipt['local_seconds']}))
if __name__=='__main__':main()

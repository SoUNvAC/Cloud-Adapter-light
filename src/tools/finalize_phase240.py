"""Three concise result tables, arm trade-offs, and immutable local delivery hashes."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase240/phase240_20261030'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def fmt(x):return 'NA' if x is None else f'{x:.3f}' if isinstance(x,(int,float)) else str(x)
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(x) for x in r)+' |\n' for r in rows)
def main():
 assert not (R/'DELIVERY_COMPLETE.json').exists();assert js(R/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed'
 t1=js(R/'TABLE1.json');t2=js(R/'TABLE2.json');t3=js(R/'TABLE3_ALL_SCENES.json');cost=js(R/'COST.json');diag=js(R/'DIAGNOSTICS.json')
 text='# Phase240：三个固定分枝的一轮探索\n\n## 1 数据结果\n\n单seed21701，原L8 train18／val8／test16，SPARCS train60／val10／test10。历史test已暴露，全部为探索性描述；没有新阈值、插值、路由或Source数据。Source／C复用236，C按固定LR单seed五候选重算；没有用两LR九候选结果代替。\n\n### 表1 主结果（单位%或百分点）\n\n'
 rows=[]
 for v in t1:
  if v['status']!='available':rows.append([v['domain']+'/'+v['role'],'NA','NA','NA','NA','NA','NA','NA']);continue
  rows.append([v['domain']+'/'+v['role'],v['trainable_parameters'],str(v['val_step'])+'/'+fmt(v['val_mIoU']),fmt(v['mIoU'])+'/'+fmt(v['scene_mean_mIoU']),fmt(v['delta_C_pp'])+'/'+fmt(v['delta_Source_pp']),fmt(v['Cloud_recall'])+'/'+fmt(v['Surface_to_Cloud_FPR']),fmt(v['Shadow_recall'])+'/'+fmt(v['nonShadow_to_Shadow_FPR']),','.join(fmt(x) for x in v['class_IoU'])])
 text+=table(['目标/臂','可训练参数','val步/mIoU','test池化/景均','相对C/Source','Cloud召回/Surface误报','Shadow召回/非影误报','三类IoU S/C/Sh'],rows)
 text+='\n### 表2 实现核对\n\n'+table(['目标/臂','冻结检查次数','原236输入增强顺序','训练CE/BCE/Dice','匹配权重','梯度观察'],[[v['domain']+'/'+v['branch'],v['frozen_checks'],v['schedule_exact_original236'],str(v['training_weights']),str([x['weight'] for x in v['matching']]),str(v.get('gradient_branches','A看DIAGNOSTICS；B各辅助Dice均为0'))] for v in t2])
 text+='\n固定train诊断（不是test泛化指标）：\n\n'+table(['目标/状态','固定样本数','冲突数','平均cos'],[[z['domain']+'/'+z['state'],z['samples'],z['conflicts'],fmt(z['cos_mean'])] for z in js(R/'GRADIENT_DIAGNOSTIC_SUMMARY.json')])
 text+='\nA固定train样本的mask保持／类别分数变化：\n\n'+table(['目标','样本数','mask精确相同','类别分数改变'],[[z['domain'],z['samples'],z['mask_exact'],z['class_changed']] for z in js(R/'A_MASK_INVARIANCE.json')])
 text+='\n训练G每步cos、范数、冲突／零范数记录见交付LOSSES。内部损失采样随机量不声称逐值配对。\n\n### 表3 全部L8 test景\n\n'
 text+=table(['scene / Shadows','全Cloud参考','Source','C','A','B','G','A−C / B−C / G−C'],[[v['scene']+' / '+v['usgs_shadows'],v['reference_all_Cloud'],fmt(v['Source_mIoU']),fmt(v['C_mIoU']),fmt(v.get('A_mIoU')),fmt(v.get('B_mIoU')),fmt(v.get('G_mIoU')),' / '.join(fmt(v.get(a+'_delta_C_pp')) for a in ['A','B','G'])] for v in t3 if v['domain']=='L8'])
 text+='\n完整逐景三类混淆、六种错误流变化及SPARCS全部景见CSV。L8 yes六景/no十景分层见L8_TEST_STRATA；no不代表真实无影，不改变完整16景主表。缺类召回保留NA。\n\n'
 if cost.get('TF32_disabled_for_recovered_G'):
  text+='G原首批门检均在零更新时停止，额外FP32诊断关闭TF32后以原atol1e-5/rtol1e-4通过，才恢复两条G轨迹。原A/B训练和评价未重跑；G训练后端明确禁TF32，不能把它藏作与C完全相同的数值后端。A/B原选择在原test前锁定，后续G选择在新增Gtest前锁定；不能追溯声称六臂都在最早A/Btest前统一封存。全部失败、探测与额外调用计入总成本。\n\n'
 text+=f"实际成功更新{cost['successful_updates']}；完整输入前向{cost['calls']}；累计GPU阶段{cost['GPU_stage_seconds']:.3f}秒。反传/更新（含门检与失败）见COST.json。C和Source是历史预算，不计作本次新拟合；没有Source+B新校准。\n\n## 2 结论探讨\n\n"
 for arm in ['A','B','G']:
  vals=[v for v in t1 if v['role']==arm and v['status']=='available'];effect='；'.join(v['domain']+'相对C '+fmt(v['delta_C_pp'])+'pp、相对Source '+fmt(v['delta_Source_pp'])+'pp' for v in vals) or '本轮缺少可评价模型'
  boundary={'A':'query mask保持不等于最终语义预测不变；类别读出的结果仍依赖现有query。','B':'只去掉训练Dice梯度；匹配仍含Dice、总梯度尺度也改变，不证明比例偏置原因。','G':'训练负余弦与投影更新不等于泛化因果；需把收益与类别代价共同看。'}[arm]
  text+=f'{arm}：{effect}。{boundary}这是单seed探索，不声明跨seed稳定，不按test更换所选步。\n\n'
 text+='约0.5pp只作复查线索，不是通过门；全部不利场景、末步与选定点都保留。未自动追加第二seed或其他phase。\n'
 (R/'RESULT.md').write_text(text,encoding='utf-8')
 S=ROOT/'src/research_results/phase240_20261030';assert not S.exists();S.mkdir(parents=True)
 names=['RESULT.md','TABLE1.csv','TABLE2.csv','TABLE3_ALL_SCENES.csv','INDEPENDENT_ACCEPTANCE.json','COST.json']
 for n in names:shutil.copy2(R/n,S/n)
 manifest={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.is_file() and p.name not in ['DELIVERY_MANIFEST.json','DELIVERY_COMPLETE.json']}
 (R/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 (R/'DELIVERY_COMPLETE.json').write_text(json.dumps({'status':'complete','files':len(manifest),'summary':str(S),'no_next_phase_started':True},indent=2),encoding='utf-8')
 print('Phase240 delivery complete')
if __name__=='__main__':main()

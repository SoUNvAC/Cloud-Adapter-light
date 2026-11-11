"""Compose the fixed diagnostic tables; no remote access or metric refitting."""
from pathlib import Path
import json,csv,hashlib
R=Path(__file__).resolve().parents[2]/'outputs/phase247/phase247_20261031'
def js(n):return json.loads((R/n).read_text())
def write_csv(name,rows):
 with (R/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 tables=js('RESULT_TABLES.json');ind=js('INDEPENDENT_VALIDATION.json');assert ind['passed'] and js('RECOVERY_COMPLETE.json')['passed'];done=js('INFERENCE_COMPLETE.json');support=js('DATA_SUPPORT_ACCEPTANCE.json')
 models=[r['model'] for r in tables['summary']];lookup={r['model']:r for r in tables['summary']};base=lookup['Source'];summary=[]
 for row in tables['summary']:
  v={'model':row['model']}
  for k in ['scene_mean_AUC','within_weighted_AUC','cross_AUC','pooled_AUC']:v[k+'_percent']=row[k]*100;v[k+'_delta_pp']=(row[k]-base[k])*100
  summary.append(v)
 write_csv('SUMMARY.csv',summary)
 matrix=[{'model':m,**v} for m,rows in tables['matrices'].items() for v in rows];write_csv('SCENE_PAIR_AUC.csv',matrix)
 counts={r['scene']:r['counts'] for r in support['scenes']};scenes=list(counts);cm=js('CONFUSIONS.json');decisions=[]
 def decision(c):
  ious=[c[i][i]/(sum(c[i])+sum(row[i] for row in c)-c[i][i]) for i in range(3)]
  return {'fixed_three_class_mIoU_percent':sum(ious)/3*100,'Cloud_recall_percent':c[1][1]/sum(c[1])*100,'Shadow_recall_percent':c[2][2]/sum(c[2])*100,'nonShadow_FPR_percent':(c[0][2]+c[1][2])/sum(sum(c[i]) for i in [0,1])*100}
 for m in models:
  pooled=[[0]*3 for _ in range(3)]
  for scene in scenes:
   c=cm[scene+'/'+m];decisions.append({'scope':scene,'model':m,**decision(c)})
   for i in range(3):
    for j in range(3):pooled[i][j]+=c[i][j]
  decisions.append({'scope':'pooled','model':m,**decision(pooled)})
 write_csv('RAW_DECISIONS.csv',decisions)
 sample=[]
 for r in tables['same_pixel_sample']:
  v={k:r[k] for k in ['scene','model','sample_n','sample_p','sample_pair_AUC','sample_pshadow_AUC','sample_pshadow_AP']}
  for cls in ['Surface','Shadow']:
   for q,x in zip(['q25','median','q75'],r[cls+'_delta_q25_q50_q75']):v[cls+'_delta_'+q]=x
  sample.append(v)
 write_csv('SAME_PIXEL_SAMPLE.csv',sample)
 lines=['# Phase247：L7固定三景冻结头外推诊断','','## 1．数据结果','','复用Source与Phase236锁定的两个L8普通适配头C21701/C21702。本轮没有用L7标签训练、拟合、校准或选模型。三景是Phase245预先确定的代表，不按结果选择。共688个512原生窗口，完整保留边缘补零窗口。','','主评价只在2015同归档RGB/标签的明确Shadow（64）与Surface（128）支持上；分数为log(max(pShadow,1e-12))−log(max(pSurface,1e-12))。下表AUC为百分数，均使用完整有效像元，非抽样。','','|模型|景均AUC|景内按对数加权AUC|跨景AUC|池化AUC|','|---|---:|---:|---:|---:|']
 for r in tables['summary']:lines.append('|'+r['model']+'|'+ '|'.join(f"{r[k]*100:.3f}" for k in ['scene_mean_AUC','within_weighted_AUC','cross_AUC','pooled_AUC'])+'|')
 lines+=['','相对Source，两个seed分别：']
 for m in models[1:]:
  r=lookup[m];lines.append(f"- {m}：景均 {(r['scene_mean_AUC']-base['scene_mean_AUC'])*100:+.3f}pp；跨景 {(r['cross_AUC']-base['cross_AUC'])*100:+.3f}pp；池化 {(r['pooled_AUC']-base['pooled_AUC'])*100:+.3f}pp。")
 lines+=['','|场景|有效Shadow / Surface|Source AUC|C21701 AUC|C21702 AUC|','|---|---:|---:|---:|---:|']
 for scene in scenes:
  aucs=[next(v['AUC'] for v in tables['matrices'][m] if v['positive_scene']==v['negative_scene']==scene)*100 for m in models]
  lines.append(f"|{scene}|{counts[scene]['2']:,} / {counts[scene]['0']:,}|"+'|'.join(f'{a:.3f}' for a in aucs)+'|')
 lines+=['','**同像元规则样本的pShadow AP**（每块展平索引0、256、512…，同位置配对；不是全像元AP）：','','|模型|三景等权平均AP|','|---|---:|']
 for m in models:
  rows=[r for r in tables['same_pixel_sample'] if r['model']==m];lines.append(f"|{m}|{sum(r['sample_pshadow_AP'] for r in rows)/3*100:.3f}%|")
 lines+=['','样本中三景的pShadow AP对两个seed均上升。AP增益不能由对同一pShadow分数施加严格单调的阈值/偏置变换产生；因此本批变化包含排序改善。但这不能区分特征改善与非线性读出改变，也不能据此证明某个模块机制。规则网格不是随机样本，其样本AP不能冒充全景AP。','','补充实际argmax决策（原0/191及RGB无效像元忽略，三类固定平均；非Shadow为明确Surface+Cloud）：','','|模型|限定支持池化mIoU|云影召回|非Shadow→Shadow误报|','|---|---:|---:|---:|']
 for r in decisions:
  if r['scope']=='pooled':lines.append(f"|{r['model']}|{r['fixed_three_class_mIoU_percent']:.3f}%|{r['Shadow_recall_percent']:.3f}%|{r['nonShadow_FPR_percent']:.3f}%|")
 lines+=['','这些误报率是实际原argmax结果，未在L7上选择阈值，没有1%预算保证；191未覆盖，不能与完整薄云支持上的mIoU直接比较。','','## 2．结论分析','','**本批未复现“景均变好、跨景变差”的分歧。** 两个L8适配头在三景L7上全部景内AUC上升，景内加权、跨景及池化AUC也均上升。因而不能把Phase241/242的L8现象写成监督头更新的普遍后果。','','**这是有信息的外推边界，但还没确定条件。** 既有L8发现保留；SPARCS历史收益与本轮L7收益共同说明适配后并不必然跨景失配。数据集、场景组成、输入处理、标签口径与适配训练域仍相互混杂，不能仅由此归因于传感器。尤其本轮C在L8上训练，L7只有评价，所以它不是“L7监督适配造成什么变化”的实验。','','同像元评分delta的类条件四分位数见SAME_PIXEL_SAMPLE.csv。三景Surface和Shadow的变化并非相同常数，而且中位数差不能解释全部排序变化，不据此拟合新干预或事后扩大候选信号。','','只有3个相关组，不做显著性宣称或用海量像元替代场景数。没有删除难景、补景、根据结果挑选seed或修改阈值。现有数据不足以决定何种场景会失配，也没有验证安全恢复算法。','','## 3．验收、成本与口径保留','',f"- 实际{done['calls']}次网络调用，其中2064次固定窗口推理、3次重复；三个重复最大差均0。非头参数与缓冲哈希不变。",f"- 推理流水线墙钟{done['seconds']:.2f}秒，包含CPU输入验收/IO/模型装载等，不等于纯GPU计算耗时；进程采样峰值RSS {done['max_RSS_KiB']/1024**2:.3f}GiB，PyTorch峰值reserved {done['GPU_peak_reserved']/1024**3:.3f}GiB，不代表显卡总占用。",f"- CPU主分析{tables['analysis_seconds']:.2f}秒；独立核验{ind['seconds']:.2f}秒。27个全像元AUC均落在独立8192分箱算法的上下界内，最大包络宽度{ind['max_enclosure_width']*100:.5f}pp；这是一致性界核验，不冒称独立精确重算。",'- 原分析缺少sklearn而中止；已有分数全部保留。使用NumPy秩和AUC/并列分数AP恢复CPU分析，手算例与30组随机并列分数用直接两两比较/阈值枚举核对通过。新增网络调用0，未安装依赖。原FAILURE.json保留，RECOVERY_COMPLETE.json记录恢复成功。','- 原始2015包的191语义未确认，全部忽略。原0即使RGB非零也继续忽略。主Shadow/Surface支持不含这些不确定区域；不能据此宣称全部L7标签语义已修复。','- 原2019包、早期口径错误与语义hold记录保留，数据准备见../phase246/phase246_20261031/integrated_finish/INTEGRATED_RESULT.md（按outputs根目录定位）。','','## 4．下一步边界','','保留研究问题：“什么条件下，同一监督头更新改善部分局部排序，却损伤跨场景一致性？”本轮只增加一个未观察到分歧的外推对照，不启动新模块。后续若继续，应事先固定对照条件并明确要区分的解释，优先使用现有L8/SPARCS/L7配对分数，而不能继续把通用失配当作已成立前提。']
 text='\n'.join(lines)+'\n'
 text=text.replace('见../phase246/phase246_20261031/integrated_finish/INTEGRATED_RESULT.md（按outputs根目录定位）','见[数据准备报告](D:/Cloud-Adapter-light/outputs/phase246/phase246_20261031/integrated_finish/INTEGRATED_RESULT.md)')
 (R/'RESULT.md').write_text(text,encoding='utf-8')
 files=[p for p in R.iterdir() if p.is_file() and p.name not in ['LOCAL_DELIVERY_MANIFEST.json','LATEST_REMOTE.json']]
 manifest={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
 (R/'LOCAL_DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'report':str(R/'RESULT.md'),'files':len(manifest),'passed':True}))
if __name__=='__main__':main()

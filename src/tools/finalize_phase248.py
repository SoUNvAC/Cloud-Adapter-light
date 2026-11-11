"""Local case record and metadata qualification delivery; no model calls."""
from pathlib import Path
import csv,json,hashlib,statistics,html
B=Path(__file__).resolve().parents[2];R=B/'outputs/phase248/phase248_20261031'
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def csvsave(n,rows):
 with (R/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 seal=js(R/'STAGE_A_REVIEW_SEAL.json');assert sha(R/'STAGE_A_REVIEW_LOCKED.json')==seal['sha256'];assert js(R/'STAGE_B_READY.json')['prediction_masked_review_SHA']==seal['sha256']
 first=js(R/'STAGE_A_REVIEW_LOCKED.json')['cases'];revealed=js(R/'stageB/STAGE_B_CASES.json');assert {r['case_id'] for r in first}=={r['case_id'] for r in revealed};lookup={v['case_id']:v for v in revealed};tech=js(R/'TECHNICAL_AUDIT.json');post=js(R/'IMPLEMENTATION_POSTCHECK.json');assert post['passed'];elig=js(R/'ELIGIBILITY_SUMMARY.json');assert elig['confirmation_ready']==0
 records=[];names={0:'Surface',1:'Cloud',2:'Shadow'}
 for r in first:
  b=lookup[r['case_id']];pred={m:v['class'] for m,v in b['model_centers'].items()};agree=len(set(pred.values()))==1
  if b['stratum'].endswith('disagree'):assert not agree
  elif b['stratum'].endswith('agree'):assert agree
  records.append({**r,'sampling_stratum_revealed_after_seal':b['stratum'],'Source_class':names[pred['Source']],'C21701_class':names[pred['C21701']],'C21702_class':names[pred['C21702']],'post_reveal_note':'中心类别仅为缓存模型输出，不据此改写阶段A判断或参考标签'})
 csvsave('CASE_RECORDS.csv',records)
 flows=js(R/'stageB/STAGE_B_ERROR_FLOWS.json');flow_rows=[];validated=[]
 for v in flows:
  cube=v['reference_Source_C_27'];prior=js(B/f"outputs/phase242/phase242_20261031/delivery/scene_receipts/test/{v['domain']}/{v['scene']}.json");assert cube==prior['C'][v['model']]['transition_reference_Source_C']
  total=[sum(sum(row) for row in layer) for layer in cube];assert total==prior['reference'];a=cube[2];n=sum(sum(row) for row in a)
  bd=v['Shadow_boundary_Source_C'];inside=v['Shadow_interior_Source_C'];assert all(bd[i][j]+inside[i][j]==a[i][j] for i in range(3) for j in range(3))
  row={'scene':v['scene'],'domain':v['domain'],'model':v['model'],'reference_Shadow':n,'Source_Shadow_recall_percent':100*sum(a[2])/n,'C_Shadow_recall_percent':100*sum(row[2] for row in a)/n,'Source_correct_to_C_Surface':a[2][0],'Source_correct_to_C_Cloud':a[2][1],'Source_wrong_to_C_Shadow':a[0][2]+a[1][2],'Source_Cloud_to_C_Surface_with_reference_Shadow':a[1][0],'C_Surface_fraction_of_reference_Shadow_percent':100*sum(row[0] for row in a)/n}
  for label,matrix in [('boundary',bd),('interior',inside)]:
   size=sum(sum(row) for row in matrix);source_wrong=sum(sum(row) for row in matrix[:2]);c_wrong=sum(row[0]+row[1] for row in matrix);row.update({label+'_support':size,label+'_Source_correct_to_C_wrong':matrix[2][0]+matrix[2][1],label+'_Source_error_percent':100*source_wrong/size,label+'_C_error_percent':100*c_wrong/size,label+'_error_increase_pp':100*(c_wrong-source_wrong)/size})
  flow_rows.append(row);validated.append({'scene':v['scene'],'model':v['model'],'old242_27_transition_exact':True,'reference_count_match':True,'boundary_plus_interior_exact':True})
 csvsave('SHADOW_ERROR_FLOWS.csv',flow_rows)
 norm=js(R/'stageB/INPUT_NORMALIZATION_AUDIT.json');norm_rows=[]
 for scene in dict.fromkeys(v['scene'] for v in norm):
  a=[v for v in norm if v['scene']==scene];norm_rows.append({'scene':scene,'inputs':len(a),'raw_RGB_span_median_DN':statistics.median(v['span_DN'] for v in a),'normalization_gain_median':statistics.median(v['normalization_gain'] for v in a),'span_min':min(v['span_DN'] for v in a),'span_max':max(v['span_DN'] for v in a)})
 csvsave('NORMALIZATION_SUMMARY.csv',norm_rows)
 ip=[];positions={v['scene']:v for v in post['scenes']}
 for v in tech:
  q=positions[v['scene']];assert v['inputs']==v['label_cache_exact']==v['input_hash_verified'];assert v['score_hash_verified']==3*v['inputs'];assert v['rgb_reconstruction_exact']==v['inputs'];assert v['rgb_reconstruction_max_error']==0
  ip.append({'scene':v['scene'],'domain':v['domain'],'inputs':v['inputs'],'date':q['expected_date'],'WGS84_bounds':json.dumps(q['WGS84_bounds']),'solar_elevation':v['sun']['SUN_ELEVATION'],'native_shape':json.dumps(v['georef']['native_shape']),'model_RGB_check':'same local11band RGB+minmax, exact' if v['domain']=='L8' else 'official photo RGB + INTER_LINEAR 1000->512 replay; not MS->photo reconstruction','label_cache_exact':v['label_cache_exact'],'score_hash_verified':v['score_hash_verified'],'cached_valid_reference_pixels':v['valid_cached_reference_pixels'],'raw_valid_reference_pixels':v['raw_valid_reference_pixels']})
 csvsave('INPUT_AUDIT.csv',ip)
 validation={'passed':True,'locked_review_SHA_unchanged':True,'cases':len(records),'sampling_agreement_strata_verified':True,'full_case_flows_against_Phase242':validated,'L8_RGB_patches_exact':sum(v['inputs'] for v in tech if v['domain']=='L8'),'score_files_hash_verified':sum(v['score_hash_verified'] for v in tech),'case_inputs_checked':sum(v['inputs'] for v in tech),'candidate_qualification_rows':elig['rows'],'new_forward':0,'new_training':0,'new_threshold_scan':0,'semantic_ground_truth_independently_confirmed':False}
 save('VALIDATION.json',validation)
 # Small targeted adjudication list contains every selected Shadow center in the key scene.
 a_index={v['case_id']:v for v in js(R/'STAGE_A_CASES.json')};targets=[]
 for v in records:
  if v['scene']==tech[0]['scene'] and v['reference_center']==2:
   a=a_index[v['case_id']];targets.append({'case_id':v['case_id'],'scene':v['scene'],'native_row':a['evaluation_row'],'native_col':a['evaluation_col'],'question':'能否独立确认是落地云影，而非云体、地形阴影或雪冰本征明暗？','judgment':'','basis':'','uncertainty':'','reviewer':''})
 csvsave('PRIORITY_ADJUDICATION.csv',targets)
 # Evidence viewer preserves the review order with model results behind disclosure controls.
 sections=[]
 for i,t in enumerate(tech,1):
  rows=[v for v in records if v['scene']==t['scene']];table='<table><tr><th>编号</th><th>阶段A原判断</th><th>观察</th><th>不确定性</th></tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v[k]))+'</td>' for k in ['case_id','assessment','visible_observation','uncertainty'])+'</tr>' for v in rows)+'</table>'
  sections.append(f'<section><h2>{html.escape(t["scene"])}</h2><p>先看影像/参考与已封存记录。图中红十字为中心。黑色包含窗口越界显示留白，不全部代表原始Fill。</p><a href="stageA/A_scene{i}_overview.png">原生场景概览</a><img src="stageA/A_scene{i}_sheet.png">{table}<details><summary>展开阶段B：Source/C预测对照</summary><img src="stageB/B_scene{i}_sheet.png"></details></section>')
 page='<!doctype html><meta charset="utf-8"><title>Phase248案例记录</title><style>body{font:16px system-ui;max-width:1150px;margin:24px auto;padding:0 16px}img{display:block;max-width:100%;margin:16px 0}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px;vertical-align:top}section{margin:48px 0}summary{cursor:pointer;font-weight:bold;background:#eee;padding:14px}</style><h1>固定五景案例审查</h1><p>40个目的性案例；AI视觉判断，非独立人工裁决，不能估算整景标注准确率。阶段A SHA：'+seal['sha256']+'</p>'+''.join(sections)
 (R/'CASE_VIEWER.html').write_text(page,encoding='utf-8')
 key=[v for v in flow_rows if v['scene']==tech[0]['scene']];control=[v for v in flow_rows if v['domain']=='SPARCS'];t=tech[0]
 lines=['# Phase248：固定五景小型案例审查','','## 结论先行','','**未发现本地发布影像到模型输入、标签、缓存与Phase242拼接绑定的实现错位；但关键冰雪景的物理云影参考，尚未获得独立确认。** 当前保留“发布参考口径下可复现的损伤”，暂不把它升级为已证实的云影识别机制失败。没有删除该景、重标或修复标签，也未运行新模型。','','另一项交付：[Landsat资格摘要](LANDSAT_ELIGIBILITY.md)、逐景LANDSAT_ELIGIBILITY.csv。现有库存中确认就绪的独立云影场景为0；元数据候选仍保留，不按模型表现选择。','','## 一、已确认事实','','### 数据和实现','','|固定场景|评价输入数|太阳高度|输入核查|','|---|---:|---:|---|']
 for v in tech:lines.append(f"|{v['scene']}|{v['inputs']}|{float(v['sun']['SUN_ELEVATION']):.2f}°|"+('本地11波段RGB重建全部逐值一致' if v['domain']=='L8' else '官方photo/mask/data逐字节一致；1000→512缩放复核')+'|')
 lines+=['','- 四景L8共486个512裁块，按既有B4/B3/B2、全通道minmax和uint8转换，从本地发布11波段栅格重建与实际PNG逐像元相同，最大差0；原生影像与参考的CRS/仿射/shape一致，日期/WRS与MTL一致。窗口坐标唯一且在512格网上，无重复/重叠窗口。','- 五景共487个输入的原生标签映射/有效区域与缓存一致；1461个分数文件及预测SHA/混淆核查通过。Phase241与Phase242的这1461条输入/分数/标签绑定逐项相同；新提取的10组27格错误流精确复现Phase242。未发现新的裁切、缩放或拼接异常。','- SPARCS控制的photo、mask、10波段data与官方本地ZIP中唯一成员CRC/SHA一致。其RGB双线性1000→512、标签最近邻1000→512为既定处理：评价格网约58.6m，不等于L8原生30m；不把这个缩放差异说成新发现的错误。','- L8所谓“原影像”在本轮是本地发布的11波段uint8栅格，不是原始单波段L1 uint16 DN原件。上游量化/合成过程未独立重建；SPARCS多波段到官方photo的颜色制作过程亦未重建。精确复现内层处理不等于证明上游辐射处理无误。','','### 参考口径下的错误流','','以下只统计既有评价裁块有效参考Shadow；不冒充未纳入裁块的全原生场景统计。','','|场景|Source召回|C21701召回|C21702召回|','|---|---:|---:|---:|']
 for scene in dict.fromkeys(v['scene'] for v in flow_rows):
  a,b=[v for v in flow_rows if v['scene']==scene];lines.append(f"|{scene}|{a['Source_Shadow_recall_percent']:.2f}%|{a['C_Shadow_recall_percent']:.2f}%|{b['C_Shadow_recall_percent']:.2f}%|")
 lines+=['',f"关键景共有7,511,161个有效参考Shadow。Source召回原本仅5.63%，不能说这是一个原本云影识别良好的场景。适配后大部分参考Shadow被判Surface，分别占{key[0]['C_Surface_fraction_of_reference_Shadow_percent']:.2f}%／{key[1]['C_Surface_fraction_of_reference_Shadow_percent']:.2f}%。",'','|关键景指标|C21701|C21702|','|---|---:|---:|']
 for label,col in [('Source原正确Shadow→Surface','Source_correct_to_C_Surface'),('Source原正确Shadow→Cloud','Source_correct_to_C_Cloud'),('Source原错误→恢复Shadow','Source_wrong_to_C_Shadow'),('原已错判Cloud→改判Surface','Source_Cloud_to_C_Surface_with_reference_Shadow')]:lines.append(f"|{label}|{key[0][col]:,}|{key[1][col]:,}|")
 lines+=['','最大的改变流还包括三百多万原已被Source错判Cloud的参考Shadow转为Surface；这不是同等数量的“新增错误像元”。此处区别分类错误总量、错误类别迁移和跨景排序损伤。','','|关键景参考区域|Shadow支持|C21701错误率增量|C21702错误率增量|','|---|---:|---:|---:|']
 for label,k in [('内部','interior'),('边界','boundary')]:lines.append(f"|{label}|{key[0][k+'_support']:,}|{key[0][k+'_error_increase_pp']:+.3f}pp|{key[1][k+'_error_increase_pp']:+.3f}pp|")
 lines+=['','边界定义为参考Shadow内距边缘3个评价像元，L8先在原生整景生成再裁块，避免把输入裁块边缘当参考边界。关键景内部支持占87.60%；原正确Shadow的新增丢失约83.8%／84.1%在内部。**损伤不是仅在边界，但不能据计数说内部单位像元风险更高；边界的错误率增幅反而更大。**','','SPARCS控制的局部跨景排序损伤也不能等同为全景漏影增加：其参考Shadow召回反而提升到57.17%／50.67%，恢复的Shadow多于丢失者。原正确Shadow丢失分别1362／2525个，其中流向Cloud为1150／1618个，与关键L8景几乎全流向Surface不同。','','### 判读顺序与样本边界','','每景8个、共40个目的性小案例，覆盖模型一致/分歧、参考Shadow内部/边界、Surface和空间对照；不挑分数损伤极值。程序据旧预测界定抽样层，图版中遮蔽层和预测、打乱编号。先完成影像/参考审查并锁定STAGE_A_REVIEW_LOCKED.json的SHA，远端验证后才生成Source/C图版。原判断保持不变。','','这是AI视觉审查，审查者知道此前研究背景，不是独立人工盲审。部分图版边缘黑色来自展示窗口越界补黑，和原生Fill不同；原生上下文窗768像元、细节窗192像元，L8输入细节不提供裁块外上下文。40点不能推算整景标签正确率。','','## 二、合理怀疑（尚非已证实原因）','','1. **极低太阳高度与冰雪参考歧义。** 关键景为2014-02-23、南极区域约80°S，太阳高度9.446°；三个Surface来源景约59–63°，SPARCS控制38.98°。图版可见大范围低对比亮地表及明暗条带。云体、落地云影、地形阴影和雪冰本征明暗的区分不足；当前不能确认这些暗部的物理来源，也不能仅凭低太阳角认定错标。','2. **既定逐块归一化对低动态范围影像的放大。** 此过程逐值复现，不是新发现的程序错误。关键景113块RGB原始跨度中位数仅14 DN，线性增益中位数18.21；三个Surface来源景分别89、163.5、176 DN，对应2.87、1.56、1.45。图上模型输入的颜色/细纹理明显被放大。这是场景相关输入条件，不能由这次观察断言它造成C损伤，也没有据此更换归一化或重跑挑优。','3. **局部参考切分争议。** 在V2_05、V3_04，参考Surface中心附近可见浅云丝/云边，阶段A已标记合理怀疑；多个云/影/地表过渡点属于混合。它们是具体待裁决位置，不是整景标签错误率证据。','','## 三、无法判断','','- 关键景四个抽中的参考Shadow中心（V1_02、V1_03、V1_06、V1_08）究竟是不是落地云影，本轮RGB审查没有独立确认；没有任何点被本轮正式裁为错标。','- 没有多时相、独立人工裁决或可靠高度/地形证据，不能把主损伤归因为标注差异、地形、太阳角、输入归一化或模型特征中的任意一个。','- 没有原始L1单波段输入/处理链重建，不能排除上游制作影响；本轮没有发现足以要求修复的确定输入/配准错误。','- 既有发布参考下的排序损伤与物理云影识别损伤尚不能等同；不能为了去掉异常而删除关键景或修改标签。','','## 四、本轮决策','','**保留案例，收窄主张，暂不据此扩展机制实验。** 输入到缓存的内层实现可信，参考口径下的损伤可复现，而且包含参考内部；但关键景的云影语义还不够清楚，未满足“数据和参考基本可信后再作机制外推”的条件。','','下一项真正需要补的是这四个已固定Shadow中心及其上下文的独立遥感判读，明确落地云影/云体/地形阴影/雪冰明暗与不确定性。PRIORITY_ADJUDICATION.csv给出坐标和留空判断字段，不要求重标整景。若仍无法裁决，论文应将其作为发布参考口径下的异常评分案例，不能宣称普遍云影机制失效。','','## 交付与成本','','- [图像与逐例记录页面](CASE_VIEWER.html)：先A影像/参考/封存判断，再展开B预测。','- CASE_RECORDS.csv保留40例的阶段A原判断及后续预测中心；SHADOW_ERROR_FLOWS.csv为10组全缓存统计。','- INPUT_AUDIT.csv、NORMALIZATION_SUMMARY.csv、IMPLEMENTATION_POSTCHECK.json记录输入、位置、官方ZIP及阶段间绑定。','- LANDSAT_ELIGIBILITY.csv为399条Landsat资格记录；已暴露/待核实/排除分开，确认就绪0，不能把行数当独立样本数。','- 新训练、新模型前向、新阈值搜索均0。阶段A CPU流水线37.35秒、阶段B 9.53秒；额外ZIP/元数据核查耗时未统一计量，不声称这是全任务总耗时。原始影像、标签和分数缓存均未修改。']
 text='\n'.join(lines)+'\n'
 text=text.replace('此处区别分类错误总量、错误类别迁移和跨景排序损伤。','此处区别分类错误总量、错误类别迁移和跨景排序损伤。Source原argmax召回低不等于其排序差，也不抵消此前Source排序优于C的证据。')
 (R/'CASE_REVIEW.md').write_text(text,encoding='utf-8')
 manifest={str(p.relative_to(R)):{'bytes':p.stat().st_size,'SHA256':sha(p)} for p in R.rglob('*') if p.is_file() and p.name not in ['DELIVERY_SHA256.json']}
 save('DELIVERY_SHA256.json',manifest);print(json.dumps({'passed':True,'cases':len(records),'score_files':validation['score_files_hash_verified'],'flow_groups':len(flow_rows),'eligibility_rows':elig['rows'],'files':len(manifest)},ensure_ascii=False))
if __name__=='__main__':main()

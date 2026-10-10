"""Join actual visual observations to fixed points; never derive terrain from GT."""
import csv,hashlib,json
from pathlib import Path
from collections import Counter

ROOT=Path('outputs/phase73/phase73_20261010')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def table(p,data):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def main():
    points=rows(ROOT/'sampled_points.csv');observations=read(ROOT/'visual_observations.json')
    assert set(observations['points'])=={p['point_id'] for p in points}
    labels=['Surface','Cloud','Shadow'];reviewed=[]
    for p in points:
        pid=p['point_id'];types,evidence,confidence=observations['points'][pid]
        issue,note=observations['label_page_notes'][pid.rsplit('_',1)[0]]
        transition=f"GT={labels[int(p['gt'])]}, L3={labels[int(p['L3_prediction'])]}, B={labels[int(p['B_prediction'])]}; "
        reviewed.append(dict(point_id=pid,product=p['product'],group_id=p['group_id'],category=p['category'],
            imagery_types=types,imagery_evidence=evidence,confidence=confidence,label_issue_types=issue,label_evidence=transition+note,
            reviewer='Codex_AI_not_human_ground_truth',status='reviewed_with_uncertainty'))
    table(ROOT/'interpretation_reviewed.csv',reviewed)
    flows=rows(ROOT/'error_flows.csv');picks=read(ROOT/'selected_groups.json')
    summary=read(ROOT/'summary.json');summary['status']='completed_with_uncertain_AI_visual_typing'
    summary['visual_review']=dict(points=len(reviewed),reviewer='Codex AI',independent_human_labels=False,
        imagery_pages=24,GT_prediction_pages=24,score_display_QA_points=7,
        confidence_counts=dict(Counter(r['confidence'] for r in reviewed)),
        no_confirmed_label_error_rate=True,no_confirmed_registration_error=True)
    summary['pooled_error_flows']={n:{k:sum(int(r[n+'_'+k]) for r in flows) for k in ['Surface_to_Shadow','Cloud_to_Shadow','Shadow_to_Surface','Shadow_to_Cloud']} for n in ['L3','B']}
    for n in ['L3','B']:
        summary[n+'_group_mean_error_flow']={}
        for k in summary['pooled_error_flows'][n]:
            values=[float(r[n+'_'+k+'_fraction']) for r in flows if r[n+'_'+k+'_fraction']]
            summary[n+'_group_mean_error_flow'][k]=dict(groups=len(values),fraction=sum(values)/len(values))
    (ROOT/'completion_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# Phase73：固定缓存诊断与失败分型','', '## 1. 数据结果','',
        '已完成64相关组错误流；固定选7组、96点（新增FP28、丢失TP24、救回TP20、稳定TP24），每点原生257×257。64组L3/B confusion均精确复现Phase72，无拟合、调阈值或新网络前向。',
        '已先看24页RGB/FCC/B08/B11/B12，再看24页GT/L3/B；96点均留有AI判读依据与置信等级，非人工审核真值。7点分数显示另作视觉QA，所有96点原始中心logit已保存。',
        '', '| 错误流（全部有效像元计数，不是独立样本数） | L3 | B | B−L3 |','|---|---:|---:|---:|']
    for k in summary['pooled_error_flows']['L3']:
        a=summary['pooled_error_flows']['L3'][k];b=summary['pooled_error_flows']['B'][k];lines.append(f'| {k} | {a:,} | {b:,} | {b-a:+,} |')
    lines+=['', '新增FP966,831，消除FP212,813，净增754,018；救回TP125,635，丢失TP37,198，净增88,437。逐组计数、真类分母、比例及组平均摘要均保留，不以像元数做显著性检验。',
        '', '支持差异：T20FNH公开Shadow比例0.585743773959199%，官方发布mask Shadow通道0像元，有效Shadow0、无效区域Shadow0。差异不是有效mask删除云影造成的；元数据与发布标签不一致，上游原因unknown，不替换该组或改通过规则。',
        '', '| 代表tile | 固定选入原因 | ΔAP（pp） | ΔFPR（pp） | 新增FP | 丢失TP |','|---|---|---:|---:|---:|---:|']
    for product,pick in sorted(picks.items()):
        r=pick['row'];ap='未定义' if r['delta_ap_pp'] is None else f"{r['delta_ap_pp']:+.4f}"
        tile=product.split('_T')[1].split('_')[0]
        lines.append(f"| T{tile} | {','.join(pick['reasons'])} | {ap} | {r['delta_fpr']*100:+.4f} | {r['new_FP']:,} | {r['lost_TP']:,} |")
    lines+=['', '## 2. 结论探讨','',
        'B既救回真实参考云影，又制造更多误报；仅凭净误报增加，不能判断是否只是阈值校准不足，也不能由本轮直接断言具体机制。T06CWV的浅暗斑被明显缩小，同时在山脊暗坡上增加Shadow；同景较深暗团仍能保留，说明不是所有云影一概丢失。',
        'T36TVP/S01、T22HEK/S03、T51QWF/S05等不同相关组均出现B较L3边缘细碎、61邻域混入亮云与暗区的图像线索；这可称重复诊断现象，但“局部均值被云污染导致损伤”仍是待否证假说，尚无因果证明。',
        'T52WDT/S02_new_FP_01–04表现为随山脊坡谷展开的暗区，GT/L3为Surface而B为Shadow；T06CWV/S07_new_FP_01–04也呈雪冰山脊暗坡。只有2组，不称跨至少3组重复。T19NEB/S06_new_FP_01–02位于连续弯曲水道，属于一个相关组的个案；水体不能从其他通体暗的窗口强行推断。',
        '疑似云边界、混合窗口与暗背景均已逐点记录。未证实系统性配准错误或标签漏标，不能用“标签坏了”抵消Phase72失败；也不能由诊断抽样估计总体地物比例/标签错误率。',
        '', '### 后续最多两个分枝（仅准备，不执行）','',
        '**分枝1：检验局部光谱证据对云边混合的敏感性。** 证据为S01_new_FP_02、S03_new_FP_02、S05_new_FP_04（完整产品见抽点表），不同组同时存在救回/丢失点。可否证假说：局部光谱项在混合邻域引入与真实云影无关的高频分数变化。所需信息：影像、已冻结B/L3分数、Source非Shadow类别/边界信息；已有材料可用于准备，但可靠的独立语义验证仍不足。最小对照：以后另立协议，在fit/development比较原B与同容量、固定云边一致性约束版本，保持窗口/公式/阈值选择规则一致；若误报不减或恢复收益消失则不支持。该对照本轮未执行。',
        '**分枝2：区分地形阴影与目标云影。** 证据为S02_new_FP_01–04和S07_new_FP_02–04，仅两组个案。可否证假说：缺少照明几何使B把坡面阴影判作云影。所需信息：精确产品太阳角、cube CRS/affine/像素大小/裁切对应及合法可用DEM；现有Phase70记录不齐，输入not_ready。最小对照：待独立核验格网后，保持原B冻结，与加入固定物理地形照明排除约束的同输出比较，并检查救回TP是否受损；不得用GT挑排除区域。不以调整阈值/窗口或删除失败景代替该对照，本轮不下载、不投影。',
        '太阳角、cube地理对应缺失维持not_ready，标注shapefile CRS不能直接赋给cube，云高若未来投影需要也必须独立取得。两个分枝只是备选假说，不是已验证方法。新方法仍需新的未使用数据确认；原65c已使用，Phase72失败及原H1支持门结论保持。',
        '', '## 文件与复现','',
        '- `error_flows.csv`：64组全量错误流及分母。`support_difference.csv`：12→11差异。',
        '- `selected_groups.json`、`sampling_audit.json`、`sampled_points.csv`：固定选组/坐标/seed/候选及索引SHA；完整产品ID为准。',
        '- `blind_pages/`、`label_pages/`：先影像后标签的96点总览；`panels/`：每点影像与GT/预测/分数面板共192文件。',
        '- `interpretation.csv`为远端原始空模板，保留不覆盖；**最终判读用`interpretation_reviewed.csv`**，原始实际观察保存在`visual_observations.json`。',
        '- 绿色Surface、黄色Cloud、蓝色Shadow，灰色无效/边缘补区，洋红中心十字与61×61框。分数图为固定sigmoid(logit)的[0,1]灰阶，只用于显示，L3/B冻结logit阈值不同；不能拿相同灰度误判同决策。',
        '- `display_stretch.json`固定每景image-valid的2/98显示拉伸，不进入特征。`geometry_missing.csv`及`geometry_readiness.json`仅查已有记录。',
        '- `input_sha256.json`绑定原始缓存/锁/代码，`remote_sha256.txt`及`local_verification.json`核对共享交付；`completion_summary.json`加入本地真实判读状态，远端`summary.json`仍保留生成时pending历史。',
        '- 实际远端新目录为shared下`phase73_20261010_retry01`，成功PID567462/EXIT0。首次PID567392在面板变量冲突后退出1，失败产物保留；修复不变样本、模型或阈值。只传shared，不涉及work_dirs。']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()

"""Render Phase94 numerical results and interpretation separately from sealed outputs."""
import argparse, hashlib, json
from pathlib import Path

def render(root):
    read=lambda n:json.loads((root/n).read_text(encoding='utf-8-sig'))
    manifest=read('DELIVERY_SHA256.json')
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
    r=read('phase94_report.json');t=read('four_threshold_lock.json');sl=read('split_input_lock.json')
    f=lambda x:'未定义' if x is None else f'{100*x:.4f}'
    pp=lambda x:'未定义' if x is None else f'{x:+.4f}'
    ci=lambda it:'未定义' if it['ci95'] is None else '['+', '.join(f'{x:+.4f}' for x in it['ci95'])+']'
    lines=['# Phase94：留出相关组校准实验','','## 1．数据结果','','### 固定设计与执行','',
        '从原153个fit相关组按SHA256(PHASE94_SPLIT_V1|group_id)升序固定选100组拟合，余53组仅校准。只用原代表与原抽样索引；输入SHA与分组在读取标签数组前锁定。两次新拟合，不复用153组模型参数、不再拟合153组。',
        'G7_100使用六原始TOA波段及Source云影logit，复用Phase93的200轮固定HGB配置（模型random_state=93012）；SourceS_100使用Source云影logit与B08/B11/B12，沿用Phase67约束逻辑回归。标准化仅100组。',
        '两模型各按全部有效非Shadow像元冻结T_in（100拟合组）和T_cal（53校准组），每组等权、exact-ties、宏FPR≤1%；未用development/ALCD选阈值。',
        '评价32 development组（7组有参考云影）、ALCD原版9组、原版/修订版共同6组。零分割网络训练、零网络前向、零影像下载。区间仅相关组bootstrap10000次，seed94012。','',
        f"实际拟合尝试{r['new_fits']}次（最终2模型）。若授权故障恢复，仅SourceS补算；G7复用保存系数。最初SourceS拟合因标准化浮点验收过严在保存前中止，首进程成本未保存，不补造；详见repair_authorization_lock.json。",'',
        '### 四阈值及校准边界','',
        '|模型|阈值来源|原分数阈值|对应概率阈值|参与负类组数|本来源宏FPR %|放宽至下一并列分数的FPR %|',
        '|---|---|---:|---:|---:|---:|---:|']
    for n,m in t['models'].items():
        for subset,v in m.items():lines.append(f"|{n}|T_{subset}|{v['threshold']:.12g}|{v['probability_threshold']:.12g}|{v['negative_groups']}|{f(v['macro_fpr'])}|{f(v['next_macro_fpr'])}|")
    lines+=['','G7阈值直接与float32概率比较。SourceS在development比较float32 logit；ALCD先在20m对logit取sigmoid，再float64平均3×3到60m，与上表对应概率阈值比较；不先平均波段或logit。','',
        '### 冻结工作点','',
        '|评价池|策略|宏AP %|宏召回 %|宏精确率 %|宏FPR %|FPR单侧95%上界 %|宏Shadow IoU %|超1%组数|零召回组数|池化AP %|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for domain,v in r['domains'].items():
        for n,s in v['policies'].items():
            lines.append('| '+' | '.join([domain,n,*[f(s['macro'][k]['mean']) for k in ['ap','recall','precision','fpr']],f(s['macro']['fpr']['upper95_one_sided']),f(s['macro']['shadow_iou']['mean']),f"{s['groups_above_1percent']}/{s['groups']}",str(len(s['zero_recall_groups'])),f(s['pooled']['ap'])])+' |')
    lines+=['','### 同模型T_cal−T_in：配对组差，单位百分点','',
        '|评价池|模型|ΔFPR [95%CI]|Δ召回 [95%CI]|Δ精确率 [95%CI]|ΔShadow IoU [95%CI]|',
        '|---|---|---|---|---|---|']
    for domain,v in r['domains'].items():
        for n in ['G7_100','SourceS_100']:
            c=v['comparisons'][n+'_T_cal_minus_'+n+'_T_in'];cells=[]
            for k in ['fpr','recall','precision','shadow_iou']:
                it=c['delta_'+k+'_pp'];cells.append(pp(it['mean'])+' '+ci(it))
            lines.append('| '+' | '.join([domain,n,*cells])+' |')
    lines+=['','### 两模型排序差：G7_100−SourceS_100，与阈值无关','',
        '|评价池|逐组平均AP差，百分点|配对95%CI|最差逐组AP差，百分点|',
        '|---|---:|---|---:|']
    for domain,v in r['domains'].items():
        c=v['comparisons']['G7_100_T_cal_minus_SourceS_100_T_cal'];it=c['delta_ap_pp'];vals=[x['delta_ap_pp'] for x in c['pairs'] if x['delta_ap_pp'] is not None]
        lines.append(f"|{domain}|{pp(it['mean'])}|{ci(it)}|{pp(min(vals))}|")
    lines+=['','### 最差逐组召回变化（T_cal−T_in）','',
        '|评价池|模型|产品|Δ召回，百分点|',
        '|---|---|---|---:|']
    for domain,v in r['domains'].items():
        for n in ['G7_100','SourceS_100']:
            worst=v['comparisons'][n+'_T_cal_minus_'+n+'_T_in']['worst_recall']
            if worst:lines.append(f"|{domain}|{n}|{worst['product']}|{pp(worst['delta_recall_pp'])}|")
    lines+=['','### 训练/校准诊断：宏FPR %，不称未见组表现','',
        '|模型|数据|阈值来源|宏FPR %|单侧95%上界 %|',
        '|---|---|---|---:|---:|']
    for v in read('train_calibration_diagnostics.json')['rows']:
        it=v['macro_fpr'];lines.append(f"|{v['model']}|{v['data_subset']}|T_{v['threshold_source']}|{f(it['mean'])}|{f(it['upper95_one_sided'])}|")
    lines+=['','### 成本与验收','',
        f"拟合、输入验证和全部校准总计 {r['cost']['fit']['seconds']:.2f} 秒；独立评价 {r['cost']['evaluation_seconds']:.2f} 秒。拟合进程累计RSS高水位 {r['cost']['fit']['process_peak_rss_bytes']/2**30:.3f} GiB；评价进程高水位 {r['cost']['evaluation_peak_rss_bytes']/2**30:.3f} GiB；最多4 CPU线程。RSS不是单个模型孤立内存。",
        'CPU复算检查逐组/池化整数计数、bootstrap区间、四个阈值来源预算与下一个exact-tie边界、两阈值AP不变量。独立树/线性计算仅确定性探针，mean9与阈值决策逐块检查；不宣称所有像元做了第二次树推理。',
        '全部逐组指标、无参考云影组误报、Surface/Cloud分层FPR、池化召回/精确率/FPR及最差损伤保存在per_group.csv、four_strategy_totals.csv和phase94_report.json。二元分数不构造三类mIoU。原153组模型只在historical_reference.csv附列。','',
        '## 2．结论探讨','']
    for domain,v in r['domains'].items():
        for n in ['G7_100','SourceS_100']:
            a=v['policies'][n+'_T_in'];b=v['policies'][n+'_T_cal'];d=v['comparisons'][n+'_T_cal_minus_'+n+'_T_in']
            point=b['macro']['fpr']['mean'];upper=b['macro']['fpr']['upper95_one_sided']
            verdict='点估计仍超1%预算' if point>.01 else ('点估计与单侧95%上界均满足1%预算' if upper is not None and upper<=.01 else '点估计满足1%，但区间上界超预算，稳定性未获支持')
            lines.append(f"- {domain}的{n}：留出阈值下{verdict}。相对训练内阈值，宏FPR变化{pp(d['delta_fpr_pp']['mean'])}个百分点、宏召回变化{pp(d['delta_recall_pp']['mean'])}个百分点。")
    lines+=['','阈值变动没有恢复排序：同模型原始分数、每组AP与池化AP在两个阈值间完全相同。若误报降低但召回同时降低，属于工作点权衡。不能因两个模型实际FPR不同而仅按召回排名选择部署赢家。',
        '本轮只隔离同一100组模型更换校准来源的影响。不能用原153组模型与新100组模型的差异归因于单一校准因素；拟合组数、校准组组成、域变化和60m聚合仍可能共同作用。53组内满足预算不等于未见组或跨域保证。',
        'development与ALCD均已暴露，所有区间是探索性描述，原独立支持门失败结论不变。不追加第三种校准、置信预算搜索、模型重拟合或部署赢家选择。',
        '人工B未执行，原缺失字段保留；Phase89的54块恢复概率没有故障前SHA，这一谱系限制继续存在。标签修订差异不是像素错误率，修订标签不被假定为绝对真值。']
    out=root/'RESULTS_AND_INTERPRETATION.md'
    with out.open('x',encoding='utf-8') as h:h.write('\n'.join(lines)+'\n')
    return out
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);print(render(a.parse_args().root))

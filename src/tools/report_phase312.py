"""Concise fixed-readout evidence report, with full tables separately preserved."""
from phase312_common import *

def report():
    metrics=read(ROOT/'METRICS.csv');selection=[r for r in read(ROOT/'SELECTION.csv') if r['product']=='ALL'];ranks=[r for r in read(ROOT/'RANKING_SUMMARY.csv') if r['snow_ice']=='ALL'];fits=read(ROOT/'MODEL_FIT.csv');accept=load(ROOT/'CPU_ACCEPTANCE.json')
    lines=['# 数据结果','',
        '只用100 fit影像的Source Q90二类伪标签，从零各拟合一次RGB3/RGB6Q/MS6；Surface正类1、Cloud负类0。RGB与原始TOA六波段不乘255/不截断，六波段共同有效V6；全32组不参与采样、scaler或拟合。单PCG64(31201)，每产品最多8192，不做类平衡，非空产品总权重相同。',
        '目标为加权binary CE+0.5×0.001||beta||²，截距不惩罚，零初始L-BFGS-B maxiter500/gtol1e−7/ftol1e−12；固定rSurface≥0.90。模型只由Source伪标签拟合，不是监督Phase307探针；rSurface不是校准正确率。',
        '拟合状态：'+'；'.join(r['model']+':'+r['status']+'，样本'+r['samples']+'，迭代'+r['iterations'] for r in fits)+'。',
        '', '|读出|组均/池化mIoU %|Surface/Cloud/Shadow IoU %|Shadow宏/池化AP %|', '|---|---|---|---|']
    for r in metrics:
        fmt=lambda k:f'{float(r[k]):.4f}'
        lines.append('|'+r['strategy']+'|'+fmt('group_mean_mIoU_pct')+'/'+fmt('pooled_mIoU_pct')+'|'+'/'.join(fmt(k) for k in ('Surface_IoU_pct','Cloud_IoU_pct','Shadow_IoU_pct'))+'|'+fmt('macro_AP_pct')+'/'+fmt('pooled_AP_pct')+'|')
    lines+=['', '|规则|影像选中|参考D/K/Z选中|原D修复/余留|原B保留/撤销|', '|---|---|---|---|---|']
    for r in selection:lines.append('|'+r['rule']+'|'+r['selected_image']+'|'+'/'.join(r[k] for k in ('selected_D','selected_K','selected_Z'))+'|'+r['original_D_repaired']+'/'+r['original_D_remaining']+'|'+r['original_B_retained']+'/'+r['original_B_revoked']+'|')
    lines+=['', 'C_image=P∩V6∩PLCloud，按纯影像选取，参考无效不参与排序；只在独立评价时按原共同支持计D/K/Z。D选中是修复，K选中撤销正确Cloud，Z选中仅转另一错误。MS6等数量规则在每产品影像C内固定相同k，tie按flat升序；它们是选择性诊断，不是RGB可部署阈值。',
        '排序AUC（D为正，单类组NA；组均/池化不互代）：'+'；'.join(r['score']+f" {r['group_mean_AUC']}/{r['pooled_AUC']}，可算{r['AUC_groups']}/{r['total_groups']}组" for r in ranks)+'。',
        '全部32组/逐组union及六向错误/27格、V6额外排除、snow/ice=0/1/缺失和标签事后审计另表。Shadow二值与评分逐像元保持PL原件，所以Shadow AP/召回/FPR严格不变，不能算新增改善。', '', '# 结论分析','']
    by={r['strategy']:r for r in metrics};sel={r['rule']:r for r in selection}
    if 'MS6_90' in by:
        ms=by['MS6_90'];pl=by['Q90-PL']
        lines.append(f"MS6自身0.90相对PL组均/池化mIoU变化{float(ms['group_mean_mIoU_pct'])-float(pl['group_mean_mIoU_pct']):+.6f}/{float(ms['pooled_mIoU_pct'])-float(pl['pooled_mIoU_pct']):+.6f}pp。")
        if 'RGB6Q_90' in by:
            ref=by['RGB6Q_90'];lines.append(f"相对同参数RGB6Q自身0.90，MS6组均/池化mIoU变化{float(ms['group_mean_mIoU_pct'])-float(ref['group_mean_mIoU_pct']):+.6f}/{float(ms['pooled_mIoU_pct'])-float(ref['pooled_mIoU_pct']):+.6f}pp；覆盖量不同，不能单由该差值断言光谱增量。")
        lines.append('同数量比较必须同时看修复D与撤销K：'+'；'.join(r['rule']+' D/K/Z='+r['selected_D']+'/'+r['selected_K']+'/'+r['selected_Z'] for r in selection if r['rule']=='MS6_90' or r['rule'].endswith('equalMS6'))+'。')
        largest=next(r for r in read(ROOT/'PER_GROUP_METRICS.csv') if r['strategy']=='MS6_90' and 'T25XEG' in r['product']);damage=next(r for r in read(ROOT/'SELECTION.csv') if r['rule']=='MS6_90' and 'T25XEG' in r['product'])
        lines.append(f"最大历史损伤组T25XEG：MS6 mIoU={float(largest['mIoU_pct']):.4f}%，SurfaceIoU={float(largest['Surface_IoU_pct']):.4f}%，union={largest['union_classes']}；全D={damage['original_D']}，可选C中的D={damage['D']}，修复{damage['selected_D']}、余留{damage['original_D_remaining']}。该组单类AUC不能证明可分。")
        concentration=next(r for r in read(ROOT/'REPAIR_CONCENTRATION.csv') if r['rule']=='MS6_90' and r['top_k']=='3');lines.append(f"MS6修复中前3组占比={concentration['fraction']}，只作事后描述；固定历史前三组全部附表，不以该排序重采样或选组。")
    else:lines.append('MS6未成功收敛，不给出其阈值/等数量规则的成功结果；失败参数、梯度及具体原因保留，其他成功模型与基线照常交付。')
    lines+=['原D固定506798，原Cloud纠正B固定784478。B的Source类别为Shadow而P必须Source-Surface，因此二者天然不交：本轮B全部保留是结构恒等，不是额外学会保留。K撤销计数才显示Source-Surface区的Cloud正确损失；区域外D仍保留，不能拿C内修复率冒充总D修复。',
        '拟合样本事后参考审计状态：'+load(ROOT/'FIT_LABEL_AUDIT_STATUS.json')['status']+'；仅审计已选样本，排除参考无效，不用于重采样、重拟合或证明泛化。若失败，具体缺项/原因保留在FIT_LABEL_AUDIT_STATUS.json，主要32组结果不受其覆盖。',
        'RGB6Q只是同参数数对照，不是RGB非线性上界。Source伪标签可能复制亮地表错误；阴性结果不证明光谱无信息。scene snow/ice仅元数据，不是像元雪冰真值。单传感器、历史development反复暴露，非新独立测试；不证明云雪物理机制、可部署泛化或最终论文贡献。',
        '神经训练/前向/下载均0；无CV、拟合重启、正则/阈值扫描或追加分类器。不自动开始下一phase。']
    (ROOT/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

"""Render the fixed replay tables and short qualified historical conclusion."""
import csv,datetime,hashlib,json,os
from pathlib import Path
from prepare_phase201 import R,O,sha,write
def readcsv(name):
    with (O/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def fmt(x,percent=False):return 'NA' if x is None or x=='' else f'{float(x)*(100 if percent else 1):.6f}'
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(v).replace('|','/') for v in x)+' |' for x in rows)+'\n'
def main():
    a=json.loads((O/'ANALYSIS.json').read_text(encoding='utf8'));ind=json.loads((O/'INDEPENDENT_REVIEW.json').read_text());assert ind['status']=='passed'
    checks=readcsv('metric_reproduction.csv');assert all(x['status']=='passed' for x in checks)
    rows=[]
    states={'feasible':'满足','budget_violation':'预算违反','no_feasible_candidate':'本池无可行方案','not_applicable':'不适用'}
    for x in a['selection']:
        if x['status']!='executed':rows.append([x['block'],x['rule'],'不可执行' if x['status']=='not_executable' else '不适用']+['NA']*10);continue
        rows.append([x['block'],x['rule'],x['selected'],fmt(x['selection_score'],True),f"{x['eval_groups']}/{x['eval_AP_groups']}",fmt(x['eval_macro_AP'],True),fmt(x['eval_macro_recall'],True),fmt(x['eval_macro_FPR'],True),fmt(x['eval_max_FPR'],True),x['dominators'],states[x['mean_state']],fmt(x.get('mean_loss_pp')),states[x['tail_state']]+' / '+fmt(x.get('tail_loss_pp'))])
    s=table(['块','选择规则','选中模型','选择侧分数%','评价组数/有影组数','评价宏AP%','宏召回%','宏误报%','最大组误报%','支配者/无','平均预算状态','召回损失pp/NA','加尾部状态/损失pp'],rows)
    s+='\nA/B全图；C/D的误报仅指PixBox采样负类，不能称全景FPR。A/B/C预算1%，辅助最大组10%；D只AP选择，不比较1%预算损失。C无共同三类输出，mIoU规则不可执行。所有排序按未舍入值，精确并列按固定候选左到右；未被支配不等于最优。\n'
    (O/'TABLE1.md').write_text(s,encoding='utf8')
    rows=[]
    for x in a['definition']:
        flow='; '.join(k+'='+str(x[k]) for k in ['surface_to_cloud','cloud_to_surface','shadow_to_surface','shadow_to_cloud','nonshadow_to_shadow'])
        rows.append([x['phase'],x['rule'],x['selected'],fmt(x['old_macro_miou'],True),fmt(x['reference_fixed_macro_miou'],True),fmt(x['pooled_miou'],True),x['errors'],fmt(x['mean_error_rate'],True),flow])
    (O/'TABLE2.md').write_text(table(['候选池','mIoU规则','选中模型','旧宏mIoU%','参考类固定宏mIoU%','池化mIoU%','总错误像元','组均误分率%','主要错误流'],rows)+'\n旧定义：每组按union>0的类平均；MET定义：每组只按参考存在的类平均。二者再对32组等权；池化mIoU使用汇总混淆矩阵。MET是同侧定义敏感性，不是跨池独立证据；像元等代价目标不自动优于类别均衡目标。全部8候选和256组结果见met_all_candidates_and_groups.csv。\n',encoding='utf8')
    text='''# Phase201：历史模型选择回放

## 数据结果

- 固定A/B/C/D四块：选择侧均为原32组（7组有参考云影）；评价侧分别为Phase66原64组/15有影、Phase72原65c64组/11有影、C与D各PixBox10组/8有影。10条规则可执行；C缺共同三类输出，mIoU规则不可执行；D按协议不做mIoU或预算损失。完整[表1](TABLE1.md)、[表2](TABLE2.md)及未舍入CSV保留所有候选/逐组结果。
- **没有被支配的选中模型。** C122在固定候选池中受R支配，但没有规则选中C122。未被支配不代表最优。
- **A：池化AP选L2**，评价侧1%预算下比事后可行参照L3少1.432223个百分点宏召回，条件95%描述区间[-2.300499,5.252657]；加10%尾部约束仍相同。池化mIoU与宏AP选L3，损失均0。
- **B：三条规则均选B**；评价宏召回73.250095%、宏误报2.018860%、经验最大组误报36.335262%，平均预算超出1.018860个百分点，召回损失填NA。只有L3满足平均1%预算；加最大组10%约束后全部候选不可行，不把召回置0制造差距。
- **C：池化AP选C123**，本候选池预设目标下损失0；**宏AP选R**，比可行参照C123少6.071691个百分点宏召回，条件95%描述区间[-4.813011,17.574801]；加尾部约束仍相同。两seed分别是候选，没有平均分数或指标合成模型。
- **D：两条AP规则均选E**，评价侧未被R或N按三项目标支配；评价宏AP最高对象改为R仅是名次变化，不能判选模失败。D采用原默认工作点，不称同误报预算比较。C/D本次各自支持均为379正/8190负，未拼接其他历史阶段8188负类口径。
- **MET：Phase81旧宏mIoU选LL，参考类固定规则选LM**，总误分2,382,953→1,508,921，减少874,032像元；Phase84两套规则都选R4，总误分均2,119,235。所有错误流及未选对象均保留，定义变化不是自动的大面积错误证据。

## 结论探讨

A、C的非零损失仅针对预先固定“平均负类误报≤1%时宏召回最高”的代理目标，描述区间均跨零；不是独立确认或选择不确定性的完整区间。A的L2经验最大组误报低于L3；C的R误报低于C123，均存在合理风险/收益取舍。B是历史跨池预算失配，不能据此宣布mIoU或AP普遍错误、或已测得真实下游危害。D未出现被支配选择，不能用旧方案现象代替强基线证据。

本轮最多支持固定候选与已暴露支持上的目标依赖、跨池变化和定义敏感性。A-L2/L3、C-R/C123和B超预算是值得解释的具体历史案例；不自动提出新训练或新评价理论，旧分枝继续关闭。

## 核验、缺项和范围

输入先冻结43份小材料及代码SHA，再生成比较表；这是事后回放，不是盲法/前瞻预注册。原指标复现误差均≤1e-6；独立标准库脚本1838项检查及256组MET核验通过。相关组bootstrap10000次、seed201017只给已选对象条件描述区间；未做选模稳定性分析，不把规则/像元作为独立样本，最大误报只为经验最大值，不给联合置信保证。

[缺项](missing_inputs.csv)明确C三类输出缺失；池化AP全部复用历史原始池化统计，未用宏AP替代。[输入与代码锁](INPUT_CODE_LOCK.json)、[工程修复链](INPUT_CODE_LOCK_REPAIR.json)、[独立复核](INDEPENDENT_REVIEW.json)、[条件区间](conditional_intervals.csv)、[逐组差值](paired_group_differences.csv)、[交付SHA](SHA256SUMS.json)。两个工程问题均保留记录，未更换数据/规则。零训练、拟合、前向、下载、阈值搜索、人工判读；不修改旧文件、不push、不自动启动后续phase。
'''
    (O/'README.md').write_text(text,encoding='utf8')
    src=R/'src/research_results/phase201_20261017';src.mkdir(parents=True,exist_ok=True)
    (src/'README.md').write_text(text.replace('](TABLE','](D:/Cloud-Adapter-light/outputs/phase201/phase201_20261017/TABLE').replace('](missing_inputs.csv)','](D:/Cloud-Adapter-light/outputs/phase201/phase201_20261017/missing_inputs.csv)'),encoding='utf8')
    # Make all relative evidence links work from the source summary as well.
    source=(src/'README.md').read_text(encoding='utf8')
    import re
    source=re.sub(r'\]\((?!D:)([^)]+)\)',lambda m:'](D:/Cloud-Adapter-light/outputs/phase201/phase201_20261017/'+m[1]+')',source)
    (src/'README.md').write_text(source,encoding='utf8')
    total=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());assert total<4*1024**3
    write(O/'COMPLETION.json',dict(status='completed_with_explicit_missing_rule',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),blocks=4,selection_rules_executed=10,missing_rules=['C pooled three-class mIoU'],not_applicable=['D mIoU and budget loss'],selected_dominated=0,nonzero_loss_rules=['A pooled_AP','C macro_AP'],budget_violation_rules=['B pooled_miou','B pooled_AP','B macro_AP'],bootstrap_resamples=10000,bootstrap_seed=201017,independent_review=ind,full_process_guard=json.loads((O/'CPU_GUARD.json').read_text()) if (O/'CPU_GUARD.json').exists() else None,
        successful_analysis_seconds=a['resource']['wall_seconds'],tracemalloc_peak_bytes=a['resource']['tracemalloc_peak_bytes'],memory_measurement_scope='Python traced allocations, not full process RSS; all inputs are small matrices/tables, bootstrap max10000x64',output_bytes_before_final_receipts=total,training=0,fitting=0,model_calls=0,downloads=0,threshold_search=0,push=False,original_files_unchanged=True,no_followup=True))
    write(O/'FINAL_CODE_SHA256.json',{str(p.relative_to(R)).replace('\\','/'):sha(p) for p in (R/'src/tools').glob('*phase201*.py')})
    write(O/'SHA256SUMS.json',dict(files={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'},source_summary_sha256=sha(src/'README.md')))
    print('Phase201 delivered locally; no Git operations or followup phase.')
if __name__=='__main__':main()

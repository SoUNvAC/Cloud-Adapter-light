"""Publish separate-pool aggregate tables and all matched benchmark timings."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

R=Path(__file__).resolve().parents[2]
O=R/'src/research_results/phase119_20261015'
U=R/'outputs/phase119/phase119_20261015'


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,b):
    with p.open('xb') as f:f.write(b)
def write(p,obj):save(p,(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
def table(path,rows):
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def rows(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def number(v,n=3):return f'{float(v):.{n}f}' if v not in ('',None) else 'NA'


def main():
    assert (O/'ANALYSIS.md').exists() and (O/'NEXT_STEP.md').exists()
    for n,h in load(O/'EVALUATION_LOCK.json')['source_sha256'].items():assert sha(R/n)==h,n
    for pool in ('Catalogue',):
        for suffix in ('summary.csv','per_group.csv','comparisons.json'):
            name=pool+'_'+suffix;save(O/name,(U/'delivery/aggregate'/name).read_bytes())
    for pool in ('PixBox','Catalogue'):
        assert load(O/(pool+'_CPU_ACCEPTANCE.json') if pool=='PixBox' else U/'delivery/Catalogue_CPU_ACCEPTANCE.json')['status']=='passed'
    costs=load(U/'delivery/BENCHMARK_COSTS.json');job=load(U/'delivery/COST_SUMMARY.json')
    assert costs['all18_results_reported'] and len(costs['rows'])==18
    assert job['actual_model_calls']<=4000 and job['inference_wall_seconds']<=7200
    e={(r['product'],r['repetition']):r for r in costs['rows'] if r['policy']=='E'}
    benchmark=[]
    for r in costs['rows']:
        base=e[(r['product'],r['repetition'])]
        benchmark.append(dict(r,GPU_time_ratio_to_matched_E=r['GPU_model_seconds']/base['GPU_model_seconds'],
                              whole_time_ratio_to_matched_E=r['whole_scene_read_to_point_seconds']/base['whole_scene_read_to_point_seconds']))
    table(O/'benchmark_all18.csv',benchmark)
    resources=[]
    for m in costs['model_info']:
        s=job['workers'][m['policy']];assert s['status']=='completed_scores_sealed'
        resources.append(dict(m,CPU_RSS_peak_bytes=s['rss_peak_bytes'],GPU_reserved_peak_bytes=s['gpu_reserved_peak_bytes'],
                              GPU_observed_sampled_peak_bytes=s['gpu_observed_peak_bytes'],
                              actual_total_calls=s['all_calls'],main_calls=s['main_calls'],warmup_calls=s['warmup_calls']))
    table(O/'model_resources.csv',resources)
    totals={p:{'GPU_seconds':sum(r['GPU_model_seconds'] for r in benchmark if r['policy']==p),
               'whole_seconds':sum(r['whole_scene_read_to_point_seconds'] for r in benchmark if r['policy']==p)} for p in ('E','R','N')}
    for p in totals:
        totals[p]['GPU_ratio_to_E']=totals[p]['GPU_seconds']/totals['E']['GPU_seconds']
        totals[p]['whole_ratio_to_E']=totals[p]['whole_seconds']/totals['E']['whole_seconds']
    write(O/'COST_RATIO_SUMMARY.json',{'matched_six_scene_runs_per_policy':totals,'no_fastest_run_selection':True,
          'measurement_scope':'same three fixed PixBox scenes, both full repetitions; resource peaks are entire isolated worker',
          'OS_cache_and_order_not_eliminated':True})
    text=['# Phase119：Omni单模型精度／部署成本对照','','## 1．数据结果','',
          '本轮固定E＝原V1 RegNetY-004＋ConvNeXtV2-Nano集成；R＝仅RegNetY-004；N＝仅ConvNeXtV2-Nano。R与Phase117输入补丁R不是同一设置。本轮未加载任何输入补丁，零训练、拟合、新下载、校准或阈值搜索。均为原B04／B03／B8A、20m、patch1000／overlap300、batch1、float32、CPU拼接、compile=False和作者confidence规则。','']
    for pool in ('PixBox','Catalogue'):
        ss=rows(O/(pool+'_summary.csv'));pg=rows(O/(pool+'_per_group.csv'));cc=load(O/(pool+'_comparisons.json'))['rows']
        assert len(ss)==3
        for p in ('E','R','N'):
            row=next(s for s in ss if s['setting']==p)
            av=[float(g[p+'_AP']) for g in pg if g[p+'_AP']]
            assert abs(statistics.mean(av)*100-float(row['macro_AP_pp']))<1e-9
            assert sum(int(g[p+'_TP']) for g in pg)==int(row['pooled_TP'])
            assert sum(int(g[p+'_FP']) for g in pg)==int(row['pooled_FP'])
        text += [f'### {pool}','',f"原{ss[0]['all_groups']}相关组，实际{ss[0]['AP_groups']}组有正负类可计算AP；共同支持下参考云影{ss[0]['pooled_positive']}、非云影{ss[0]['pooled_negative']}。原无影组为NA，继续参与误报描述。",'',
                 '| 设置 | 逐组平均AP (%) | 池化AP (%) | 逐组平均召回 (%) | 逐组平均负类误报 (%) | 最坏组误报 (%) |',
                 '|---|---:|---:|---:|---:|---:|']
        for v in ss:text.append('| '+v['setting']+' | '+' | '.join(number(v[k]) for k in ('macro_AP_pp','pooled_AP_pp','macro_recall_pp','macro_negative_FP_pp','worst_group_negative_FP_pp'))+' |')
        text += ['', '| 固定比较 | 平均ΔAP (pp) | 组配对bootstrap95%区间 | 改善／稳定／下降组 |','|---|---:|---|---|']
        for v in cc:text.append(f"| {v['comparison']} | {number(v['mean_delta_AP_pp'])} | [{number(v['CI95_pp'][0])}, {number(v['CI95_pp'][1])}] | {v['improved_gt1pp']}/{v['stable_within1pp']}/{v['declined_lt_minus1pp']} |")
        text += ['', '| 设置 | 池化召回 (%) | 池化精确率 (%) | 池化负类误报 (%) | TP／FP |','|---|---:|---:|---:|---|']
        for v in ss:text.append(f"| {v['setting']} | {number(float(v['pooled_recall'])*100)} | {number(float(v['pooled_precision'])*100)} | {number(float(v['pooled_negative_FP_fraction'])*100)} | {v['pooled_TP']}/{v['pooled_FP']} |")
        text += ['',f'全部逐组AP、召回、精确率、误报、样本量与三项原始差值见{pool}_per_group.csv；稀疏支持保留标注，未补样或挑选场景。','']
    text += ['PixBox误报是所选有效负类点比例；Catalogue误报是密集发布参考下有效非云影像元比例。两池分别计算，不能混作同一总体或同一全景FPR。阈值固定0.5，没有扫描。Catalogue沿用Phase117原共同有效掩膜，包括既有观测／发布参考有效规则；PixBox没有额外套用Source六波段筛选。','',
             '相关组等权；PixBox同组产品先合并计算AP。每项差值固定10000次配对组bootstrap、seed119015，区间仅作描述。方向按>1pp／±1pp以内／<−1pp划分，未另设显著性或最小增益通过门，未将像元、产品或重复计时当新独立组。','',
             '### 部署成本：三景各两次完整新前向','',
             '基准按SHA256(PHASE119_BENCH|product_id)排序先锁定：'+', '.join(costs['fixed_scene_order'])+'。三个独立进程按E／R／N顺序；各自只加载所需模型，每个成员一次预热。E重新完整推理，缓存读取不充作推理时间。','',
             '| 设置 | 总参数 (M) | 推理可训练参数 | 权重 (MB) | 加载 (秒) | CPU RSS峰 (GiB) | GPU reserved峰 (GiB) | GPU观测峰 (GiB) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in resources:text.append(f"| {v['policy']} | {number(v['total_parameters']/1e6)} | 0 | {number(v['weight_bytes']/1e6)} | {number(v['load_seconds'])} | {number(v['CPU_RSS_peak_bytes']/1024**3)} | {number(v['GPU_reserved_peak_bytes']/1024**3)} | {number(v['GPU_observed_sampled_peak_bytes']/1024**3)} |")
    text += ['', '权重字节不含环境和缓存；加载时间另计。参数加载时的requires_grad数量在model_resources.csv中，推理时全部冻结。RSS和GPU reserved为整个独立worker峰值；GPU外部观测为定期采样，不冒充连续精确峰值。','',
             '| 设置 | 三景两遍GPU模型计算和 (秒) | 相对E | 读入至取点整景时间和 (秒) | 相对E |','|---|---:|---:|---:|---:|']
    for p,v in totals.items():text.append(f"| {p} | {number(v['GPU_seconds'])} | {number(v['GPU_ratio_to_E'])} | {number(v['whole_seconds'])} | {number(v['whole_ratio_to_E'])} |")
    text += ['', '| 设置 | 产品（日期／tile） | 第几遍 | GPU模型计算 (秒) | 整景读入至取点 (秒) | 整景相对匹配E |','|---|---|---:|---:|---:|']
    for v in benchmark:
        pid=v['product'];label=pid.split('_')[2][:8]+'/'+pid.split('_T')[1][:5]
        text.append(f"| {v['policy']} | {label} | {v['repetition']} | {number(v['GPU_model_seconds'])} | {number(v['whole_scene_read_to_point_seconds'])} | {number(v['whole_time_ratio_to_matched_E'])} |")
    text += ['', 'GPU计算用CUDA事件同步计量模型段，整景为同一读入／取点实现；SHA内容核验、额外独立CPU验收、压缩与模型加载均另计，正常作者归一化／CPU拼接仍在整景时间内。18次值全部列出，没有只报最快一次；进程顺序、系统缓存与共享硬件影响没有消除，不据此声称普适加速倍数。','',
             '## 2．结论探讨','',(O/'ANALYSIS.md').read_text(encoding='utf-8').strip(),'',
             '## 3．下一小步','',(O/'NEXT_STEP.md').read_text(encoding='utf-8').strip(),'',
             '## 4．验收与边界','',
             f"实际累计模型调用{job['actual_model_calls']}，推理任务保守wall {job['inference_wall_seconds']:.3f}秒；含构建、预热和所有完整重复，明细在COST_SUMMARY.json及各worker回执。宿主物理盘余量{job['host_physical_free_bytes']/1e9:.3f}GB，打包前新增中间文件{job['new_intermediate_bytes_before_packaging']}bytes。",'',
             'E三景旧分数复现、各设置整幅重复、独立CPU raw平均／拼接／取点confidence误差均≤1e−5，覆盖与gradient一致；Catalogue还用真实R／N原始logit重建E核对旧缓存，没有从集成confidence反推单模型。两池先封存全部分数和共同支持，再分别连接原参考，独立并列AP与标量组bootstrap验收通过，旧E基线重新计算复现。','',
             '本轮未评价153个fit组，未自动启动训练或下一实验。原影像、权重、参考与历史失败不改写；无work_dirs打包／同步／清理。Catalogue密集分数和raw logits留远端shared，PixBox逐点分数及原参考连接留本地受控data，不入Git。仅代码、小聚合结果与SHA按push／pull交付。','',
             '两池已经暴露，训练地点交集与参考语义限制保留；这是固定单成员消融和成本对照，不能单独宣称新算法贡献、独立外部确认、全景安全性或跨硬件加速保证。最迟2026-10-20北京时间停止线保持。','']
    report='\n'.join(text).encode();save(O/'README.md',report);save(U/'FINAL_RESULTS.md',report)
    write(O/'COMPLETION.json',{'phase':119,'status':'completed_exploratory_accuracy_cost_tradeoff',
          'two_pools_reported_separately':True,'all18_benchmarks_reported':True,'CPU_acceptance_passed':True,
          'actual_model_calls':job['actual_model_calls'],'no_new_training_fitting_download_or_threshold_search':True,
          'no_followup_started':True,'work_dirs_transferred':False})
    paths=sorted(p for p in O.rglob('*') if p.is_file())+sorted(R.glob('src/tools/*phase119*.py'))
    paths += [R/'src/research_plans/PHASE119_SINGLE_MODEL_EFFICIENCY_20261015.md',R/'src/research_plans/EXPLORATION_POLICY_20261014.md']
    write(O/'REPOSITORY_DELIVERY_SHA256.json',{p.relative_to(R).as_posix():sha(p) for p in paths})
    for p in O.iterdir():
        if p.is_file() and p.name!='.gitattributes' and not (U/p.name).exists():save(U/p.name,p.read_bytes())
    print({'status':'completed','actual_calls':job['actual_model_calls'],'repository_files':len(paths)})


if __name__=='__main__':main()

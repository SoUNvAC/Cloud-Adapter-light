"""Truthful three-initialization report; no manuscript rewrite or new experimental decisions."""
from datetime import datetime,timezone
from phase315_common import *
def main():
    assert load(ROOT/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed' and load(ROOT/'CONFIRMATION_ADMISSION_LOCK.json')['ready_independent_pools']==0
    metrics=read(ROOT/'SEED_METRICS.csv');rep=read(ROOT/'REPLICATION_SUMMARY.csv');directions=read(ROOT/'DIRECTION_COUNTS.csv');summary=read(ROOT/'SEED_METRIC_SUMMARY.csv');stages=load(ROOT/'STAGES.json');cpu=load(ROOT/'CPU_ACCEPTANCE.json');calls=load(ROOT/'BOTH_SEALED.json')['forwards'];identity=read(ROOT/'SEED_IDENTITY.csv')
    lines=['# Phase315 固定日程下独立LoRA初始化重复','','## 数据结果','','原Source、100fit/32development、Q90-PL日程及超参数保持；只新增31501/31502，30201只引用既有端点。A整体摘要3个均不同、初始B均0，294912可训练参数。每臂1000尝试/955更新/45跳过；逐产品/逐步maskSHA及原总覆盖75469158、三类[40622307,33554491,1292360]、累计裁剪181307540像元与[524,644,458]类出现步均复现。',
      '每个seed同初始状态的原304接纳步AST对照、零LoRA/Source输出、梯度与更新门均≤1e-6；门检恢复初始/梯度/优化器/RNG。旧30201按排序首dev重放9窗通过。Python/NumPy/torch/CUDA各用本seed，固定裁剪翻转来自原seed30001；冻结模块eval/dropout0。沿原TF32关闭、deterministic warn_only；保留CuBLAS非确定性警告，未改环境或人为加噪，不能宣称覆盖所有随机不确定性。',
      '', '|seed|组均mIoU|池化mIoU|组均增量pp|Cloud→Shadow净减|B真正Cloud纠正|Surface→Cloud净增|D原正确Surface损伤|宏Shadow召回增量pp|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rep:
        m=next(m for m in metrics if m['seed']==r['seed']);lines.append('|'+r['seed']+'|'+f"{float(m['group_mean_mIoU_pct']):.4f}|{float(m['pooled_mIoU_pct']):.4f}|{float(r['group_mIoU_delta_pp']):+.4f}|{r['Cloud_Shadow_net_reduction']}|{r['B_cloud_corrections']}|{r['Surface_Cloud_net_increase']}|{r['D_surface_damage']}|{float(r['macro_Shadow_recall_delta_pp']):+.4f}|" )
    lines+=['','Source为同一固定端点，未赋伪造seed标准差。三seed各指标算术均值/样本SD(ddof1)/min/max及Source差见SEED_METRIC_SUMMARY、REPLICATION_STATISTICS与SEED_SOURCE_DELTAS；全部32逐组/缺类union/AP/召回/FPR分母、六向错误和27格均保留。30201历史D0=506798/B0=784478精确复现；D/B各seed独立定义，交/并/仅旧/仅新/Jaccard及D0/B0去向只称初始化一致性，非SP/CM干预修复/保留。']
    totalgpu=sum(r['seconds'] for r in stages.values());lines.append(f'新正式前向2×1259=2518；实际总前向{calls}（含旧9窗/全部门检/构建），GPU管线{totalgpu:.2f}秒；CPU参考统计{cpu["seconds"]:.2f}秒/峰值{cpu["peak_cpu_bytes"]}字节。两臂训练及评价均封存后才读取主参考，未重建教师/新增数据下载/环境升级。探针16次/臂不选模，只保存初末LoRA。')
    lines+=['','确认池仅查既有来源与使用索引：PixBox16产品/10组是已暴露点参考，Kappa身份/几何/Shadow支持缺项，Catalogue旧切分不能按名称当未见，ALCD参考/分辨率与历史问题未闭合。合格独立三类确认池0个；搜索边界及“未发现≠证明未见”写入表，没读取旧sealed像元或新池表现。','','## 结论分析','']
    for r in directions:lines.append(f"{r['observation']}：{r['direction_count']}/3同向，效应范围[{r['effect_min']},{r['effect_max']}]；固定协议初始化重复的描述，不作3seed p值或均值正值显著性论证。")
    group=read(ROOT/'GROUP_SEED_DELTA_SUMMARY.csv');mixed=[r for r in group if r['metric']=='mIoU_pct' and int(r['positive']) and int(r['negative'])]
    lines.append(f'组均不能替代组间分布：32组中{len(mixed)}组mIoU增量出现跨seed正负分歧，完整均值/范围/正负零计数、历史D0前三组及边界/非边界/排除支持均附表。缺类组离散计分按原规则保留，不改变雪冰真值或Phase314双标口径。')
    consistent=all(int(r['direction_count'])==3 for r in directions)
    lines.append('三个预设方向均在3/3重复，支持本配置固定日程下的初始化重复性；收益及Surface损伤、Shadow召回代价必须一起报告。' if consistent else '预设方向并非全部3/3重复，按具体seed/指标及逐组表降级为初始化/配置敏感的个案；不追加seed或调参救回。')
    lines+=['独立确认准入当前仍为0，因此不启动新池评价。若未来补齐确切池身份/使用/空间/正Shadow支持并冻结端点，才可另审确认协议；本轮不自动触发该动作。PL重复不验证单seed SP/CM的跨seed结论，不能称独立目标域泛化；不改写论文中心结论、不重开其他分枝。']
    text='\n'.join(lines)+'\n';assert len(text)<4700;(ROOT/'RESULT.md').write_text(text,encoding='utf-8')
    (ROOT/'phase315.md').write_text(f'''# Phase315 实验记录

- 目标：仅Q90-PL两个独立LoRA初始化重复，加既有确认池准入清单；不复验SP/CM或新池。
- 改动：复用原304接纳步AST、固定1000步及全部教师/Source缓存；新seed31501/31502，各955更新45跳过；A摘要三seed不同、B零、冻结Source参数/buffer保持。
- 门检：旧30201末步9窗身份≤1e-6；各新seed零LoRA/Source和同初始AST loss/梯度/更新≤1e-6，门检后初始/优化器/梯度/RNG恢复。独立CPU复核全部32组/27格/6向错误和D/B/重叠边缘。
- 网络/环境：gzs cloud-lite-pt210 FP32单GPU串行；零下载/升级；本地代码push→gzs pull。nvidia-smi不在PATH，用现有torch显存/进程检查；保留原CuBLAS warn_only非确定性警告，不改变算子设置。
- 止损：总新增前向≤2800、GPU≤8h、CPU≤16GiB、新持久含双机与全部备份≤6GB、宿主空闲RAM/磁盘≥4GB；不盲重跑、不增seed/降精度/删组。
- 结果：实际{calls}前向、GPU管线{totalgpu:.3f}秒；三个预设方向次数见DIRECTION_COUNTS，数值范围/样本SD及逐组完整保留。原D0/B0精确复现；确认池0个具备独立三类准入条件，无新池推理。
- 局限：固定日程仅初始化重复；非96独立组、非所有随机性、非SP/CM跨seed或独立目标域确认。缺项交付，不自动改论文、排定时检查或开下一phase。
''',encoding='utf-8')
    write(ROOT/'DELIVERY_LOCK.json',dict(status='completed',utc=datetime.now(timezone.utc).isoformat(),code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase315*') if p.is_file()},report_sha256=sha(ROOT/'RESULT.md'),both_seed_barrier_sha256=sha(ROOT/'BOTH_SEALED.json'),no_new_pools_evaluated=True))
    write(ROOT/'STATUS.json',dict(status='completed',neural_forwards=calls,gpu_seconds=totalgpu,independent_confirmation_pools_ready=0));(ROOT/'EXIT_CODE').write_text('0');print(text)
if __name__=='__main__':main()

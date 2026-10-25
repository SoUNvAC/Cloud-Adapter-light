"""Compact factual report from sealed CPU tables, without changing numeric artifacts."""
from phase311_common import *

def report():
    rows=read(ROOT/'METRICS.csv');metrics={r['strategy']:r for r in rows}
    support=read(ROOT/'TRAINING_SUPPORT.csv')[0];activity=load(ROOT/'ACTIVITY_SUMMARY.json')
    damages=[r for r in read(ROOT/'DAMAGE_BENEFIT.csv') if r['product']=='ALL' and r['region']=='all']
    cm=metrics['Q90-CM'];pl=metrics['Q90-PL'];sp=metrics['Q90-SP'];f=lambda r,k:float(r[k])
    lines=['# 数据结果','',
        '固定Q90-CM，独立冻结Source同512语义得分，CE mean_A + mean_P relu[(sS_Cloud−sS_Surface)−(sT_Cloud−sT_Surface)]²，lambda=1；分母全部P。全新30201 LoRA294912参数，1000尝试955更新45跳过；Source基础参数/buffer和teacher逐张量不变。',
        'A=75469158，三类40622307/33554491/1292360；P唯一产品像元8201741、更新裁剪累计19937970。100产品/1000裁剪掩膜SHA复现，原45空A步骤不前向；初始CM≤1e−10、原304 AST lambda0 loss/梯度/更新≤1e−6通过。',
        '|策略|组均/池化mIoU %|三类IoU %|Shadow宏/池化AP %|宏召回 %|宏/最大FPR %|',
        '|---|---|---|---|---|---|']
    for r in rows:
        fmt=lambda k:f'{float(r[k]):.4f}'
        lines.append('|'+ '|'.join([r['strategy'],fmt('group_mean_mIoU_pct')+'/'+fmt('pooled_mIoU_pct'),'/'.join(fmt(k) for k in ('Surface_IoU_pct','Cloud_IoU_pct','Shadow_IoU_pct')),fmt('macro_AP_pct')+'/'+fmt('pooled_AP_pct'),fmt('macro_recall_pct'),fmt('macro_FPR_pct')+'/'+fmt('max_group_FPR_pct')])+'|')
    lines+=['','全部32组，原缺类union规则和参考支持保留，AP有效组数'+ '/'.join(r['AP_groups'] for r in rows)+'。Source/PL/SP缓存SHA复用，PL及SP各9窗身份重放通过；CM固定末步32×9窗。',
        '|策略|固定D→Surface/Cloud/Shadow|D外新Cloud/Shadow|固定B→Cloud/Shadow/Surface|',
        '|---|---|---|---|']
    for r in damages:
        lines.append('|'+r['strategy']+'|'+'/'.join(r[k] for k in ('D_to_Surface','D_still_Cloud','D_to_Shadow'))+'|'+'/'.join(r[k] for k in ('outside_D_new_Surface_to_Cloud','outside_D_new_Surface_to_Shadow'))+'|'+'/'.join(r[k] for k in ('benefit_retained_Cloud','benefit_back_Shadow','benefit_to_Surface'))+'|')
    lines+=['','D固定506798，B固定784478，不能重定义分母；集合外新增纠正另表。三种比较完整27格、六向错误、P/A/其余、全32组/全池、边界及固定前三组均附CSV。',
        f"约束实际激活：{activity['active_steps']}/955更新步骤；激活像元暴露{activity['active_pixel_exposures']}/{activity['P_updated_cumulative']}（{100*activity['active_fraction_of_all_updated_P']:.4f}%）。平均PL/CM/total={float(support['mean_PL']):.7g}/{float(support['mean_CM']):.7g}/{float(support['mean_total']):.7g}。固定1/101/500/900梯度审计见GRADIENT_AUDITS；跳过的活动量NA，未测活动数记录0不代表已测不激活。",'', '# 结论分析','',
        f"相对PL组均/池化mIoU变化{f(cm,'group_mean_mIoU_pct')-f(pl,'group_mean_mIoU_pct'):+.4f}/{f(cm,'pooled_mIoU_pct')-f(pl,'pooled_mIoU_pct'):+.4f}pp；相对SP为{f(cm,'group_mean_mIoU_pct')-f(sp,'group_mean_mIoU_pct'):+.4f}/{f(cm,'pooled_mIoU_pct')-f(sp,'pooled_mIoU_pct'):+.4f}pp。相对SP原B保留量及Shadow误报的方向需同时看，不用单一修复率宣布成功。"]
    d={r['strategy']:r for r in damages}
    lines.append(f"CM相对SP保留Cloud纠正变化{int(d['Q90-CM']['benefit_retained_Cloud'])-int(d['Q90-SP']['benefit_retained_Cloud']):+d}像元；宏/最大Shadow FPR变化{f(cm,'macro_FPR_pct')-f(sp,'macro_FPR_pct'):+.4f}/{f(cm,'max_group_FPR_pct')-f(sp,'max_group_FPR_pct'):+.4f}pp。P内真Cloud/Shadow仍纳入误保护代价；同512探针只用4产品×4匹配时点，不选模。")
    worst=read(ROOT/'LARGEST_DAMAGE_GROUP.csv')
    lines.append('最大历史损伤组T25XEG：'+'；'.join(r['strategy']+f" SurfaceIoU={float(r['Surface_IoU_pct']):.4f}%，mIoU={float(r['mIoU_pct']):.4f}%，union={r['union_classes']}，Surface→Cloud/Shadow={r['Surface_to_Cloud']}/{r['Surface_to_Shadow']}" for r in worst)+'。全D206076与P子集197082分开，SP33087 Shadow误报仅作历史参照；不得删单类组。')
    lines+=['直接Shadow输出偏导为0不代表共享参数或argmax不会改变Shadow。CM与KL同lambda=1不等于强度匹配；观察差异不能全归因于取消Shadow约束。',
        '训练不读目标参考/场景标签/DK包；封存后独立CPU分析。历史development已反复暴露，非新盲测；单seed不能证明稳定性或独立域收益。Phase309雪冰参考语义及逐景最终仿射限制仍在，场景标签不是像元真值。',
        '不追加lambda/seed/训练；若总体取舍仍失败且无新证据，本保护区换loss分枝到此记录，下一步需另提假设。本轮不证明云特有方法或可部署收益。']
    (ROOT/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

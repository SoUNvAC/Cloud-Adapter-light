"""Publish only SHA-verified cached/final diagnostics, without target truth."""
import csv
import json
from pathlib import Path
import shutil
from transfer_phase303 import R,LOCAL,load,sha,write

S=R/'src/research_results/phase303_20261024'


def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def main():
    final=(LOCAL/'BACKUP_FINAL.json').exists();kind='final' if final else 'cached'
    receipt=load(LOCAL/('BACKUP_'+kind.upper()+'.json'));assert receipt['all_archive_members_verified']
    source=LOCAL/('backup_'+kind)
    for n,v in load(source/('delivery_'+kind+'_FILES.json')).items():assert sha(source/n)==v['sha256'],n
    S.mkdir(parents=True,exist_ok=True)
    for p in source.iterdir():
        if p.is_file() and p.suffix in ('.json','.csv','.log','.jsonl'):
            shutil.copyfile(p,S/p.name)
    shutil.copyfile(LOCAL/('BACKUP_'+kind.upper()+'.json'),S/('BACKUP_'+kind.upper()+'.json'))
    cached=load(source/'CACHED_ACCEPTANCE.json');assert cached['status']=='passed'
    updates=read(source/'ORIGINAL_SIX_UPDATES.csv')
    lines=['# Phase303：置信度与有效训练支持诊断','','## 数据结果','',
        f"原定义0.90复现：有效{cached['valid']:,}、接受{cached['accepted']:,}（{100*cached['accepted']/cached['valid']:.6f}%），Surface/Cloud/Shadow={cached['classes']}。5产品有接受像元，1000尝试仅6步非空，实际有梯度的独立产品{cached['gradient_products']}个。",'',
        '|尝试步|产品|top/left/翻转|Surface/Cloud/Shadow接受|', '|---|---|---|---|']
    for r in updates:lines.append(f"|{r['attempt']}|{r['product']}|{r['top']}/{r['left']}/{r['horizontal_flip']}|{r['Surface']}/{r['Cloud']}/{r['Shadow']}|")
    if final:
        summary=read(source/'THRESHOLD_SUMMARY.csv')
        lines+=['','|定义|阈值|池化/产品平均覆盖%|Surface/Cloud/Shadow|非空产品/步骤|有梯度产品|各类出现步数|top1/top5%|',
            '|---|---|---|---|---|---|---|---|']
        for r in summary:
            lines.append('|'+ '|'.join([r['definition'],r['threshold'],f"{float(r['pooled_coverage_pct']):.3f}/{float(r['product_mean_coverage_pct']):.3f}",
                '/'.join(r[c] for c in ('Surface','Cloud','Shadow')),r['nonempty_products']+'/'+r['nonempty_steps'],r['gradient_products'],r['class_present_steps'],
                f"{100*float(r['top1_fraction']):.1f}/{100*float(r['top5_fraction']):.1f}" if r['top1_fraction']!='NA' else 'NA'])+'|')
        quant=read(source/'SCORE_QUANTILES.csv');lookup={(r['predicted_class'],r['quantity']):r for r in quant}
        lines+=['','|教师预测类|有效支持|sum(s) p50|A置信度 p50/p99|B置信度 p50/p99|','|---|---:|---:|---|---|']
        for cls in ('all','Surface','Cloud','Shadow'):
            a=lookup[cls,'confidence_A'];b=lookup[cls,'confidence_B'];total=lookup[cls,'sum_s']
            def fmt(v):return v if v=='NA' else f'{float(v):.6f}'
            lines.append(f"|{cls}|{a['support']}|{fmt(total['p50'])}|{fmt(a['p50'])}/{fmt(a['p99'])}|{fmt(b['p50'])}/{fmt(b['p99'])}|")
        rec=load(source/'RECOVERY_ACCEPTANCE.json');stages=load(source/'STAGES.json')
        candidates=[r['definition']+'@'+r['threshold'] for r in summary if all(int(r[c])>0 for c in ('Surface','Cloud','Shadow')) and all(v>0 for v in json.loads(r['class_present_steps']))]
        lines+=['',f"四个原scores CPU复核通过，96个缺失scores补算与首个既有scores九窗重建均通过≤1e-6容差；实际{rec['calls']}次前向、零训练/更新。四样例×0.5/1/2尺度检查通过；B尺度不变，预测固定argmax(s)。固定例s=[1,0,0]的A最大值约0.576。所有分位数采用全体有效像元精确计算，完整32行分布表另交。",'',
            '唯一像元数见accepted；重叠裁剪累计接受像元次数见cumulative_pixel_uses，两者不可混用。全部100产品、8×1000裁剪、集合交并、每类增减、零和/并列及12个尺度样例均另表保留。', '', '## 结论分析','',
            '原0.90仅留下Surface是原得分尺度与教师预测分布共同作用下的过滤结果，不是三类自训练。s为Mask2Former非负query聚合得分，softmax(s)受到绝对尺度影响，0.90不能解释为90%正确率；窗口拼接按原顺序重建并复现缓存，本轮未改变窗口协议。',
            '同时提供三类整图和原裁剪支持的固定候选：'+('、'.join(candidates) if candidates else '无')+'。这些只是训练支持候选，不证明伪标签质量，也不自动选择阈值。',
            '下一轮应先冻结置信度过滤规则和有效训练预算，再从原Source开始；不能把Phase302六次Surface-only更新作为已完成三类SFDA机制验证，也不直接据此进入有向约束。B仅为类别得分占比诊断，不是新方法或已校准概率，高占比仍可能是错误。', '',
            '本轮未读取目标真值、development影像/预测/指标或学生端点。历史研究已有开发暴露的事实保留，不声称整个研究未接触标签；不回写Phase302历史结果。']
        status='completed'
    else:
        product=read(source/'CACHED_PRODUCT_THRESHOLDS.csv');crop=read(source/'CACHED_CROP_COUNTS.csv')
        lines+=['','|原A阈值|唯一接受像元|非空产品|非空步骤|裁剪累计接受次数|','|---|---:|---:|---:|---:|']
        for threshold in ('.5','.7','.8','.9'):
            p=[r for r in product if float(r['threshold'])==float(threshold)];c=[r for r in crop if float(r['threshold'])==float(threshold)]
            lines.append(f"|{threshold}|{sum(int(r['accepted']) for r in p)}|{sum(int(r['accepted'])>0 for r in p)}|{sum(int(r['accepted'])>0 for r in c)}|{sum(int(r['accepted']) for r in c)}|")
        lines+=['','## 结论分析','',
            'CPU缓存复核已完成，GPU因其他训练任务占用而等待。低阈值总覆盖可从maximum计算；96产品缺少完整s和低阈值类别，不能把过滤后的255充当预测类别。B及完整类别分布待原Source补算后再报告。没有新增前向、训练、标签读取或学生评价。']
        status='cached_complete_waiting_gpu'
    (S/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(S/'DELIVERY_STATUS.json',dict(status=status,actual_calls=load(source/'RECOVERY_ACCEPTANCE.json')['calls'] if final else 0,
        cached_reproduced=True,truth_access=False,training_updates=0,backup_sha256=receipt['sha256']))
    note=['# Phase303 实验记录','',
        '- 目标：诊断Phase302伪标签尺度、类别偏置与真实训练支持；固定两定义四阈值，不选最优阈值，不训练或评价学生。',
        '- 改动：src/tools建立缓存复核、严格Source补算、8×1000裁剪重放、精确分位数、尺度/集合验收、资源护栏、GPU冲突等待、全量SHA备份及报告脚本；保留Phase302原文件。',
        '- 网络：本地修改后push、gzs指定仓库pull；一次git SSH临时断连，重试成功。首次前向前仅补GPU内存遥测，保留初始代码锁与修订证据；统计口径未改变。',
        '- 环境：所有计算在gzs的cloud-lite-pt210；Source SHA固定64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9，RGB/FP32/9窗与Phase302一致；没有源样本、目标标签、development、PixBox或L8访问。',
        '- 止损线：新增前向≤900，GPU≤1小时，CPU≤16GiB且累计≤1小时，新持久≤3GB，宿主空闲内存/磁盘≥4GB；检测到其他训练控制器时等待，不抢GPU。',
        f"- 缓存实测：有效{cached['valid']}、接受{cached['accepted']}、类别{cached['classes']}、5非空产品、6更新/994跳过、实际有梯度产品{cached['gradient_products']}；不将重复裁剪当新产品。", 
        f"- 备份：{kind}快照{receipt['archived_files']}成员，{receipt['bytes']}字节，SHA256={receipt['sha256']}，全部成员已回传本地并逐项核验。",
        '- 当前状态：'+status+'；详见RESULT.md。']
    if final:
        note.append('- 资源：'+json.dumps(load(source/'STAGES.json'),ensure_ascii=False)+'；GPU记账='+json.dumps(load(source/'RECOVERY_ACCEPTANCE.json'),ensure_ascii=False))
        note.append('- 保留4个完整scores样例；96份临时重建数组在统计和精确分位数完成后按已核验SHA清理，清单TRANSIENT_SCORE_CLEANUP.json保留全部身份。全部失败记录照实保留；不回写Phase302，不启动下一phase。')
    (S/'phase303.md').write_text('\n'.join(note)+'\n',encoding='utf-8')
    write(S/'SUMMARY_SHA256.json',dict(status=status,files={p.name:sha(p) for p in S.iterdir() if p.is_file() and p.name!='SUMMARY_SHA256.json'},
        code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase303*.py')}))
    print(json.dumps(load(S/'DELIVERY_STATUS.json')))


if __name__=='__main__':main()

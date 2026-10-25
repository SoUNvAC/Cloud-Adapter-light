"""Remote editorial report from frozen result tables; no new model/reference reads."""
import json
import resource
import shutil
import time
from datetime import datetime,timezone
from phase314_common import *
def number(v):return None if v=='NA' else float(v)
def fmt(v):return 'NA' if v is None else f'{v:.4f}'
def main():
    started=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    state=load(ROOT/'STATUS.json');assert state['status']=='audited_pending_report';assert load(ROOT/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed'
    manifest=read(ROOT/'DUAL_LABEL_MANIFEST.csv');difference=read(ROOT/'DELTA_B_MINUS_A.csv');enrich=read(ROOT/'CHANGE_ENRICHMENT.csv');decomp=read(ROOT/'CHANGE_DECOMPOSITION.csv');pseudo=read(ROOT/'PSEUDO_LABEL_SENSITIVITY.csv');strata=read(ROOT/'ANNOTATOR_DISAGREEMENT.csv');deltas=read(ROOT/'DUAL_REFERENCE_DELTAS.csv')
    products=load(ROOT/'CPU_ACCEPTANCE.json')['development_products'];names={p:p.split('_')[5] for p in products}
    lines=['# Phase314 双标参考下的适配收益敏感性','','## 数据结果','','固定原Source与Phase304 Q90-PL；60产品双标（CALIBRATION10、VALIDATION50）只做参考分歧普查。与原100fit交12、32dev交4；4dev均VALIDATION，12fit不计模型测试样本。main作者由classification_tags恢复，60景中30景main为B；main/alt是作者A/B的重排，不是4份参考。严格bool/1022²×3/one-hot，Jmask及各方独有支持、不合法类型逐景保留。']
    for ds in ['CALIBRATION','VALIDATION']:
        row=next(r for r in strata if r['row_type']=='stratum' and r['dataset']==ds and r['main_annotator']=='all' and r['shadows_marked']=='all');lines.append(f"{ds}：{row['products']}产品/{row['groups']}相关组；共同支持{row['support']}，组均分歧{float(row['group_mean_disagreement_pct']):.4f}%，池化分歧{float(row['disagreement_pct']):.4f}%。两池不合成独立验证。")
    lines+=['','4景原主参考Source/PL混淆8/8精确复现。随后固定J=双参考有效∩原共同评价支持∩image_valid，模型valid完整覆盖，额外排除和原因逐景记录。下表为同J组mIoU的PL−Source（pp），非新选主终点：','','|产品|作者A delta|作者B delta|B−A|方向相同|D占J|变化落D|','|---|---:|---:|---:|---|---:|---:|']
    for p in products:
        row=next(r for r in difference if r['product']==p and r['metric']=='mIoU_pct');er=next(r for r in enrich if r['product']==p);lines.append(f"|{names[p]}|{fmt(number(row['delta_A']))}|{fmt(number(row['delta_B']))}|{fmt(number(row['delta_B_minus_delta_A']))}|{row['direction_same']}|{100*float(er['D_fraction_J']):.3f}%|{fmt(None if er['D_fraction_changes']=='NA' else 100*float(er['D_fraction_changes']))}%|")
    lines+=['','三类IoU、准确率、Shadow AP/召回、六向错误及所有delta/B−A均完整列出，缺类union/AP/召回空分母为NA；E/D纠错、损伤、净变化和完整324个81格单元均已独立核对。池化与组均仅附属，未做像元重采样、4组p值或性能门槛。']
    psummary=[]
    for dataset in ['CALIBRATION','VALIDATION']:
        ids={r['product'] for r in manifest if r['range_C']=='True' and r['dataset']==dataset}
        for ref in ['A','B']:
            for c in ['Surface','Cloud','Shadow']:
                rr=[r for r in pseudo if r['product'] in ids and r['reference']==ref and r['zone']=='all' and r['region']=='accepted' and r['category']==c];den=sum(int(r['predicted']) for r in rr);correct=sum(int(r['correct']) for r in rr);truthden=sum(int(r['truth_support_all_J']) for r in rr);truthaccepted=sum(int(r['truth_support']) for r in rr)
                psummary.append(dict(dataset=dataset,products=len(ids),reference=ref,category=c,predicted=den,correct=correct,precision=correct/den if den else None,truth_all_J=truthden,truth_accepted=truthaccepted,truth_acceptance=truthaccepted/truthden if truthden else None))
    table(ROOT/'PSEUDO_ACCEPTED_POOLED.csv',psummary)
    lines+=['','12fit仅审计既有Source/Q90接受集合，接受/拒绝/全部×E/D/全体×A/B全表保留；q=s/sum(s)、阈值0.90、原覆盖/955更新均不改。按原dataset分层的接受类precision（池化，A/B同预测分母）如下：','','|dataset/产品数|类别|A precision|B precision|','|---|---|---:|---:|']
    for dataset in ['CALIBRATION','VALIDATION']:
        for c in ['Surface','Cloud','Shadow']:
            ar=next(r for r in psummary if r['dataset']==dataset and r['reference']=='A' and r['category']==c);br=next(r for r in psummary if r['dataset']==dataset and r['reference']=='B' and r['category']==c)
            lines.append(f"|{dataset}/{ar['products']}|{c}|{fmt(None if ar['precision'] is None else 100*ar['precision'])}%|{fmt(None if br['precision'] is None else 100*br['precision'])}%|")
    mi=[r for r in difference if r['product'] in products and r['metric']=='mIoU_pct'];reversals=sum(number(r['delta_A'])*number(r['delta_B'])<0 for r in mi if number(r['delta_A']) is not None and number(r['delta_B']) is not None)
    # Describe all six directions, never select a favorable event.
    eventdiff=[r for r in difference if r['product']=='POOLED4' and r['metric'].endswith('_errors')]
    cs=sum(abs(number(r['delta_B_minus_delta_A'])) for r in eventdiff if 'Surface' not in r['metric']);surface=sum(abs(number(r['delta_B_minus_delta_A'])) for r in eventdiff if 'Surface' in r['metric'])
    enriched=sum(float(r['D_fraction_changes'])>float(r['D_fraction_J']) for r in enrich if r['D_fraction_changes']!='NA')
    lines+=['','## 结论分析','',f'4个事后development个案中，mIoU增量严格正负反向{reversals}/4；逐组方向及全部指标差异以表为准，NA不补0。两参考的池化错误增量差B−A：'+ '；'.join(r['metric'].replace('_errors','')+'='+fmt(number(r['delta_B_minus_delta_A'])) for r in eventdiff)+'。正值只表示该错误增量在B参考下更大，不判断谁标错。',
      f'上述六向错误增量参考差的绝对值和：Cloud↔Shadow两项{cs:.0f}，Surface相关四项{surface:.0f}，只作事件量分解，不是独立样本或因果归因。{enriched}/4景的变化落D比例高于D占J比例；这仅是富集描述，边界/难度/类别构成均可能混杂。E上两参考恒等是代数事实，不是独立金标准；D的A→B/B→A及第三类转移全列，不能据此称模型学到作者风格。',
      '范围A的50VALIDATION双标互不咨询但共享工具/定义，10CALIBRATION经过协商；均不是独立物理真值。shadows_marked=0表示不能标而非无阴影，也不能证明备用作者完整性。4组只是历史暴露development子集，不能推断全32、源/目标标注口径偏移或谁更正确；12fit不能补成16测试组。',
      '唯一可否证后续假设：在事先固定且独立、覆盖类别/边界/场景的双标评价中，固定适配变化的参考敏感性是否仍集中于本轮出现的事件类型。需要独立产品与同支持的盲双标设计，冻结模型/终点后完整报告，不能用本轮4组来选有利作者或阈值。本轮不要求重新标注，不执行扩评、推理或训练。',
      '主参考、Source/PL端点、历史955更新均保持；未取标签并/交集重造监督，未一致性加权伪标签，未重开几何/光谱分枝。全部统计在gzs cloud-lite-pt210，零训练/拟合/前向/下载；本机只编辑、文件SHA和全量备份。资源、支持、NA和SHA见独立验收/输入锁；无定时任务或自动下一phase。']
    text='\n'.join(lines)+'\n';assert len(text)<6200;(ROOT/'RESULT.md').write_text(text,encoding='utf-8')
    record=f'''# Phase314 实验记录

- 目标：同一冻结适配变化对双标参考的敏感性，严格区分60参考普查、4dev模型个案、12fit教师质量。
- 改动：ALT_精确成员/主作者恢复、共同支持和原混淆复现、81格与E/D分解、Q90接受/拒绝质量；无模型变更。
- 环境/网络：gzs cloud-lite-pt210 CPU，零训练/拟合/前向/下载。本地编辑后push、gzs pull；一次Git SSH传输前连接关闭，同提交重试成功。README使用309现成文本，不重读PDF。
- 止损：CPU8GiB/1h，宿主空闲RAM/磁盘≥4GB，新持久含双机备份≤500MB；不重试科学失败、不补缓存推理。
- 结果：60=CAL10+VAL50，4dev均VAL；12fit分层见清单。main为B30景；原8混淆精确复现，324个81格及E/D/伪标签边缘独立通过。4个案mIoU增量严格反向{reversals}/4；不外推全32或域间口径偏移。
- 资源：统计{state['seconds']:.6f}秒，独立审计{state['audit_seconds']:.6f}秒，峰值{state['peak_cpu_bytes']}字节、最低宿主可用{state['minimum_available_memory']}字节；命名执行阶段计时不含本地编辑/SSH等待。
- 交付：5类主表、完整9/81格、所有指标delta/B−A、参考/输入/代码/缓存SHA、独立验收、完整备份。主参考不替换，无定时任务，不开始下一phase。
'''
    (ROOT/'phase314.md').write_text(record,encoding='utf-8')
    mem={k:int(v.split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())};assert mem['MemAvailable']>=4000000000 and shutil.disk_usage(ROOT).free>=4000000000
    elapsed=time.monotonic()-started;state.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat(),report_seconds=elapsed,total_seconds=state['total_seconds']+elapsed,peak_cpu_bytes=max(state['peak_cpu_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),minimum_available_memory=min(state['minimum_available_memory'],mem['MemAvailable']));assert state['total_seconds']<=3600;write(ROOT/'STATUS.json',state)
    write(ROOT/'DELIVERY_LOCK.json',dict(utc=state['completed_utc'],report_sha256=sha(ROOT/'RESULT.md'),code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase314*') if p.is_file()},from_frozen_tables_only=True,no_new_model_or_reference_reads=True))
    print(text)
if __name__=='__main__':main()

"""Clarify empty-D and undefined denominators; preserve initial editorial version and all data."""
from datetime import datetime,timezone
import resource
import shutil
import time
from phase314_common import *
def main():
    start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    assert not (ROOT/'EDITORIAL_CLARIFICATION.json').exists() and not (ROOT/'BACKUP_FILES.json').exists()
    state=load(ROOT/'STATUS.json');assert state['status']=='completed'
    e=read(ROOT/'CHANGE_ENRICHMENT.csv');assert len(e)==4 and all(int(r['D'])==0 for r in e)
    valid=[r for r in e if r['D_fraction_changes']!='NA'];assert len(valid)==2 and all(int(r['prediction_changes'])==0 for r in e if r not in valid)
    csvs={p.name:sha(p) for p in ROOT.glob('*.csv')}
    shutil.copyfile(ROOT/'RESULT.md',ROOT/'RESULT_INITIAL.md');shutil.copyfile(ROOT/'DELIVERY_LOCK.json',ROOT/'DELIVERY_LOCK_INITIAL.json')
    text=(ROOT/'RESULT.md').read_text()
    text=text.replace('4个事后development个案中，mIoU增量严格正负反向0/4；逐组方向及全部指标差异以表为准，NA不补0。','4景J上的A/B标签逐像元完全相同，D=0，所以所有收益delta相同是同参考支持下的必然结果。本子集没有检验标注分歧敏感性的机会；0/4反向不能作为跨参考稳健性证据，也不支持明显收益敏感性。')
    text=text.replace('0/4景的变化落D比例高于D占J比例；这仅是富集描述，边界/难度/类别构成均可能混杂。','只有2景发生预测变化，其变化落D比例均0；另外2景无变化，比例为NA，不能补0或计为反例。本轮没有非空D上的模型变化可供检验，不能宣称分歧区富集。')
    text=text.replace('唯一可否证后续假设：在事先固定且独立、覆盖类别/边界/场景的双标评价中，固定适配变化的参考敏感性是否仍集中于本轮出现的事件类型。需要独立产品与同支持的盲双标设计，冻结模型/终点后完整报告，不能用本轮4组来选有利作者或阈值。本轮不要求重新标注，不执行扩评、推理或训练。','12fit中，Q90接受Shadow的池化precision为A74.5393%/B73.8494%，Surface为98.6538%/99.1208%，Cloud为99.9229%/99.9686%；这说明事后教师质量读数对参考敏感，不是学生适配收益证据。12景均VALIDATION，CALIBRATION支持0保持NA。\n唯一后续可否证假设：在独立固定双标产品、非空分歧区且冻结模型确有预测变化的支持上，适配mIoU增量在A/B下仍一致。需独立于本轮4组的评价设计，事前固定清单和完整终点，不能按效果/分歧率挑有利组或作者。本轮只提可检验条件，不要求重新标注或执行扩评。')
    metrics=read(ROOT/'DUAL_REFERENCE_METRICS.csv');ap={}
    for ref in ['A','B']:
        for arm in ['Source','PL']:
            rr=[r for r in metrics if r['support_kind']=='J' and r['product']!='POOLED4' and r['reference']==ref and r['strategy']==arm]
            ap[ref+'_'+arm]=dict(groups=len(rr),AP_defined=sum(r['Shadow_AP_pct']!='NA' for r in rr),recall_defined=sum(r['Shadow_recall_pct']!='NA' for r in rr),positive_groups=sum(int(r['Shadow_positive'])>0 for r in rr))
    text=text.replace('三类IoU、准确率、Shadow AP/召回、六向错误及所有delta/B−A均完整列出，','Shadow可算性：'+ '；'.join(k+f" AP{v['AP_defined']}/4、召回{v['recall_defined']}/4" for k,v in ap.items())+'。三类IoU、准确率、Shadow AP/召回、六向错误及所有delta/B−A均完整列出，')
    (ROOT/'RESULT.md').write_text(text,encoding='utf-8')
    with (ROOT/'phase314.md').open('a',encoding='utf-8') as f:f.write('\n- 关键限定：4dev共同J的A/B标签完全一致，D=0，故无法检验分歧导致的适配收益敏感性；2景无预测变化，变化落D比例为NA。12fit教师质量有参考差异，不充当学生收益证据。初版正文和锁保留，CSV/统计/原输入不改。\n')
    assert csvs=={p.name:sha(p) for p in ROOT.glob('*.csv')}
    elapsed=time.monotonic()-start;mem={k:int(v.split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())};assert mem['MemAvailable']>=4000000000 and shutil.disk_usage(ROOT).free>=4000000000
    state.update(completed_utc=datetime.now(timezone.utc).isoformat(),clarification_seconds=elapsed,total_seconds=state['total_seconds']+elapsed);assert state['total_seconds']<=3600;write(ROOT/'STATUS.json',state)
    note=dict(utc=state['completed_utc'],code_sha256=sha(Path(__file__)),reason='all4Dempty; only2defined change fractions; distinguish C12teacher sensitivity',data_CSV_unchanged=csvs,no_reference_or_model_reread=True,no_scientific_rerun=True,Shadow_calculability=ap)
    write(ROOT/'EDITORIAL_CLARIFICATION.json',note);write(ROOT/'DELIVERY_LOCK.json',dict(utc=state['completed_utc'],report_sha256=sha(ROOT/'RESULT.md'),initial_lock_sha256=sha(ROOT/'DELIVERY_LOCK_INITIAL.json'),code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase314*') if p.is_file()},from_frozen_tables_only=True,no_new_model_or_reference_reads=True))
    print(text)
if __name__=='__main__':main()

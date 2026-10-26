"""Complete requested display summaries from sealed tables; preserve endpoint rows and original report."""
from datetime import datetime,timezone
import math
import resource
import shutil
import statistics
import time
from phase315_common import *
def main():
    started=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    assert load(ROOT/'STATUS.json')['status']=='completed' and not (ROOT/'PUBLICATION_COMPLETION.json').exists() and not (ROOT/'BACKUP_FILES.json').exists()
    original={p.name:sha(p) for p in ROOT.glob('*.csv')};shutil.copyfile(ROOT/'SEED_METRICS.csv',ROOT/'SEED_METRICS_ENDPOINTS.csv');endpoints=read(ROOT/'SEED_METRICS_ENDPOINTS.csv');assert len(endpoints)==4
    summaries={r['metric']:r for r in read(ROOT/'SEED_METRIC_SUMMARY.csv')};extra=[]
    for label,field in [('mean3','mean'),('sample_SD3_ddof1','sample_SD_ddof1'),('min3','min'),('max3','max')]:
        row={k:None for k in endpoints[0]};row.update(seed=label,groups=32)
        for k,s in summaries.items():row[k]=s[field]
        for k in ['AP_groups','recall_groups','FPR_groups']:
            assert len({r[k] for r in endpoints[1:]})==1;row[k]=endpoints[1][k]
        extra.append(row)
    table(ROOT/'SEED_METRICS.csv',endpoints+extra);assert read(ROOT/'SEED_METRICS.csv')[:4]==endpoints
    flow=read(ROOT/'ERROR_FLOW.csv');stats=[]
    for error in sorted({r['error'] for r in flow}):
        rows=[r for r in flow if r['product']=='POOLED32' and r['error']==error];assert len(rows)==3
        for field in ['student_errors','removed','added','net_reduction','corrected','changed_to_other_error','from_correct','rate_change_pp']:
            values=[float(r[field]) for r in rows if r[field]!='NA'];stats.append(dict(error=error,field=field,seeds=len(values),Source_errors_fixed=rows[0]['Source_errors'],truth_support_fixed=rows[0]['truth_support'],mean=statistics.mean(values) if values else None,sample_SD_ddof1=statistics.stdev(values) if len(values)>1 else None,min=min(values) if values else None,max=max(values) if values else None,Source_SD=None))
    table(ROOT/'ERROR_FLOW_STATISTICS.csv',stats)
    per=read(ROOT/'PER_GROUP_METRICS.csv');lookup={(r['seed'],r['product']):r for r in per};groups=[r['product'] for r in per if r['seed']=='Source'];signs=[];conflicts=[]
    for seed in ['30201','31501','31502']:
        values=[float(lookup[seed,p]['mIoU_pct'])-float(lookup['Source',p]['mIoU_pct']) for p in groups];signs.append(dict(seed=seed,groups=32,positive=sum(v>0 for v in values),negative=sum(v<0 for v in values),zero=sum(v==0 for v in values),min_delta_pp=min(values),max_delta_pp=max(values),threshold='exact sign; no new performance threshold'))
    for r in read(ROOT/'GROUP_SEED_DELTA_SUMMARY.csv'):
        if int(r['positive']) and int(r['negative']):
            for seed in ['30201','31501','31502']:
                y=lookup[seed,r['product']][r['metric']];x=lookup['Source',r['product']][r['metric']];v=None if 'NA' in [x,y] else float(y)-float(x);conflicts.append(dict(product=r['product'],group_id=r['group_id'],metric=r['metric'],seed=seed,delta_pp=v,sign=None if v is None else 'positive' if v>0 else 'negative' if v<0 else 'zero'))
    table(ROOT/'GROUP_SIGNS_BY_SEED.csv',signs)
    if conflicts:table(ROOT/'GROUP_DIRECTION_CONFLICTS.csv',conflicts)
    else:
        import csv
        with (ROOT/'GROUP_DIRECTION_CONFLICTS.csv').open('w',newline='',encoding='utf-8-sig') as f:csv.writer(f).writerow(['product','group_id','metric','seed','delta_pp','sign'])
    shutil.copyfile(ROOT/'RESULT.md',ROOT/'RESULT_PRE_TABLE_COMPLETION.md');shutil.copyfile(ROOT/'DELIVERY_LOCK.json',ROOT/'DELIVERY_LOCK_PRE_TABLE_COMPLETION.json')
    text=(ROOT/'RESULT.md').read_text();needle='Source为同一固定端点，未赋伪造seed标准差。';text=text.replace(needle,needle+'SEED_METRICS含Source/3端点及均值、样本SD、min/max行；统计行不伪造混淆，固定可算组数是支持说明。原4端点表另存SEED_METRICS_ENDPOINTS，完整六向流摘要见ERROR_FLOW_STATISTICS。')
    text=text.replace('组均不能替代组间分布：','各seed改善/损伤/不变组数：'+'；'.join(r['seed']+f"={r['positive']}/{r['negative']}/{r['zero']}" for r in signs)+'。逐组/指标的方向分歧完整列入GROUP_DIRECTION_CONFLICTS；标签边界不称物理薄云边缘。组均不能替代组间分布：')
    (ROOT/'RESULT.md').write_text(text,encoding='utf-8')
    for n,h in original.items():
        if n!='SEED_METRICS.csv':assert sha(ROOT/n)==h,n
    assert sha(ROOT/'SEED_METRICS_ENDPOINTS.csv')==original['SEED_METRICS.csv']
    for p in (R/'src/tools').glob('*phase315*.py'):
        target=ROOT/'code'/p.name
        if target.exists():assert sha(target)==sha(p),p.name
        else:shutil.copyfile(p,target)
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024;assert available>=4000000000 and shutil.disk_usage(ROOT).free>=4000000000
    note=dict(utc=datetime.now(timezone.utc).isoformat(),code_sha256=sha(Path(__file__)),original_csv_SHA256=original,endpoint_table_retained_exact=True,other_original_CSV_unchanged=True,no_model_or_reference_rereads=True,no_new_forwards_or_training=True,group_direction_conflict_rows=len(conflicts),seconds=time.monotonic()-started,peak_cpu_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,host_available_end=available)
    write(ROOT/'PUBLICATION_COMPLETION.json',note);write(ROOT/'DELIVERY_LOCK.json',dict(status='completed',utc=note['utc'],code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase315*') if p.is_file()},report_sha256=sha(ROOT/'RESULT.md'),original_delivery_lock_sha256=sha(ROOT/'DELIVERY_LOCK_PRE_TABLE_COMPLETION.json'),no_new_pools_evaluated=True))
    with (ROOT/'phase315.md').open('a',encoding='utf-8') as f:f.write('\n- 表格收口：SEED_METRICS补齐均值/样本SD/min/max显示行，原4端点字节保留；六向流全部字段的seed摘要及逐组方向分歧完整追加。其他原CSV哈希不变，未重新训练/前向/读取参考。状态查询SSH有握手关闭，独立控制器未受影响，未重启实验。\n')
    print(json.dumps(signs));print('Requested tables complete; raw endpoint/statistical data preserved')
if __name__=='__main__':main()

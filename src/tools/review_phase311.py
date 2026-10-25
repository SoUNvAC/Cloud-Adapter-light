"""Post-seal independent CSV/accounting review on gzs; preserve generated report."""
import collections
import math
import resource
import shutil
import time
from phase311_common import *

def main():
    started=time.monotonic()
    assert R==Path('/home/scv/Cloud-Adapter-light') and load(ROOT/'STATUS.json')['status']=='completed'
    assert not (ROOT/'REVIEW_ACCEPTANCE.json').exists(),'Preserve completed review'
    end=load(ROOT/'ENDPOINT_LOCK.json');assert end['status']=='sealed' and time.time()>end['sealed_epoch']
    execution=load(ROOT/'EXECUTION_LOCK.json')
    for n,h in execution['code'].items():assert sha(R/n)==h,n
    for n,h in execution['inputs'].items():assert sha(n)==h,n
    assert sha(ROOT/'final_lora.pth')==end['final_sha256'] and sha(ROOT/'TRAIN_CURVE.jsonl')==end['curve_sha256']
    for n,h in load(ROOT/'ANALYSIS_INPUT_SHA256.json').items():assert sha(n)==h,n
    matrices={(r['product'],r['strategy']):json.loads(r['confusion']) for r in read(ROOT/'PER_GROUP_METRICS.csv')}
    matrices.update({('ALL',r['strategy']):json.loads(r['confusion']) for r in read(ROOT/'METRICS.csv')})
    cells={}
    for r in read(ROOT/'TRANSITIONS27.csv'):
        key=(r['product'],r['region'],r['comparison']);index=(C.index(r['truth']),C.index(r['Source_pred']),C.index(r['student_pred']))
        bucket=cells.setdefault(key,{})
        assert index not in bucket;bucket[index]=int(r['count'])
    assert len(cells)==33*4*3 and all(len(t)==27 for t in cells.values())
    for (pid,region,cmp),t in cells.items():
        left,right=cmp.split('->')
        if region=='all':
            assert [[sum(t[c,i,j] for j in range(3)) for i in range(3)] for c in range(3)]==matrices[pid,left]
            assert [[sum(t[c,i,j] for i in range(3)) for j in range(3)] for c in range(3)]==matrices[pid,right]
        for idx,n in t.items():
            if region=='all':assert n==sum(cells[pid,reg,cmp][idx] for reg in ('P','A','other'))
            if pid=='ALL':assert n==sum(v[idx] for (p,reg,comp),v in cells.items() if p!='ALL' and reg==region and comp==cmp)
    for r in read(ROOT/'ERROR_FLOW.csv'):
        t=cells[r['product'],r['region'],r['comparison']];truth,dest=r['error'].split('->');c=C.index(truth);j=C.index(dest)
        old=sum(t[c,j,k] for k in range(3));new=sum(t[c,i,j] for i in range(3));same=t[c,j,j];support=sum(v for (y,i,k),v in t.items() if y==c)
        checks=dict(truth_support=support,Source_errors=old,student_errors=new,removed=old-same,added=new-same,net_reduction=old-new,corrected=t[c,j,c],changed_to_other_error=old-same-t[c,j,c],from_correct=t[c,c,j])
        assert all(int(r[k])==v for k,v in checks.items()),r
        assert r['rate_change_pp']=='NA' if not support else abs(float(r['rate_change_pp'])-100*(new-old)/support)<1e-10
    curve=[json.loads(x) for x in (ROOT/'TRAIN_CURVE.jsonl').read_text().splitlines()]
    assert len(curve)==1000 and [r['attempt'] for r in curve]==list(range(1,1001))
    crops={int(r['attempt']):r for r in read(ROOT/'CROP_SUPPORT.csv')}
    for r in curve:
        crop=crops[r['attempt']]
        assert all(r[k]==int(crop[k]) for k in ('A','P','Surface','Cloud','Shadow'))
        assert r['P_sha256']==crop['P_sha256'] and r['A_sha256']==crop['A_sha256']
        assert r['teacher_calls']==r['student_calls']==int(r['updated']) and r['updated']==bool(r['A']) and r['lambda_CM']==1 and r['lr']==1e-4
        if not r['updated']:
            assert all(r[k] is None for k in ('PL','CM','total','lora_grad_norm','active_fraction','d_mean','d_max','mT_mean','mS_mean'))
            assert r['active_count']==0
        else:
            assert all(math.isfinite(r[k]) for k in ('PL','CM','total','lora_grad_norm')) and r['CM']>=0
            assert abs(r['PL']+r['CM']-r['total'])<2e-7
            assert (r['CM_grad_norm'] is not None)==(r['attempt'] in (1,101,500,900))
            if r['P']:
                assert 0<=r['active_count']<=r['P'] and abs(r['active_fraction']-r['active_count']/r['P'])<1e-12
                assert r['d_mean']**2<=r['CM']+2e-7 and r['CM']<=r['d_max']**2+2e-7
            else:assert r['CM']==r['active_count']==0 and r['active_fraction'] is None
    activity=load(ROOT/'ACTIVITY_SUMMARY.json')
    assert activity['active_pixel_exposures']==sum(r['active_count'] for r in curve)
    assert activity['P_updated_cumulative']==sum(r['P'] for r in curve if r['updated'])==19937970
    groups=read(ROOT/'DAMAGE_BENEFIT.csv')
    for strategy in ('Q90-PL','Q90-SP','Q90-CM'):
        for region in ('all','P','A','other'):
            pooled=next(r for r in groups if r['strategy']==strategy and r['region']==region and r['product']=='ALL')
            records=[r for r in groups if r['strategy']==strategy and r['region']==region and r['product']!='ALL']
            for k in pooled:
                if k not in ('strategy','region','product','group_id','D_repair_fraction','Cloud_correction_retention'):assert int(pooled[k])==sum(int(r[k]) for r in records),k
            assert sum(int(pooled[k]) for k in ('D_to_Surface','D_still_Cloud','D_to_Shadow'))==int(pooled['original_D'])
            assert sum(int(pooled[k]) for k in ('benefit_to_Surface','benefit_retained_Cloud','benefit_back_Shadow'))==int(pooled['original_Cloud_correction'])
    calls=[json.loads(x) for x in (ROOT/'CALLS.jsonl').read_text().splitlines()]
    assert len(calls)==2238 and len(calls)<=2400
    contexts=collections.Counter(r['context'] for r in calls)
    assert contexts['formal_train']==1910 and contexts['evaluation']==306 and contexts['gates']==6
    # Preserve generated wording; this supplemental review changes no scientific CSV or model.
    numeric_sha={p.name:sha(p) for p in ROOT.glob('*.csv')}
    report=ROOT/'RESULT.md';shutil.copyfile(report,ROOT/'RESULT_GENERATED.md')
    metrics={r['strategy']:r for r in read(ROOT/'METRICS.csv')};dam={r['strategy']:r for r in groups if r['product']=='ALL' and r['region']=='all'}
    cm=metrics['Q90-CM'];pl=metrics['Q90-PL'];sp=metrics['Q90-SP']
    overall=float(cm['group_mean_mIoU_pct'])-float(pl['group_mean_mIoU_pct']);retain=int(dam['Q90-CM']['benefit_retained_Cloud'])-int(dam['Q90-SP']['benefit_retained_Cloud']);fpr=float(cm['macro_FPR_pct'])-float(sp['macro_FPR_pct'])
    answer=f"\n复核结论：相对PL，组均mIoU变化{overall:+.6f}pp；相对SP，原B保留Cloud变化{retain:+d}、宏Shadow FPR变化{fpr:+.6f}pp。"+('本次同时保留更多原Cloud纠正且降低宏Shadow误报。' if retain>0 and fpr<0 else '未同时实现更多原Cloud纠正保留和更低宏Shadow误报。')+'总体取舍仍须并看池化指标及新增错误；不能由一个方向改善概括整体成功。\n'
    report.write_text(report.read_text(encoding='utf-8')+answer,encoding='utf-8')
    shutil.copyfile(ROOT/'TABLE_README.md',ROOT/'TABLE_README_GENERATED.md')
    (ROOT/'TABLE_README.md').write_text('TRANSITIONS27字段 truth/Source_pred/student_pred 对应真值/比较左端/比较右端；ERROR_FLOW字段 Source_errors/student_errors 同理，不恒指Source。comparison明确三组对照。D/B固定于Source与PL，不按新策略重定义。D外新Cloud/Shadow均相对Source正确Surface计数，非PL→CM新增事件数；后者见ERROR_FLOW。P/A/other由影像Source定义。跳过步active_count=0只是未求值占位；全部边际、池化及分区由独立CSV检查复核。\n',encoding='utf-8')
    assert all(sha(ROOT/n)==h for n,h in numeric_sha.items())
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    assert peak<16*1024**3 and available>=4000000000
    script=R/'src/tools/review_phase311.py';shutil.copyfile(script,ROOT/'code'/script.name)
    write(ROOT/'REVIEW_ACCEPTANCE.json',dict(status='passed',original_execution_code_and_inputs_unchanged=True,analysis_input_SHA_checked=True,independent_CSV_27cells_margins_partition_pool_and_errorflow_checked=True,all_numeric_CSV_unchanged=numeric_sha,original_report_sha256=sha(ROOT/'RESULT_GENERATED.md'),reviewed_report_sha256=sha(ROOT/'RESULT.md'),calls=len(calls),contexts=dict(contexts),review_script_sha256=sha(script),seconds=time.monotonic()-started,peak_cpu_bytes=peak,minimum_available_memory=available,model_forwards=0,new_training=0))
    print(json.dumps(dict(status='passed',calls=len(calls),overall_delta_pp=overall,retention_change_vs_SP=retain,macro_FPR_delta_vs_SP=fpr)))

if __name__=='__main__':
    try:main()
    except BaseException:
        append('REVIEW_FAILURE.jsonl',dict(traceback=__import__('traceback').format_exc(),epoch=time.time(),action='Preserve inputs and output; diagnose before any retry.'));raise

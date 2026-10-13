"""Local PixBox reference join; fixed matched-control aggregates and reports."""
import csv,shutil
from phase107_common import *
OUT=REPO/'outputs/phase107/phase107_20261013'
DATA=REPO/'data/pixbox_s2_cmix_5036991/phase107_20261013'

def main():
    seal=read(OUT/'PixBox_score_seal.json');lock=read(RESULT/'pre_fit_lock.json')
    assert sha(DATA/'PixBox_scores_no_labels.json')==seal['score_sha256']
    assert sha(OUT/'model_lock.json')==seal['model_lock_sha256']
    assert sha(OUT/'development_aggregate.json')==seal['development_aggregate_sha256']
    assert sha(OUT/'R_model.json')==read(OUT/'model_lock.json')['R_model_sha256']
    write(RESULT/'LOCAL_SCORE_SEAL.json',dict(utc=now(),score_sha256=seal['score_sha256'],before_reference_join=True),True)
    ref=REPO/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json';assert sha(ref)==lock['reference_sha256']
    refs={r['point_ids'][0]:r for r in read(ref)};scores=read(DATA/'PixBox_scores_no_labels.json')['rows']
    assert len(scores)==len(refs)==9500 and len({r['point_id'] for r in scores})==9500
    groups={};joined=[]
    for r in scores:
        q=refs[r['point_id']];assert all(str(r[k])==str(q[k]) for k in ('product_id','group_id','row20','col20'))
        assert (r['R'] is not None)==r['input_valid'] and (r['S'] is not None)==r['input_valid']
        item=dict(**r,reference=q['reference']);joined.append(item)
        if r['input_valid'] and q['reference']!='ignore':groups.setdefault(r['group_id'],[]).append(item)
    assert len(groups)==10
    allrows=[r for rr in groups.values() for r in rr]
    assert len(allrows)==8567 and sum(r['reference']=='positive_ground_cloud_shadow' for r in allrows)==379
    assert sum(r['reference']=='ignore' for r in joined)==931 and sum(not r['input_valid'] for r in joined)==2
    csvwrite(DATA/'joined_reference_scores_private.csv',joined)
    prior=REPO/'src/research_results/phase106_20261013'
    assert sha(prior/'per_group_results.csv')==lock['phase106_group_sha256']
    assert sha(prior/'two_score_summary.csv')==lock['phase106_summary_sha256']
    with (prior/'per_group_results.csv').open(encoding='utf-8',newline='') as f:old={r['group_id']:r for r in csv.DictReader(f)}
    with (prior/'two_score_summary.csv').open(encoding='utf-8',newline='') as f:summary={r['score']:r for r in csv.DictReader(f)}
    rows=[]
    # Complete Phase106 A/S reproduction before any new R ranking evaluation.
    for gid,rr in sorted(groups.items()):
        y=np.array([r['reference']=='positive_ground_cloud_shadow' for r in rr])
        a=ap(y,[r['A'] for r in rr]);s=ap(y,[r['S'] for r in rr]);o=old[gid]
        assert (a is None)==(o['A_AP']=='') and (s is None)==(o['B_AP']=='')
        if a is not None:assert a==float(o['A_AP']) and s==float(o['B_AP'])
        rows.append(dict(group_id=gid,tile=rr[0]['product_id'].split('_T')[1][:5],positive=int(y.sum()),negative=int((~y).sum()),
                         A_AP=a,R_AP=None,S_AP=s,S_minus_R_pp=None,R_minus_A_pp=None))
    y=np.array([r['reference']=='positive_ground_cloud_shadow' for r in allrows]);pooled={}
    for name,oldname in [('A','A'),('S','B')]:
        pooled[name]=ap(y,[r[name] for r in allrows]);assert pooled[name]*100==float(summary[oldname]['pooled_AP_percent'])
        assert float(np.mean([r[name+'_AP'] for r in rows if r[name+'_AP'] is not None]))*100==float(summary[oldname]['macro_AP_percent'])
    for row in rows:
        rr=groups[row['group_id']];yy=np.array([r['reference']=='positive_ground_cloud_shadow' for r in rr])
        value=ap(yy,[r['R'] for r in rr]);row['R_AP']=value
        if value is not None:row['S_minus_R_pp']=(row['S_AP']-value)*100;row['R_minus_A_pp']=(value-row['A_AP'])*100
    pooled['R']=ap(y,[r['R'] for r in allrows])
    totals=[dict(domain='PixBox',score=name,macro_AP_percent=float(np.mean([r[name+'_AP'] for r in rows if r[name+'_AP'] is not None]))*100,
            pooled_AP_percent=pooled[name]*100,AP_groups=8,total_groups=10) for name in ('A','R','S')]
    dev=read(OUT/'development_aggregate.json');assert len(dev['rows'])==32 and dev['CPU']['status']=='passed'
    comps=comparisons(rows,'PixBox')+dev['comparisons']
    csvwrite(RESULT/'two_pool_three_score_summary.csv',dev['totals']+totals)
    csvwrite(RESULT/'PixBox_per_group.csv',rows);csvwrite(RESULT/'development_per_group.csv',dev['rows'])
    write(RESULT/'fixed_comparisons.json',dict(rows=comps,sole_primary='PixBox S-R',both_pools_previously_exposed=True),True)
    write(RESULT/'CPU_acceptance.json',dict(status='passed',Phase106_A_S_AP_exact=True,development_A_S_AP_exact=True,
        AP_independent_bin_replay=True,paired_group_intervals_scalar_replayed=True,PixBox_points=9500,positive=379,negative=8188,
        ignore=931,invalid=2,PixBox_score_replay=seal['CPU'],development_score_replay=dev['CPU'],
        matched_fit_input_lock_sha256=sha(OUT/'fit_input_lock.json'),sole_fit_model_lock=read(OUT/'model_lock.json'),
        zero_new_network_forwards=True,no_new_thresholds=True),True)
    main=next(r for r in comps if r['primary'])
    table='\n'.join(f"|{r['domain']}|{r['score']}|{r['macro_AP_percent']:.3f}|{r['pooled_AP_percent']:.3f}|" for r in dev['totals']+totals)
    differences='\n'.join(f"|{r['domain']}|{r['comparison']}|{r['mean_delta_AP_pp']:.3f}|[{r['CI95_pp'][0]:.3f}, {r['CI95_pp'][1]:.3f}]|{r['improved']}/{r['stable']}/{r['declined']}|" for r in comps)
    conclusion='特异性证据不确定' if main['evidence']=='uncertain' else '这两个固定同容量读出下观察到光谱增量' if main['evidence']=='positive' else 'RGB读出达到或超过光谱读出，削弱当前特异性主张'
    text=f'''# Phase107 Source读出同容量RGB对照\n\n## 1 数据结果\n\nA为原Source概率，R为唯一新拟合的Source_RGB四输入逻辑回归，S为原153组Source_spectral四输入逻辑回归。复用153组原索引/标签/组权，均约束Source系数非负，正则.001；S不重拟合，R不拟合阈值。\n\n|池|评分|宏AP(%)|池化AP(%)|\n|---|---|---:|---:|\n{table}\n\n|池|固定比较|平均AP差(pp)|配对相关组95%描述区间(pp)|改善/稳定/下降|\n|---|---|---:|---|---|\n{differences}\n\n唯一主要描述比较为PixBox S−R；8组AP，development7组AP。bootstrap10000次seed107013；无正类AP为NA。最差PixBox S−R组{main['worst_group']}：{main['worst_delta_AP_pp']:.3f}pp。全部逐组数值见两个per_group表。\n\n拟合次数1，新增网络训练/前向/影像下载/阈值为0；原A/S旧AP完整复现后才评价R，独立CPU标准化、评分、AP并列和配对区间验收通过。\n\n## 2 结论分析\n\n{conclusion}。两池性能均已暴露，本轮是探索性对照，不是独立确认；不与Phase105/106事后合并成新确认。即使S超过R，也只支持这两个固定同容量读出下的光谱线索，不能证明光谱物理机制或所有RGB方法较差。区间跨零只称证据不确定，不称等效。未新增阈值或决策指标，不选部署赢家，不继续搜索波段/正则；保留参考语义、选择性采样、未知基础预训练独立性及原失败结论。\n'''
    (RESULT/'README.md').write_text(text,encoding='utf-8',newline='\n')
    for name in ('R_model.json','model_lock.json','fit_input_lock.json','PixBox_score_seal.json'):shutil.copyfile(OUT/name,RESULT/name)
    write(RESULT/'cost_and_completion.json',dict(status='complete',utc=now(),fit_attempts=1,
        fit_seconds=read(OUT/'model_lock.json')['fit_seconds'],new_forwards=0,new_download_bytes=0,new_thresholds=0,
        PixBox_score_sha256=seal['score_sha256'],local_private_join_sha256=sha(DATA/'joined_reference_scores_private.csv'),
        report_sha256={p.name:sha(p) for p in sorted(RESULT.iterdir()) if p.is_file()}),True)
    for p in RESULT.iterdir():
        if p.is_file():shutil.copyfile(p,OUT/p.name)
    print(json.dumps(main,ensure_ascii=True,indent=2))

if __name__=='__main__':main()

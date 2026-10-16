"""Reference-side SCD statistics, after immutable label-blind strata locking."""
import argparse
import collections
from pathlib import Path
import traceback
from scd1_common import *


def mean(values):
    a=[v for v in values if v is not None]
    return sum(a)/len(a) if a else None


def evaluate_groups(rows,groups,tiles,layer):
    output=[]
    from phase89_cpu_ap import average_precision_score
    for gid in groups:
        rr=[r for r in rows if r['group_id']==gid]
        y=[r['reference']=='positive_ground_cloud_shadow' for r in rr]
        entry=dict(layer=layer,group_id=gid,tiles=tiles[gid],points=len(rr),positive=sum(y),negative=len(y)-sum(y))
        for setting in SETTINGS:
            scores=[r[setting] for r in rr]
            m=metrics(y,scores)
            if m['AP'] is not None:
                import numpy as np
                near(m['AP'],average_precision_score(y,np.array(scores,np.float32)))
            for k,v in m.items():entry[setting+'_'+k]=v
        for a,b in [('H10','C20'),('C20','O20')]:
            entry[a+'-'+b+'_AP_pp']=(entry[a+'_AP']-entry[b+'_AP'])*100 if entry[a+'_AP'] is not None else None
        output.append(entry)
    return output


def reproduce(root,groups,rows):
    old={r['group_id']:r for r in csvread(root/'per_group_results.csv')}
    assert set(old)=={r['group_id'] for r in groups} and len(groups)==10
    for r in groups:
        previous=old[r['group_id']]
        assert int(previous['positive'])==r['positive'] and int(previous['negative'])==r['negative']
        assert previous['tiles']==r['tiles']
        for setting in SETTINGS:
            for current,legacy in [('AP','AP'),('TP','TP'),('FP','FP'),('recall','recall'),('precision','precision'),('FPR','negative_FP_fraction')]:
                raw=previous[setting+'_'+legacy]
                near(r[setting+'_'+current],float(raw) if raw else None)
    summaries=[]
    old_summary={r['setting']:r for r in csvread(root/'three_setting_summary.csv')}
    for setting in SETTINGS:
        pooled=metrics([r['reference']=='positive_ground_cloud_shadow' for r in rows],[r[setting] for r in rows])
        s=dict(setting=setting,AP_groups=sum(r[setting+'_AP'] is not None for r in groups),
               macro_AP_pp=mean([r[setting+'_AP'] for r in groups])*100,
               pooled_AP_pp=pooled['AP']*100,
               macro_recall_pp=mean([r[setting+'_recall'] for r in groups])*100,
               macro_negative_FP_pp=mean([r[setting+'_FPR'] for r in groups])*100,
               worst_group_negative_FP_pp=max(r[setting+'_FPR'] for r in groups)*100)
        for k,v in s.items():
            if k not in ('setting','AP_groups'):near(v/100,float(old_summary[setting][k])/100)
        assert s['AP_groups']==int(old_summary[setting]['AP_groups'])==8
        for k in ('positive','negative','TP','FP'):
            assert pooled[k]==int(old_summary[setting]['pooled_'+k])
        for a,b in [('recall','recall'),('precision','precision'),('FPR','negative_FP_fraction')]:
            near(pooled[a],float(old_summary[setting]['pooled_'+b]))
        summaries.append(s)
    for c in read(root/'FIXED_COMPARISONS.json')['rows']:
        a,b=c['comparison'].split('-')
        delta=mean([(r[a+'_AP']-r[b+'_AP'])*100 for r in groups if r[a+'_AP'] is not None])
        near(delta/100,c['mean_delta_AP_pp']/100)
        assert c['n_related_groups']==8
    return summaries


def run(root,geometry_root,reference,out,budget):
    manifest=load_inputs(root)
    assert sha(Path(__file__).with_name('phase89_cpu_ap.py'))==read(root/'EVALUATION_LOCK.json')['source_sha256']['src/tools/phase89_cpu_ap.py']
    for name,h in read(geometry_root/'CODE_BINDINGS.json')['source_sha256'].items():
        assert sha(Path(__file__).with_name(name))==h, name
    lock=read(geometry_root/'STRATA_LOCK.json')
    assert lock['before_reference_join'] and not lock['reference_read']
    assert sha(geometry_root/'point_geometry.csv')==lock['geometry_sha256']
    assert sha(root/'INPUT_MANIFEST.json')==lock['input_manifest_sha256']
    assert list(lock['strata'])==list(LAYERS)
    geometry_rows=csvread(geometry_root/'point_geometry.csv')
    assert len(geometry_rows)==len({r['point_id'] for r in geometry_rows})==9500
    groups=sorted({r['group_id'] for r in geometry_rows});assert len(groups)==10
    tiles={gid:','.join(sorted({r['product_id'].split('_T')[1][:5] for r in geometry_rows if r['group_id']==gid})) for gid in groups}
    counts={gid:{k:0 for k in LAYERS} for gid in groups}
    for r in geometry_rows:
        for layer in memberships(r):counts[r['group_id']][layer]+=1
    assert counts==lock['group_layer_unlabelled_counts']
    scores=score_points(root)
    assert lock['score_sha256']==read(root/'COMMON_SUPPORT_LOCK.json')['all_score_sha256']
    # This is the first read of point reference semantics in either SCD stage.
    assert sha(reference)==manifest['reference_sha256']==read(root/'EVALUATION_LOCK.json')['original_reference_sha256']
    original=read(reference)
    refs={q['point_ids'][0]:q for q in original}
    assert len(refs)==len(original)==9500 and set(refs)==set(scores['O20'])
    rows=[];ignore=0
    for r in geometry_rows:
        q=refs[r['point_id']]
        assert all(str(q[k])==str(r[k]) for k in ('product_id','group_id','row10','col10','row20','col20'))
        assert q['reference'] in ('ignore','positive_ground_cloud_shadow','negative_surface','negative_cloud')
        if r['common_valid']!='True':continue
        if q['reference']=='ignore':ignore+=1;continue
        item=dict(r,reference=q['reference'])
        for setting in SETTINGS:item[setting]=scores[setting][r['point_id']]['shadow_confidence']
        rows.append(item)
    pos=sum(r['reference']=='positive_ground_cloud_shadow' for r in rows)
    assert (pos,len(rows)-pos,ignore)==(379,8190,931)
    full=evaluate_groups(rows,groups,tiles,'all')
    summaries=reproduce(root,full,rows)
    write(out/'FULL_REPRODUCTION.json',dict(status='passed',per_group=full,summary=summaries,
        positive=pos,negative=len(rows)-pos,ignore=ignore,original_points=9500,
        no_support_loss=True,no_new_intervals=True,original_eight_group_definition=True))
    details=[];summaries=[]
    text=['|分层|有效点数|正／负数|AP有定义组数|H10−C20平均AP差 pp|C20−O20平均AP差 pp|最坏组及AP差 pp|备注|',
          '|---|---|---|---|---|---|---|---|']
    for layer in LAYERS:
        subset=[r for r in rows if layer in memberships(r)]
        table=full if layer=='all' else evaluate_groups(subset,groups,tiles,layer)
        details.extend(table)
        eligible=[r for r in table if r['O20_AP'] is not None]
        aa=mean([r['H10-C20_AP_pp'] for r in table]);bb=mean([r['C20-O20_AP_pp'] for r in table])
        positive=sum(r['reference']=='positive_ground_cloud_shadow' for r in subset)
        sparse=sum(r['O20_sparse'] for r in table)
        no_positive=sum(r['positive']==0 for r in table);no_negative=sum(r['negative']==0 for r in table)
        s=dict(layer=layer,points=len(subset),positive=positive,negative=len(subset)-positive,AP_groups=len(eligible),
               H10_minus_C20_mean_AP_pp=aa,C20_minus_O20_mean_AP_pp=bb,sparse_groups=sparse,
               no_positive_groups=no_positive,no_negative_groups=no_negative,
               unlabelled_points=sum(counts[g][layer] for g in groups),
               ignore_points=sum(counts[g][layer] for g in groups)-len(subset))
        summaries.append(s)
        worst=[]
        for key in ('H10-C20_AP_pp','C20-O20_AP_pp'):
            if eligible:
                w=min(eligible,key=lambda r:(r[key],r['group_id']))
                worst.append(key.replace('_AP_pp','')+':'+w['tiles']+f" {w[key]:+.4f}")
            else:worst.append(key.replace('_AP_pp','')+':NA（无正负俱全组）')
        note=f"稀疏{sparse}/10组；无正{no_positive}、无负{no_negative}（AP为NA）；ignore {s['ignore_points']}；无覆盖/支持损失0/0"
        if layer=='all':note+='；118全体三策略AP及0.5操作指标复现一致；9500原点保留'
        if len(eligible)==1:note+='；仅1组有定义，不推断总体规律'
        for tile in ('22WES','44SKJ','54TWN','19FDV'):
            rr=next(r for r in table if tile in r['tiles'].split(','))
            if rr['O20_AP'] is not None:
                note+=f"；{tile}正/负{rr['positive']}/{rr['negative']}，H−C/C−O {rr['H10-C20_AP_pp']:+.2f}/{rr['C20-O20_AP_pp']:+.2f}pp"
            else:note+=f"；{tile} NA（正/负{rr['positive']}/{rr['negative']}）"
        fmt=lambda v:'NA（无正负俱全组）' if v is None else f'{v:+.6f}'
        title=layer.replace('all','全体').replace('>=','≥')
        text.append(f"|{title}|{len(subset)}|{positive}/{len(subset)-positive}|{len(eligible)}/10|{fmt(aa)}|{fmt(bb)}|{'；'.join(worst)}|{note}|")
        budget.receipt()
    csvwrite(out/'GROUP_LAYER_RESULTS.csv',details)
    write(out/'LAYER_SUMMARIES.json',summaries)
    write(out/'REFERENCE_JOIN_RECEIPT.json',dict(status='passed',reference_sha256=sha(reference),
        geometry_sha256=lock['geometry_sha256'],strata_lock_sha256=sha(geometry_root/'STRATA_LOCK.json'),
        labels_uploaded=False,all_ten_groups_retained_in_each_layer=True,
        no_new_bootstrap_or_threshold_search=True,independent_tied_AP_checked=True))
    receipt=budget.receipt();write(out/'RESOURCE_RECEIPT.json',receipt)
    text.append(f"|运行状态|—|—|—|—|—|—|完成；累计CPU阶段{receipt['cumulative_CPU_stage_seconds']:.3f}秒；几何/9500点身份/原指标复现/双实现AP检查通过；无异常；执行端/home/scv/shared/phasescd_1_20261015（参考连接本地）|")
    (out/'phasescd-1-result.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    (out/'EXIT_CODE').write_text('0\n');budget.receipt()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-root',type=Path,required=True);p.add_argument('--geometry-root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
    a.output_root.mkdir(parents=True,exist_ok=False)
    prior=read(a.geometry_root/'RESOURCE_RECEIPT.json')['cumulative_CPU_stage_seconds']
    budget=Budget([a.input_root,a.geometry_root,a.output_root],prior);budget.receipt()
    try:run(a.input_root,a.geometry_root,a.reference,a.output_root,budget)
    except Exception:
        write(a.output_root/'FAILURE.json',dict(error=traceback.format_exc(),new_model_calls=0))
        status_table(a.output_root,'几何已完成，参考连接或全体复现失败，详见执行端FAILURE.json','部分完成')
        raise


if __name__=='__main__':main()

"""Frozen Phase67 input controls on Phase89 caches; CPU only, no fit/forward."""
import io
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
import numpy as np
import run_phase89 as old
from phase65d_recovery_math import logit_score, predict_logit
from phase67_controls import CONTROLS, combined, selected
from phase89_math import area, sigmoid, calibrate, decisions
from phase89_cpu_ap import average_precision_score, self_test
from phase65d_metrics import confusion, corrected, curve

ROOT = old.SHARED / 'phase90_20261012'
DELIVERY = ROOT / 'delivery'
P67 = old.CAT / 'phase67_20261010'
PROTOCOL = old.REPO / 'src/research_plans/PHASE90_EXTERNAL_INPUT_ABLATIONS_20261012.md'
CONTROL_SHA = 'c2a83964073918015460e0ad89a9857b60b6a764e0f8562652472b224915f5ad'
P67_REPORT_SHA = '49e50f4f1bea2ef71e5dc533846243dd0fbc543e32adb0b46cc6529185afab71'
P89_MANIFEST_SHA = 'a2320cb6e34cdaf6cc60cfd76e5c8adbe6d893c46dd39f682ddae5484ad8c812'
NAMES = (*old.NAMES, *CONTROLS)
VERSIONS = [('original_main', 'truth', 'valid'), ('original_common', 'truth', 'common'),
            ('revised_common', 'revised', 'common')]
SEED = 90012


def interval(values):
    x = np.asarray(values, np.float64)
    result = dict(n_groups=len(x), mean=float(x.mean()) if len(x) else None,
                  ci95=None, upper95_one_sided=None, seed=SEED, resamples=10000,
                  descriptive_only=True, unit='equal related-group representative')
    if len(x) > 1:
        draws = np.random.default_rng(SEED).integers(len(x), size=(10000, len(x)))
        means = x[draws].mean(1)
        result.update(ci95=np.quantile(means, [.025, .975]).tolist(),
                      upper95_one_sided=float(np.quantile(means, .95)))
    return result


def independent_ap(y, s, reference):
    y = np.asarray(y) == 2
    if not y.any() or y.all():
        assert reference is None
    else:
        value = average_precision_score(y, np.asarray(s, np.float32))
        assert abs(value-reference) < 1e-12, (value, reference)


def independent_prob(x, model, shape):
    # No predict_logit/sigmoid/area calls: separate CPU standardization and sum.
    z = (x-np.array(model['mean'])) / np.array(model['scale'])
    logits = (z @ np.array(model['theta'][:-1]) + model['theta'][-1]).astype(np.float32)
    p = np.exp(-np.logaddexp(0., -logits.astype(np.float64))).reshape(shape)
    return sum(p[i::3, j::3] for i in range(3) for j in range(3))/9


def inputs(ms, ss, spectra, name):
    assert spectra.shape[:2] == ms.shape == ss.shape and spectra.shape[-1] == 6
    if name == 'L2_RGB':
        columns = [logit_score(ms).ravel(), logit_score(ss).ravel(),
                   spectra[..., 2].ravel(), spectra[..., 1].ravel(), spectra[..., 0].ravel()]
    elif name == 'Source_spectral':
        columns = [logit_score(ss).ravel(), *[spectra[..., i].ravel() for i in [3, 4, 5]]]
    elif name == 'MsRE_spectral':
        columns = [logit_score(ms).ravel(), *[spectra[..., i].ravel() for i in [3, 4, 5]]]
    else:
        raise ValueError(name)
    return np.column_stack(columns).astype(np.float32)


def control_decision(score, probabilities, model):
    network='Source' if model['nonshadow']=='source' else 'MsRE'
    return np.where(score>=sigmoid(model['threshold']),2,
                    probabilities[network][...,:2].argmax(-1)).astype(np.uint8)


def blocks(row):
    product = row['product']
    spectra = old.array(row, 'spectra20')
    idx = [next(r for r in old.read(old.ROOT / ('scores_'+n) / 'index.json')['rows']
                if r['product'] == product) for n in old.NAMES[:2]]
    assert len(idx[0]['blocks']) == len(idx[1]['blocks']) == 36
    for sb, mb in zip(idx[0]['blocks'], idx[1]['blocks']):
        assert all(sb[k] == mb[k] for k in ['y60', 'x60', 'h60', 'w60'])
        values = []
        for block in [sb, mb]:
            path = old.ROOT / block['path']
            assert old.digest(path) == block['sha256']
            a = np.load(path, allow_pickle=False)
            assert a.shape == (sb['h60']*3, sb['w60']*3, 3) and a.dtype == np.float32
            assert np.isfinite(a).all()
            values.append(a)
        y, x, h, w = [sb[k] for k in ['y60', 'x60', 'h60', 'w60']]
        yield y, x, h, w, values[0], values[1], spectra[y*3:(y+h)*3, x*3:(x+w)*3]


def preflight():
    assert not ROOT.exists(), 'Never overwrite or automatically rerun a Phase90 attempt'
    old.ROOT = old.SHARED / 'phase89_recovered_20261011'
    assert old.digest(old.ROOT/'delivery/DELIVERY_SHA256.json') == P89_MANIFEST_SHA
    assert old.digest(P67/'control_lock.json') == CONTROL_SHA
    assert old.digest(P67/'delivery/phase67_report.json') == P67_REPORT_SHA
    for rel, sha in old.read(old.ROOT/'delivery/DELIVERY_SHA256.json').items():
        assert old.digest(old.ROOT/'delivery'/rel) == sha, rel
    old.locked()
    controls = old.read(P67/'control_lock.json')
    report67 = old.read(P67/'delivery/phase67_report.json')
    assert report67['control_lock_sha256'] == CONTROL_SHA
    assert old.digest(P67/'delivery/control_lock.json') == CONTROL_SHA
    for name, sha in controls['code_sha256'].items():
        assert old.digest(old.REPO/'src/tools'/name) == sha, name
    assert set(controls['models']) == set(CONTROLS)
    for name, m in controls['models'].items():
        assert m['columns'] == CONTROLS[name]['columns']
        assert m['nonshadow'] == CONTROLS[name]['nonshadow']
        assert len(m['mean']) == len(m['scale']) == len(m['columns'])
        assert np.isfinite([*m['mean'], *m['scale'], *m['theta'], m['threshold']]).all()
        assert min(m['scale']) > 0
    prepared = old.read(old.ROOT/'prepared/index.json')['rows']
    assert len(prepared) == len({r['group_id'] for r in prepared}) == 9
    assert sum(r['revised_available'] for r in prepared) == 6
    ROOT.mkdir(); DELIVERY.mkdir()
    pin = dict(created_utc=old.now(),protocol_sha256=old.digest(PROTOCOL),
               phase67_control_sha256=CONTROL_SHA, models=controls['models'],
               phase67_report_sha256=old.digest(P67/'delivery/phase67_report.json'),
               phase89_delivery_manifest_sha256=P89_MANIFEST_SHA,
               phase89_recovery_lock_sha256=old.digest(old.ROOT/'recovery_lock.json'),
               phase89_runtime_index_sha256=old.digest(old.ROOT/'delivery/input_and_runtime_index.json'),
               prepared=old.read(old.ROOT/'prepared/index.json'),
               inference={n:old.read(old.ROOT/('scores_'+n)/'index.json') for n in old.NAMES[:2]},
               processing=dict(input_bands=['B02','B03','B04','B08','B11','B12'],
                   RGB_control_order=['B04','B03','B02'],RGB_scale='physical TOA; no 255 multiplier',
                   clip=[1e-6,1-1e-6],network_logit='float64 clip/log then cast combined features float32',
                   shallow_logit='frozen standardization float64; output cast float32',
                   aggregation='sigmoid at20m float64 then 3x3 arithmetic mean at60m',
                   decisions='score>=sigmoid(original threshold); nonshadow original frozen network60m argmax',
                   AP='existing tied-score float32 non-interpolated AP',bootstrap_seed=SEED,resamples=10000),
               human_original_sha256=old.digest(old.HUMAN),human_B='not_executed',
               frozen_before_external_control_calculation=True,
               new_training=0,new_fits=0,new_forwards=0,new_downloads=0,
               formal_confirmation=False,reused_precrash_probability_SHA_limitation=True,
               code_sha256={p.name:old.digest(p) for p in [Path(__file__),Path(__file__).with_name('phase89_cpu_ap.py')]})
    old.write(DELIVERY/'parameter_input_lock.json', pin)
    return prepared, controls['models'], old.read(old.ROOT/'delivery/phase89_report.json'), report67


def reproduce89(prepared, reference):
    models = old.locked()[2]; checks = []
    for row in prepared:
        prob = old.probabilities(row['product'])
        coverage = np.zeros(row['shape60'], np.uint8)
        for y, x, h, w, ss, ms, spec in blocks(row):
            sl = np.s_[y:y+h, x:x+w]
            for name, values in [('Source',ss),('MsRE',ms)]:
                assert np.array_equal(area(values,3).astype(np.float32),prob[name][sl])
            calibrated = calibrate(ms[...,2],ss[...,2],spec[...,3:],models)
            for name in ['L2','L3']:
                assert np.array_equal(area(calibrated[name],3),prob[name][sl]), (row['product'],name)
            coverage[sl] += 1
        assert (coverage == 1).all()
        pred = decisions(prob,models)
        for version,tkey,mkey in VERSIONS:
            truth=old.array(row,tkey);mask=old.array(row,mkey)
            if not mask.any():continue
            y=truth[mask]
            for name in old.NAMES:
                ref=next(r for r in reference['versions'][version]['policies'][name]['rows'] if r['product']==row['product'])
                s=prob[name][...,2][mask] if name in old.NAMES[:2] else prob[name][mask]
                assert np.array_equal(confusion(y,pred[name][mask]),ref['confusion'])
                assert old.ap(s,y==2)==ref['ap']
                independent_ap(y,s,ref['ap'])
                checks.append(dict(product=row['product'],version=version,policy=name,score_reaggregation_exact=True,confusion_exact=True,AP_exact=True))
        print('Phase89 reproduction',row['tile'],len(checks),flush=True)
    old.write(DELIVERY/'phase89_reproduction.json',dict(rows=checks,statistic_rows=len(checks),native_blocks=648,no_forward=True))


def reproduce67(models, reference):
    from run_phase65d_recovery import records, prepare_verify, PREP, OLD, load, features
    sources=[{r['product']:r for r in old.read(OLD/'development_val'/('index_'+n+'.json'))['rows']} for n in ['source','msre']]
    checks=[]
    with zipfile.ZipFile(old.CAT/'subscenes.zip') as z:
        for product,r in records('development_val').items():
            prepare_verify(r);truth=np.load(PREP/r['mask_path']);valid=truth!=255
            rawbytes=z.read('subscenes/'+product+'.npy')
            assert old.hashlib.sha256(rawbytes).hexdigest()==r['member_sha256']['image']
            raw=np.load(io.BytesIO(rawbytes));ss=load(OLD,sources[0][product],'score');ms=load(OLD,sources[1][product],'score')
            x=combined(features(raw,ms,ss,valid),raw[...,[3,2,1]][valid])
            ns={n:load(OLD,sources[i][product],'nonshadow')[valid] for i,n in enumerate(['source','msre'])}
            for name,model in models.items():
                score=predict_logit(selected(x,name),model)
                ref=next(t for t in reference['policies'][name]['rows'] if t['product']==product)
                assert np.array_equal(confusion(truth[valid],corrected(score,model['threshold'],ns[model['nonshadow']])),ref['confusion'])
                value=old.ap(score,truth[valid]==2);assert value==ref['ranking']['ap']
                independent_ap(truth[valid],score,value)
                checks.append(dict(product=product,policy=name,confusion_exact=True,AP_exact=True,
                                   recomputed_logit_sha256=old.hashlib.sha256(score.tobytes()).hexdigest()))
            print('Phase67 reproduction',len(checks)//3,32,flush=True)
    assert len(checks)==96
    old.write(DELIVERY/'phase67_reproduction.json',dict(rows=checks,
        note='Phase67 did not seal per-pixel control score arrays; exact historical confusion/AP reproduced; current recomputed logit hashes recorded without claiming an old byte seal.',
        independent_score_bin_AP=True,no_fit=True,no_forward=True))


def aggregate_controls(row, models):
    result={n:np.empty(row['shape60'],np.float64) for n in CONTROLS};checks=[]
    for y,x,h,w,ss,ms,spec in blocks(row):
        for name,model in models.items():
            xx=inputs(ms[...,2],ss[...,2],spec,name)
            value=area(sigmoid(predict_logit(xx,model)).reshape(h*3,w*3),3)
            # Independent selection starts from the full ordered feature table.
            mc=np.clip(ms[...,2].astype(np.float64),1e-6,1-1e-6)
            sc=np.clip(ss[...,2].astype(np.float64),1e-6,1-1e-6)
            full=np.column_stack([(np.log(mc)-np.log1p(-mc)).ravel(),(np.log(sc)-np.log1p(-sc)).ravel(),
                                  spec[...,3:].reshape(-1,3),spec[...,[2,1,0]].reshape(-1,3)]).astype(np.float32)
            other=independent_prob(full[:,model['columns']],model,(h*3,w*3))
            error=float(np.max(np.abs(value-other)))
            np.testing.assert_allclose(value,other,rtol=0,atol=2e-15)
            threshold=float(np.exp(-np.logaddexp(0.,-float(model['threshold']))))
            assert np.array_equal(value>=sigmoid(model['threshold']),other>=threshold)
            result[name][y:y+h,x:x+w]=value
            checks.append(dict(policy=name,y60=y,x60=x,max_absolute_mean_error=error,decisions_equal=True))
    return result,checks


def summarize_policy(rows, scores, y):
    cm=np.sum([r['confusion'] for r in rows],axis=0)
    pooled=dict(ap=old.ap(scores,y==2),**old.measures(cm))
    independent_ap(y,scores,pooled['ap'])
    result=dict(rows=rows,pooled=pooled,groups=len(rows),
                zero_recall_groups=[r['product'] for r in rows if r['recall']==0],
                groups_above_1percent=sum(r['fpr'] is not None and r['fpr']>.01 for r in rows),
                no_reference_shadow_groups=sum(r['positive']==0 for r in rows))
    for dest,key in [('macro_ap','ap'),('macro_recall','recall'),('macro_precision','precision'),
                     ('macro_fpr','fpr'),('macro_miou_percent','miou_percent'),
                     ('macro_surface_fpr','surface_fpr'),('macro_cloud_fpr','cloud_fpr')]:
        result[dest]=interval([r[key] for r in rows if r[key] is not None])
    result['macro_iou_percent']=[interval([r['iou_percent'][i] for r in rows if r['iou_percent'][i] is not None]) for i in range(3)]
    result['no_shadow_macro_fpr']=interval([r['fpr'] for r in rows if not r['positive'] and r['fpr'] is not None])
    # Independent flat-count matrix and metrics, not production confusion helper.
    assert sum(map(sum,pooled['confusion']))==len(y)
    return result


def evaluate(prepared, models, reference):
    rows={v:{n:[] for n in NAMES} for v,_,_ in VERSIONS}
    pools={v:{n:[] for n in NAMES} for v,_,_ in VERSIONS};ys={v:[] for v,_,_ in VERSIONS};cpu=[]
    for row in prepared:
        product=row['product'];prob=old.probabilities(product)
        new,checks=aggregate_controls(row,models);prob.update(new)
        cpu.append(dict(product=product,block_checks=checks))
        pred=decisions(prob,old.locked()[2])
        for name,m in models.items():
            pred[name]=control_decision(prob[name],prob,m)
        for version,tkey,mkey in VERSIONS:
            truth=old.array(row,tkey);mask=old.array(row,mkey)
            if not mask.any():continue
            y=np.asarray(truth[mask]);ys[version].append(y)
            for name in NAMES:
                score=prob[name][...,2][mask] if name in old.NAMES[:2] else prob[name][mask]
                metric=old.row_metrics(y,pred[name][mask],score)
                independent_ap(y,score,metric['ap'])
                independent_cm=[[int(np.count_nonzero((y==i)&(pred[name][mask]==j))) for j in range(3)] for i in range(3)]
                assert metric['confusion']==independent_cm
                if name in old.NAMES:
                    ref=next(t for t in reference['versions'][version]['policies'][name]['rows'] if t['product']==product)
                    assert metric=={k:ref[k] for k in metric}
                rows[version][name].append(dict(product=product,group_id=row['group_id'],tile=row['tile'],**metric))
                pools[version][name].append(np.asarray(score,np.float32))
        print('Phase90 external controls',row['tile'],flush=True)
    result=dict(created_utc=old.now(),versions={},formal_confirmation=False,
                zero_training_fitting_threshold_tuning_network_forwards_downloads=True,
                human_B='not_executed',reference_is_not_certified_truth=True,
                support=reference['support'],reused_precrash_probability_SHA_limitation=True,
                paired_bootstrap_seed=SEED,paired_bootstrap_resamples=10000)
    tables=[];groupcsv=[];pairs_csv={n:[] for n in CONTROLS}
    for version,_,_ in VERSIONS:
        policies={};labels=np.concatenate(ys.pop(version))
        for name in NAMES:
            score=np.concatenate(pools[version].pop(name));q=summarize_policy(rows[version][name],score,labels);policies[name]=q
            if name in old.NAMES:
                ref=reference['versions'][version]['policies'][name]
                assert q['pooled']==ref['pooled']
                for key in ['macro_ap','macro_recall','macro_fpr','macro_miou_percent']:
                    assert q[key]['mean']==ref[key]['mean']
            po=q['pooled']
            tables.append(dict(version=version,policy=name,groups=q['groups'],ap_groups=q['macro_ap']['n_groups'],
                **{k:q[k]['mean'] for k in ['macro_ap','macro_recall','macro_precision','macro_fpr','macro_miou_percent','macro_surface_fpr','macro_cloud_fpr']},
                fpr_upper95=q['macro_fpr']['upper95_one_sided'],fpr_ci95=json.dumps(q['macro_fpr']['ci95']),
                **{'pooled_'+k:po[k] for k in ['ap','recall','precision','fpr','miou_percent','shadow_to_surface','shadow_to_cloud']},
                **{'pooled_'+n+'_iou_percent':po['iou_percent'][i] for i,n in enumerate(['surface','cloud','shadow'])},
                groups_above_1percent=q['groups_above_1percent'],zero_recall_groups=len(q['zero_recall_groups']),
                no_reference_shadow_groups=q['no_reference_shadow_groups'],no_shadow_macro_fpr=q['no_shadow_macro_fpr']['mean']))
            for r in q['rows']:groupcsv.append(dict(version=version,policy=name,**{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()}))
            print('Phase90 pooled CPU verified',version,name,flush=True)
        comparisons={}
        for name in CONTROLS:
            paired=[]
            for a,b in zip(policies['L3']['rows'],policies[name]['rows']):
                assert a['group_id']==b['group_id']
                pair=dict(version=version,product=a['product'],group_id=a['group_id'],tile=a['tile'],
                    delta_ap_pp=(a['ap']-b['ap'])*100 if a['ap'] is not None and b['ap'] is not None else None,
                    delta_recall_pp=(a['recall']-b['recall'])*100 if a['recall'] is not None and b['recall'] is not None else None,
                    delta_fpr_pp=(a['fpr']-b['fpr'])*100 if a['fpr'] is not None and b['fpr'] is not None else None,
                    delta_miou_pp=a['miou_percent']-b['miou_percent'])
                paired.append(pair);pairs_csv[name].append(pair)
            delta=[p['delta_ap_pp'] for p in paired if p['delta_ap_pp'] is not None]
            stat=interval(delta)
            stat.update(improved=sum(d>1 for d in delta),stable=sum(-1<=d<=1 for d in delta),declined=sum(d< -1 for d in delta),
                worst_AP=min((p for p in paired if p['delta_ap_pp'] is not None),key=lambda p:p['delta_ap_pp']),
                worst_recall=min((p for p in paired if p['delta_recall_pp'] is not None),key=lambda p:p['delta_recall_pp']),pairs=paired,
                comparison_role='primary' if name=='L2_RGB' else 'secondary',not_causal_single_coefficient_effect=True)
            comparisons['L3_minus_'+name]=stat
        result['versions'][version]=dict(policies=policies,comparisons=comparisons)
    old.write(DELIVERY/'phase90_report.json',result)
    old.csvwrite(DELIVERY/'seven_policy_tables.csv',tables)
    old.csvwrite(DELIVERY/'per_group.csv',groupcsv)
    old.csvwrite(DELIVERY/'common_support_tables.csv',[r for r in tables if r['version']!='original_main'])
    for name,rr in pairs_csv.items():old.csvwrite(DELIVERY/('paired_L3_minus_'+name+'.csv'),rr)
    old.write(DELIVERY/'CPU_VERIFIED.json',dict(created_utc=old.now(),self_tests=self_test(),
        per_group_confusion_and_independent_score_bin_AP_checks=len(groupcsv),pooled_AP_checks=len(tables),
        calibrated_block_checks=cpu,aggregation_atol=2e-15,
        original_four_policy_scores_confusion_AP_exact=True,all_new_decisions_independently_equal=True,
        sklearn_not_available_not_claimed=True,new_forward=0))
    # Preserve the original human wording and empty fields; never supply a second reviewer.
    for filename in ['human57_automatic_comparison.csv','reference_support.csv']:
        shutil.copyfile(old.ROOT/'delivery'/filename,DELIVERY/filename)
    shutil.copyfile(old.HUMAN,DELIVERY/'expert_A_final_original.json')
    shutil.copyfile(P67/'control_lock.json',DELIVERY/'phase67_control_lock.json')
    render(result)


def render(result):
    def f(x,scale=1):return 'NA' if x is None else f'{x*scale:.4f}'
    lines=['# Phase90：冻结输入对照的外部参考评价','','## 1．数据结果','',
        '固定Phase89的9组原版主分析与6组双版本共同支持；三个对照来自Phase67原冻结系数、标准化和阈值。零训练、拟合、调阈值、网络前向和下载。',
        '统计单位为相关组，10000次配对bootstrap、seed90012；全部区间仅描述，三比较不称正式多重检验通过。',
        '','| 原版9组 | 宏AP % | 池化AP % | 宏召回 % | 宏FPR % | FPR单侧95%上界 % | 宏mIoU % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,q in result['versions']['original_main']['policies'].items():
        vals=[f(q['macro_ap']['mean'],100),f(q['pooled']['ap'],100),f(q['macro_recall']['mean'],100),
              f(q['macro_fpr']['mean'],100),f(q['macro_fpr']['upper95_one_sided'],100),f(q['macro_miou_percent']['mean'])]
        lines.append('|'+name+'|'+'|'.join(vals)+'|')
    lines+=['','| 支持/比较（L3−对照） | 平均ΔAP pp | 95%描述区间 pp | 改善/稳定/下降 | 最差ΔAP pp / tile | 最差Δ召回 pp / tile |',
            '|---|---:|---|---|---|---|']
    for version,v in result['versions'].items():
        for name,q in v['comparisons'].items():
            ci='NA' if q['ci95'] is None else '['+', '.join(f(x) for x in q['ci95'])+']'
            lines.append(f"|{version}/{name}|{f(q['mean'])}|{ci}|{q['improved']}/{q['stable']}/{q['declined']}|{f(q['worst_AP']['delta_ap_pp'])}/{q['worst_AP']['tile']}|{f(q['worst_recall']['delta_recall_pp'])}/{q['worst_recall']['tile']}|")
    lines+=['','逐组原始差、混淆矩阵、三类IoU、漏影去向、Surface/Cloud分层FPR、零召回和支持数量完整保留于CSV及JSON。AP无支持时为NA，不填0。',
        '基线点估计与Phase89精确复现；本轮统一使用seed90012，故基线描述区间可与Phase89的seed89011略有差别。',
        '','## 2．结论分析','']
    for name in CONTROLS:
        qs=[result['versions'][v]['comparisons']['L3_minus_'+name] for v,_,_ in VERSIONS]
        main=qs[0]
        if main['ci95'] and main['ci95'][0]>0:message='原版主分析描述区间在零以上，支持相对该对照的参考一致性增量。'
        elif main['ci95'] and main['ci95'][1]<0:message='原版主分析描述区间在零以下，不支持L3优于该对照。'
        else:message='原版主分析区间跨零或支持不足，证据不确定，不能解释为等效。'
        message+=f" 同一6组支持上原版/修订版平均差分别为{f(qs[1]['mean'])}/{f(qs[2]['mean'])}pp。"
        if qs[1]['mean']*qs[2]['mean']<0:message+='两版方向相反，必须降低稳定增益的主张。'
        if name=='L2_RGB':message+='此比较检验额外光谱相对RGB的特异性。'
        elif name=='Source_spectral':message+='无明确优势时，不主张当前MsRE分数必需。'
        else:message+='无明确优势时，不主张Source分数必需。'
        lines.append(name+'：'+message)
    lines+=['','各对照系数曾分别拟合，不能将差值解释为单一输入的纯因果效应。Source_spectral使用Source非Shadow读出；AP/Shadow不受该读出影响，三类mIoU差包含读出差异。',
        '冻结1%误报预算不变；不选择本轮赢家部署，不在ALCD重校准，也不回填Phase89主要比较。',
        '公开参考版本不是经过全面裁决的真值；人工B未执行，原人工A及空字段原样保留。本池已经暴露，评价是事后探索而非独立确认。',
        'Phase89复用54块在故障前没有输出SHA封存；恢复时身份/结构检查和现有SHA不消除该历史限制。',
        '全部计算只读取既有20m概率/六波段TOA与允许的Catalogue development，不读取旧sealed像元、不改变原失败主检验。',
        '大缓存不复制、不新增保存、不同步work_dirs。']
    with (DELIVERY/'README.md').open('x',encoding='utf-8') as f0:f0.write('\n'.join(lines)+'\n')


def main():
    started=time.monotonic();prepared,models,ref,ref67=preflight()
    reproduce89(prepared,ref)
    reproduce67(models,ref67)
    evaluate(prepared,models,ref)
    old.write(DELIVERY/'DELIVERY_SHA256.json',{p.name:old.digest(p) for p in sorted(DELIVERY.iterdir()) if p.is_file()})
    old.write(ROOT/'COMPLETED.json',dict(created_utc=old.now(),elapsed_seconds=time.monotonic()-started,
        delivery_manifest_sha256=old.digest(DELIVERY/'DELIVERY_SHA256.json'),new_forwards=0,new_fits=0,new_downloads=0,
        no_work_dirs_transfer=True,formal_confirmation=False))
    print('Phase90 COMPLETED',flush=True)


if __name__=='__main__':main()

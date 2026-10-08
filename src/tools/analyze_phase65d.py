"""Development-only correction lock, then fixed ranking/operating-point diagnostics."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from prepare_phase65a_explore import digest
from phase65b_evidence_probe import authorized
from phase65d_metrics import curve,threshold_at_budget,confusion,measures,corrected,score_summary,FPRS

T18='S2A_MSIL1C_20180803T141051_N0206_R110_T18FYG_20180803T192500'


def indices(root,cohort):
    paths=[root/cohort/('index_'+m+'.json') for m in ['source','msre']]
    data=[json.loads(p.read_text()) for p in paths]
    rows=[{r['product']:r for r in d['rows']} for d in data]
    if rows[0].keys()!=rows[1].keys() or data[0]['manifest_sha256']!=data[1]['manifest_sha256']:raise ValueError('Unpaired score cache')
    for product in rows[0]:
        if rows[0][product]['truth_sha256']!=rows[1][product]['truth_sha256'] or rows[0][product]['group_id']!=rows[1][product]['group_id']:
            raise ValueError('Truth/group pairing changed')
    return rows,paths


def load(root,row,key):
    if key in ['truth','brightness']:path=row[key+'_path']
    else:path=row['files'][key]['path']
    return np.load(root/path,mmap_mode='r')


def full_measure(root,rows,mode,threshold=None):
    cms=[];per_scene=[]
    for r in rows.values():
        truth=load(root,r,'truth')
        pred=load(root,r,'pred') if mode=='original' else corrected(load(root,r,mode),threshold,load(root,r,'nonshadow'))
        cm=confusion(truth,pred);cms.append(cm)
        per_scene.append(dict(product=r['product'],group_id=r['group_id'],**measures(cm)))
    result=measures(np.sum(cms,axis=0));positive=[r['recall'] for r in per_scene if r['recall'] is not None]
    result.update(scene_macro_recall=float(np.mean(positive)),shadow_supported_scenes=len(positive),per_scene=per_scene)
    return result


def freeze(root):
    lock=root/'calibration_lock.json'
    if lock.exists():raise FileExistsError('Never refit or overwrite calibration')
    if (root/'phase65b_evidence').exists():raise ValueError('65b extraction must follow lock')
    rows,paths=indices(root,'development_val');assert len(rows[0])==32
    source=full_measure(root,rows[0],'original');uncorrected=full_measure(root,rows[1],'original')
    budget=source['false_positive'];candidates=[]
    for name in ['score','margin']:
        negatives=np.concatenate([load(root,r,name)[(load(root,r,'truth')!=2)&(load(root,r,'truth')!=255)] for r in rows[1].values()])
        threshold=threshold_at_budget(negatives,budget);del negatives
        stats=full_measure(root,rows[1],name,threshold)
        candidates.append(dict(name=name,threshold=threshold,metrics=stats,
                               feasible=stats['false_positive']<=budget and stats['miou_percent']>=source['miou_percent']))
    original=dict(name='original',threshold=None,metrics=uncorrected,
                  feasible=uncorrected['false_positive']<=budget and uncorrected['miou_percent']>=source['miou_percent'])
    candidates.append(original);eligible=[c for c in candidates if c['feasible']]
    # Largest macro recall; deterministic tie preference original, then probability, then margin.
    tie={'original':2,'score':1,'margin':0}
    selected=max(eligible,key=lambda c:(c['metrics']['scene_macro_recall'],tie[c['name']])) if eligible else original
    report=dict(status='frozen_development_only',development_scenes=32,
        selection_rule='Maximize development scene-macro shadow recall, pooled false positives <= Source original, pooled mIoU >= Source original; no heldout selection',
        original_source=source,original_msre=uncorrected,source_fpr_budget=source['fpr'],source_false_positive_budget=budget,
        candidates=candidates,selected_name=selected['name'],selected_threshold=selected['threshold'],
        restores_source_macro_recall_on_development=selected['metrics']['scene_macro_recall']>=source['scene_macro_recall'],
        threshold_meaning='Fixed global decision baseline, not probability calibration or semantic label validation',
        input_hashes={str(p.relative_to(root)):digest(p) for p in paths},phase65c_authorized=False)
    lock.write_text(json.dumps(report,indent=2,allow_nan=False))
    print('calibration frozen',report['selected_name'],report['selected_threshold'],flush=True)


def summarize(root):
    lock_path=root/'calibration_lock.json';lock=json.loads(lock_path.read_text())
    if (root/'diagnostic_report.json').exists():raise FileExistsError('Preserve diagnostics')
    cohort_report={};exports=[];plot_data={};table=[]
    for cohort in ['development_val','phase65b_evidence']:
        rows,paths=indices(root,cohort)
        if cohort=='phase65b_evidence':
            for p in paths:
                if json.loads(p.read_text())['calibration_lock_sha256']!=digest(lock_path):raise ValueError('Lock changed after heldout access')
        result={};scene_stats={}
        for mi,model in enumerate(['source','msre']):
            original=full_measure(root,rows[mi],'original');model_result={'original':original,'ranking':{}}
            for score in ['score','native_score','margin']:
                labels=[];scores=[];individual=[]
                for product,r in rows[mi].items():
                    truth=load(root,r,'truth');valid=truth!=255;y=truth[valid]==2;s=load(root,r,score)[valid]
                    c=curve(s,y);individual.append(dict(product=product,group_id=r['group_id'],**{k:v for k,v in c.items() if k not in ['precision','recall','fpr','thresholds']}))
                    labels.append(y);scores.append(s)
                    table.append(dict(cohort=cohort,model=model,score=score,product=product,positive=c['positive'],negative=c['negative'],ap=c['ap'],roc_auc=c['roc_auc'],recall_at_fpr_001=c['matched_fpr'].get('0.01',{}).get('recall')))
                pooled=curve(np.concatenate(scores),np.concatenate(labels));del labels,scores
                macro=[r['ap'] for r in individual if r['ap'] is not None]
                model_result['ranking'][score]=dict(pooled={k:v for k,v in pooled.items() if k not in ['precision','recall','fpr','thresholds']},
                    scene_macro_ap=float(np.mean(macro)),supported_scenes=len(macro),per_scene=individual)
                plot_data[cohort+'_'+model+'_'+score]=pooled
                scene_stats[model+'_'+score]=individual
                print('ranked',cohort,model,score,flush=True)
            if model=='msre':model_result['frozen_correction']=full_measure(root,rows[mi],lock['selected_name'],lock['selected_threshold'])
            result[model]=model_result
        # Scene bootstrap, not pixel bootstrap; curves at fixed FPR are descriptive oracle diagnostics.
        source={r['product']:r for r in scene_stats['source_score']};msre={r['product']:r for r in scene_stats['msre_score']}
        supported=[p for p in source if source[p]['ap'] is not None and msre[p]['ap'] is not None]
        paired={}
        for rate in FPRS:
            delta=np.array([msre[p]['matched_fpr'][str(rate)]['recall']-source[p]['matched_fpr'][str(rate)]['recall'] for p in supported])
            rng=np.random.default_rng(65);boot=np.mean(delta[rng.integers(0,len(delta),(10000,len(delta)))],axis=1)
            paired[str(rate)]=dict(scene_macro_delta_recall=float(delta.mean()),scene_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),supported_scenes=len(delta))
        result['matched_fpr_scene_comparison']=paired
        cohort_report[cohort]=result
        for model,row_map in zip(['source','msre'],rows):
            if T18 in row_map:
                r=row_map[T18];truth=load(root,r,'truth');brightness=load(root,r,'brightness');valid=truth!=255
                t18=dict(product=T18,original=measures(confusion(truth,load(root,r,'pred'))),ranking={},distributions={})
                if model=='msre':t18['frozen_correction']=measures(confusion(truth,load(root,r,'pred') if lock['selected_name']=='original' else corrected(load(root,r,lock['selected_name']),lock['selected_threshold'],load(root,r,'nonshadow'))))
                for name in ['score','native_score','margin']:
                    scores=load(root,r,name);t18['ranking'][name]=curve(scores[valid],truth[valid]==2)
                    strata={'shadow':valid&(truth==2),'nonshadow':valid&(truth!=2),'dark_surface':valid&(truth==0)&(brightness<.08)}
                    t18['distributions'][name]={n:score_summary(scores[m]) for n,m in strata.items()}
                    dark=valid&((truth==2)|((truth==0)&(brightness<.08)))
                    if name=='score':t18['shadow_vs_dark_surface_ranking']=curve(scores[dark],truth[dark]==2)
                plot_data['t18_'+model]=t18
    t18={m:{k:v for k,v in plot_data['t18_'+m].items() if k!='ranking'} for m in ['source','msre']}
    for m in t18:
        t18[m]['ranking']={name:{k:v for k,v in c.items() if k not in ['precision','recall','fpr','thresholds']} for name,c in plot_data['t18_'+m]['ranking'].items()}
        t18[m]['shadow_vs_dark_surface_ranking']={k:v for k,v in t18[m]['shadow_vs_dark_surface_ranking'].items() if k not in ['precision','recall','fpr','thresholds']}
    report=dict(status='complete_fixed_phase65d_diagnostic',calibration_lock_sha256=digest(lock_path),cohorts=cohort_report,t18=t18,
        interpretation='Development selection is optimistic; 65b previously inspected and T18 chosen after observed failure, neither is new independent confirmation. Ranking FPR thresholds on evaluation labels are diagnostic oracle points, not deployable tuned thresholds.',
        score_definition='Main: softmax of three aggregated Mask2Former class-mask scores. Secondary: raw shadow aggregate and shadow-minus-best-other margin. None asserted calibrated correctness probabilities.',
        fpr_definition='False shadow / valid reference nonshadow (surface plus cloud), not 1-precision',
        pr_definition='Exact tied-score Average Precision; all valid pixels, no pixel subsampling. Curves shown at deterministic subset of thresholds.',
        raw_scores_kept=str(root),calibration=lock,phase65c_authorized=False)
    (root/'diagnostic_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (root/'plot_data.json').write_text(json.dumps(plot_data,allow_nan=False))
    with (root/'scene_ranking.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print('65d diagnostics complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','summarize']);p.add_argument('--root',required=True)
    a=p.parse_args();root=authorized(a.root);freeze(root) if a.action=='freeze' else summarize(root)

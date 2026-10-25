"""Independent post-seal reference analysis; no fit or neural imports/calls."""
import gc
import io
import sys
import zipfile
import traceback
import numpy as np
from phase312_common import *
import analyze_phase304 as legacy
import auc_phase306 as ranking

INPUTS={}
def pin(p,expected=None):
    h=sha(p);assert expected is None or h==expected,str(p);INPUTS[str(p)]=h
    return load(p) if Path(p).suffix=='.json' else h
def metric(y,p,base):
    mat=np.bincount((3*y+p).astype(np.int64),minlength=9).reshape(3,3)
    assert np.array_equal(mat,legacy.confusion_matrix(y,p,labels=[0,1,2]))
    iou=legacy.ious(mat)
    for c in range(3):
        union=int(((y==c)|(p==c)).sum())
        assert not union or abs(float(iou[c])-int(((y==c)&(p==c)).sum())/union)<1e-12
    return dict(support=len(y),confusion=mat.tolist(),union_classes=int(np.isfinite(iou).sum()),mIoU_pct=100*float(np.nanmean(iou)),Surface_IoU_pct=100*float(iou[0]) if np.isfinite(iou[0]) else None,Cloud_IoU_pct=100*float(iou[1]) if np.isfinite(iou[1]) else None,Shadow_IoU_pct=100*float(iou[2]) if np.isfinite(iou[2]) else None,AP_pct=base['AP_pct'],recall_pct=base['recall_pct'],FPR_pct=base['FPR_pct'])
def main():
    assert 'torch' not in sys.modules
    seal=pin(ROOT/'READOUT_LOCK.json');assert seal['status']=='sealed' and time.time()>seal['sealed_epoch']
    assert sha(ROOT/'FIT_LOCK.json')==seal['fit_lock_SHA'] and sha(ROOT/'PREDICTION_LOCK.json')==seal['prediction_lock_SHA']
    fitlock=pin(ROOT/'FIT_LOCK.json');prediction=pin(ROOT/'PREDICTION_LOCK.json');pin(ROOT/'sampling.npz',fitlock['sampling_SHA'])
    for n,h in fitlock['models'].items():pin(ROOT/'models'/(n+'.json'),h)
    stageplan=Path('/home/scv/shared/phase300_20261020');plan={r['product_id']:r for r in pin(stageplan/'EVALUATION_INPUT_PLAN.json')['rows'] if r['pool']=='Catalogue'}
    support={r['product_id']:r for r in pin(stageplan/'EXPECTED_SUPPORT.json')['rows'] if r['pool']=='Catalogue'}
    masks=R/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip';pin(masks)
    tags=read(R/'data/sentinel2_cloud_mask_catalogue_4172871/classification_tags.csv');pin(R/'data/sentinel2_cloud_mask_catalogue_4172871/classification_tags.csv')
    snow={r['scene']:r['snow/ice'] if r.get('snow/ice') in ('0','1') else 'missing' for r in tags}
    measures=[];selections=[];ranks=[];flow=[];cells=[];parts={'Source':[],'Q90-PL':[]};truths=[];rankparts=[];pooled={};globalD=globalB=0
    with zipfile.ZipFile(masks) as archive:
        for row in seal['rows']:
            pid=row['product'];gid=row['group_id'];scen=snow.get(pid,'missing');p=Path('/home/scv/shared/phase205_20261019/infer/R4/scores/Catalogue')/(pid+'.npz');pin(p)
            with np.load(p) as z:ids0=z['flat_indices'];v0=z['valid']
            assert ah(ids0)==support[pid]['flat_indices_sha256'] and ah(v0)==support[pid]['common_valid_sha256']
            common=np.zeros(1022**2,bool);common[ids0[v0]]=True;assert int(common.sum())==support[pid]['count'];common=common.reshape(1022,1022)
            raw=archive.read('masks/'+pid+'.npy');assert hashlib.sha256(raw).hexdigest()==plan[pid]['group']['mask_member_sha256']
            m=np.load(io.BytesIO(raw));assert (m.sum(-1)[common]==1).all();y=m.argmax(-1).astype(np.uint8)
            pin(row['source_file'],row['source_SHA']);pin(row['student_file'],row['student_SHA']);pin(row['prediction_file'],row['sha256']);pin(row['readout_file'],row['readout_SHA'])
            with np.load(row['source_file']) as z:source=z['pred'];sv=z['shadow'];sourcevalid=z['valid'];assert np.array_equal(source,z['scores'].argmax(0))
            with np.load(row['student_file']) as z:pl=z['pred'];shadow=z['shadow'];plvalid=z['valid'];assert np.array_equal(pl,z['scores'].argmax(0)) and ah(shadow)==row['Shadow_score_SHA']
            assert np.array_equal(sourcevalid,plvalid) and not (common&~plvalid).any()
            base=legacy.metric(y[common],pl[common],shadow[common]);src=legacy.metric(y[common],source[common],sv[common])
            truths.append(y[common]);parts['Source'].append((source[common],sv[common]));parts['Q90-PL'].append((pl[common],shadow[common]))
            for name,pred,mt in [('Source',source,src),('Q90-PL',pl,base)]:measures.append(dict(strategy=name,product=pid,group_id=gid,snow_ice=scen,**metric(y[common],pred[common],mt)))
            D=common&(y==0)&(source==0)&(pl==1);B=common&(y==1)&(source==2)&(pl==1);globalD+=int(D.sum());globalB+=int(B.sum())
            with np.load(row['prediction_file']) as z,np.load(row['readout_file']) as rz:
                ids=z['flat_indices'];assert np.array_equal(ids,rz['flat_indices']);P=np.unpackbits(z['P_pack'],count=1022**2).reshape(1022,1022).astype(bool);V6=np.unpackbits(z['V6_pack'],count=1022**2).reshape(1022,1022).astype(bool)
                assert np.array_equal(ids,np.flatnonzero(P&V6));candidate=np.unpackbits(rz['C_pack'],count=len(ids)).astype(bool);assert np.array_equal(candidate,pl.ravel()[ids]==1)
                cids=ids[candidate];cy=y.ravel()[cids];cv=common.ravel()[cids];cc=cy[cv];Dn=int((cc==0).sum());Kn=int((cc==1).sum());Zn=int((cc==2).sum());assert Dn+Kn+Zn==int(cv.sum())
                assert not (B&P).any() and not ((source!=0)&P).any()
                for name in list(fitlock['successful_models'])+['Source_qSurface']:
                    score=z['qSurface'] if name=='Source_qSurface' else z['r_'+name];values=score[candidate][cv];binary=cc!=2;auc=ranking.auc(cc[binary]==0,values[binary])
                    ranks.append(dict(score=name,product=pid,group_id=gid,snow_ice=scen,D=Dn,K=Kn,Z_excluded=Zn,AUC=auc,NA_reason='no D or no K' if auc is None else None))
                    rankparts.append(dict(score=name,product=pid,snow_ice=scen,y=cc[binary]==0,score_values=values[binary]))
                ms6_n=None
                if 'MS6_90' in row['rules']:ms6_n=int(np.unpackbits(rz['MS6_90'],count=len(ids)).sum())
                for rule in row['rules']:
                    selected=np.unpackbits(rz[rule],count=len(ids)).astype(bool);assert not (selected&~candidate).any()
                    if rule.endswith('equalMS6'):assert int(selected.sum())==ms6_n
                    chosen=ids[selected];chosen_common=chosen[common.ravel()[chosen]];truth_selected=y.ravel()[chosen_common];n=[int((truth_selected==c).sum()) for c in range(3)]
                    result=pl.copy();result.ravel()[chosen]=0
                    assert np.array_equal(result==2,pl==2) and np.all((result==pl)|((pl==1)&(result==0)))
                    assert not B.ravel()[chosen_common].any() and int(D.ravel()[chosen_common].sum())==n[0]
                    selections.append(dict(rule=rule,product=pid,group_id=gid,snow_ice=scen,P_image=int(P.sum()),P_V6_image=len(ids),C_image=len(cids),selected_image=len(chosen),C_reference=Dn+Kn+Zn,D=Dn,K=Kn,Z=Zn,selected_D=n[0],selected_K=n[1],selected_Z=n[2],unselected_D=Dn-n[0],unselected_K=Kn-n[1],unselected_Z=Zn-n[2],D_repair_fraction=n[0]/Dn if Dn else None,K_revoke_fraction=n[1]/Kn if Kn else None,original_D=int(D.sum()),original_D_repaired=n[0],original_D_remaining=int(D.sum())-n[0],original_B=int(B.sum()),original_B_retained=int(B.sum()),original_B_revoked=0,outside_C_D=int(D.sum())-Dn,changed_reference=n[0]+n[1]+n[2],Shadow_binary_and_score_unchanged=True))
                    if rule!='Q90-PL':measures.append(dict(strategy=rule,product=pid,group_id=gid,snow_ice=scen,**metric(y[common],result[common],base)))
                    t=np.bincount((9*y[common]+3*pl[common]+result[common]).astype(np.int64),minlength=27).reshape(3,3,3)
                    assert np.array_equal(t.sum(2),np.asarray(base['confusion'])) and np.array_equal(t.sum(1),np.asarray(metric(y[common],result[common],base)['confusion']))
                    assert [int(t[c,1,0]) for c in range(3)]==n and int(t.sum()-sum(t[c,i,i] for c in range(3) for i in range(3)))==sum(n)
                    pooled.setdefault(rule,np.zeros((3,3,3),np.int64));pooled[rule]+=t
                    pre=dict(product=pid,group_id=gid,comparison='Q90-PL->'+rule);flow+=legacy.events(t,pre);cells+=legacy.cells(t,pre)
            del m
            gc.collect();print('reference group',pid,flush=True)
    assert globalD==506798 and globalB==784478
    summaries=[];yy=np.concatenate(truths);baseline_pooled={}
    for name in ('Source','Q90-PL'):
        pred=np.concatenate([p for p,s in parts[name]]);scores=np.concatenate([s for p,s in parts[name]]);baseline_pooled[name]=legacy.metric(yy,pred,scores)
    for name in sorted({r['strategy'] for r in measures}):
        rows=[r for r in measures if r['strategy']==name];assert len(rows)==32
        mat=sum(np.asarray(r['confusion']) for r in rows);iou=legacy.ious(mat);bp=baseline_pooled['Source' if name=='Source' else 'Q90-PL']
        summaries.append(dict(strategy=name,groups=32,AP_groups=sum(r['AP_pct'] is not None for r in rows),group_mean_mIoU_pct=average(r['mIoU_pct'] for r in rows),pooled_mIoU_pct=float(np.nanmean(iou))*100,Surface_IoU_pct=float(iou[0])*100,Cloud_IoU_pct=float(iou[1])*100,Shadow_IoU_pct=float(iou[2])*100,macro_AP_pct=average(r['AP_pct'] for r in rows),pooled_AP_pct=bp['AP_pct'],macro_recall_pct=average(r['recall_pct'] for r in rows),macro_FPR_pct=average(r['FPR_pct'] for r in rows),max_group_FPR_pct=max(r['FPR_pct'] for r in rows if r['FPR_pct'] is not None),confusion=mat.tolist()))
    prior=read(PL/'analysis/FIXED_RESULTS.csv');pin(PL/'analysis/FIXED_RESULTS.csv')
    for name,oldname in [('Source','Source'),('Q90-PL','LoRA')]:
        obs=next(r for r in summaries if r['strategy']==name);ref=next(r for r in prior if r['strategy']==oldname)
        for k in obs:
            if k.endswith('_pct'):assert abs(obs[k]-float(ref['AP_pct' if k=='pooled_AP_pct' else k]))<1e-6,k
    for rule,t in pooled.items():
        pre=dict(product='ALL',group_id='ALL',comparison='Q90-PL->'+rule);flow+=legacy.events(t,pre);cells+=legacy.cells(t,pre)
        records=[r for r in selections if r['rule']==rule];numeric=[k for k,v in records[0].items() if isinstance(v,int) and not isinstance(v,bool)]
        values={k:sum(r[k] for r in records) for k in numeric}
        selections.append(dict(rule=rule,product='ALL',group_id='ALL',snow_ice='ALL',**values,D_repair_fraction=values['selected_D']/values['D'] if values['D'] else None,K_revoke_fraction=values['selected_K']/values['K'] if values['K'] else None,Shadow_binary_and_score_unchanged=True))
    # Preserve column ordering when merging pooled numeric rows.
    order=list(selections[0]);selections=[{k:r[k] for k in order} for r in selections]
    ranking_summary=[];strata=[]
    for score in sorted({r['score'] for r in ranks}):
        for scene in ('ALL','0','1','missing'):
            rs=[r for r in ranks if r['score']==score and (scene=='ALL' or r['snow_ice']==scene)];ps=[r for r in rankparts if r['score']==score and (scene=='ALL' or r['snow_ice']==scene)]
            ys=np.concatenate([r['y'] for r in ps]) if ps else np.array([],bool);ss=np.concatenate([r['score_values'] for r in ps]) if ps else np.array([],np.float64)
            ranking_summary.append(dict(score=score,snow_ice=scene,total_groups=len(rs),AUC_groups=sum(r['AUC'] is not None for r in rs),D=sum(r['D'] for r in rs),K=sum(r['K'] for r in rs),Z_excluded=sum(r['Z_excluded'] for r in rs),group_mean_AUC=average(r['AUC'] for r in rs),pooled_AUC=ranking.auc(ys,ss)))
    for rule in pooled:
        for scene in ('0','1','missing'):
            rs=[r for r in selections if r['rule']==rule and r['product']!='ALL' and r['snow_ice']==scene]
            strata.append(dict(rule=rule,snow_ice=scene,groups=len(rs),D=sum(r['D'] for r in rs),K=sum(r['K'] for r in rs),Z=sum(r['Z'] for r in rs),selected_D=sum(r['selected_D'] for r in rs),selected_K=sum(r['selected_K'] for r in rs),selected_Z=sum(r['selected_Z'] for r in rs)))
    table(ROOT/'METRICS.csv',summaries);table(ROOT/'PER_GROUP_METRICS.csv',measures);table(ROOT/'SELECTION.csv',selections);table(ROOT/'RANKING.csv',ranks);table(ROOT/'RANKING_SUMMARY.csv',ranking_summary);table(ROOT/'SCENE_STRATA.csv',strata);table(ROOT/'ERROR_FLOW.csv',flow);table(ROOT/'TRANSITIONS27.csv',cells)
    validity=[]
    for stage in ('fit','predict'):
        for line in (ROOT/(stage+'_IMAGE_ACCESS.jsonl')).read_text().splitlines():validity.append(dict(pool=stage,**json.loads(line)))
    table(ROOT/'SIX_BAND_VALIDITY.csv',validity)
    concentration=[]
    for rule in pooled:
        ordered=sorted([r for r in selections if r['rule']==rule and r['product']!='ALL'],key=lambda r:(-r['selected_D'],r['product']));total=sum(r['selected_D'] for r in ordered)
        for k in (1,3,5):concentration.append(dict(rule=rule,top_k=k,products=[r['product'] for r in ordered[:k]],D_repaired=sum(r['selected_D'] for r in ordered[:k]),all_D_repaired=total,fraction=sum(r['selected_D'] for r in ordered[:k])/total if total else None,post_hoc_descriptive_not_training_selection=True))
    table(ROOT/'REPAIR_CONCENTRATION.csv',concentration)
    top=('T25XEG','T40XDR','T35RPM');table(ROOT/'FIXED_TOP3_METRICS.csv',[r for r in measures if any(k in r['product'] for k in top)]);table(ROOT/'FIXED_TOP3_SELECTION.csv',[r for r in selections if any(k in r['product'] for k in top)])
    write(ROOT/'ANALYSIS_INPUT_SHA256.json',INPUTS)
    write(ROOT/'CPU_ACCEPTANCE.json',dict(status='passed',groups=32,strategies=len(summaries),zero_neural_calls=True,zero_neural_training=True,fit_attempts=fitlock['fit_attempts'],successful_models=fitlock['successful_models'],original_D=globalD,original_B=globalB,all_Shadow_binary_and_scores_identical_to_PL=True,all_equal_count_controls_exact=True,only_C_image_Cloud_to_Surface=True,reference_invalid_excluded_only_at_analysis=True,Source_PL_metrics_reproduced=True,original_B_disjoint_P_and_fully_retained=True,AUC_independent_rank_checks=ranking.CHECKS,post_seal_reference_only=True,torch_imported=False))
    audit_fit_labels(masks)
    acceptance=load(ROOT/'CPU_ACCEPTANCE.json');acceptance['fit_label_audit_status']=load(ROOT/'FIT_LABEL_AUDIT_STATUS.json')['status'];write(ROOT/'CPU_ACCEPTANCE.json',acceptance)
    from report_phase312 import report
    report()
    assert 'torch' not in sys.modules

def audit_fit_labels(masks):
    try:
        metadata=Path('/home/scv/shared/phase208_20261020/12201_FIT_INPUT_LOCK.json');bindings={r['product']:r for r in load(metadata)['bindings']};rows=sorted(load(ROOT/'FIT_IMAGES.json')['rows'],key=lambda r:r['product'])
        with np.load(ROOT/'sampling.npz') as z:ids=z['flat_indices'];group=z['product_index'];y=z['y_Surface']
        records=[]
        with zipfile.ZipFile(masks) as archive:
            for i,row in enumerate(rows):
                choose=group==i;flat=ids[choose];pseudo=np.where(y[choose]==1,0,1)
                raw=archive.read('masks/'+row['product']+'.npy');assert hashlib.sha256(raw).hexdigest()==bindings[row['product']]['mask_member_sha256']
                m=np.load(io.BytesIO(raw)).reshape(-1,3)[flat];valid=m.sum(-1)==1;truth=m.argmax(-1)
                mat=np.bincount((3*truth[valid]+pseudo[valid]).astype(np.int64),minlength=9).reshape(3,3)
                records.append(dict(product=row['product'],sampled=len(flat),reference_valid=int(valid.sum()),excluded_reference=int((~valid).sum()),confusion_truth_by_pseudo=mat.tolist(),Surface_pseudo_precision=mat[0,0]/mat[:,0].sum() if mat[:,0].sum() else None,Cloud_pseudo_precision=mat[1,1]/mat[:,1].sum() if mat[:,1].sum() else None,post_seal_only=True))
        table(ROOT/'FIT_LABEL_AUDIT.csv',records);write(ROOT/'FIT_LABEL_AUDIT_STATUS.json',dict(status='passed',groups=len(records),input_metadata_SHA=sha(metadata),used_for_refit=False))
    except Exception:write(ROOT/'FIT_LABEL_AUDIT_STATUS.json',dict(status='failed',primary_32_group_analysis_preserved=True,traceback=traceback.format_exc(),retry=False))

if __name__=='__main__':
    try:main()
    except BaseException:append('FAILURES.jsonl',dict(stage='analyze',traceback=traceback.format_exc()));raise

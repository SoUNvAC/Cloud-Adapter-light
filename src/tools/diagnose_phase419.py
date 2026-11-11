"""CPU-only diagnosis of sealed confusion matrices and saved shadow scores."""
from pathlib import Path
import json, csv, hashlib, time, traceback, itertools
import numpy as np
from scipy.ndimage import distance_transform_edt

R=Path('/home/scv/shared/phase419_20261101')
SHARED=R.parent
START=time.monotonic()
NAMES=['surface','cloud','shadow']
INPUTS={}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()

def read(p):
    p=Path(p);INPUTS[str(p)]=sha(p)
    return json.loads(p.read_text())

def save(n,v):
    (R/n).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')

def table(n,rows):
    with (R/n).open('w',newline='',encoding='utf-8') as f:
        keys=list(dict.fromkeys(k for row in rows for k in row))
        w=csv.DictWriter(f,keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def ious(cm):
    c=np.asarray(cm,dtype=np.int64);tp=np.diag(c);u=c.sum(0)+c.sum(1)-tp
    return np.divide(100.*tp,u,out=np.full(3,np.nan),where=u!=0)

def miou(cm):return float(np.nanmean(ious(cm)))

def cohort(support):
    return '+'.join(NAMES[i] for i in range(3) if support[i]>0)

def main():
    lock=read(R/'LOCK.json')
    for n,h in lock['files'].items():assert sha(R/n)==h
    for n,h in lock['references'].items():assert sha(n)==h
    toy=np.array([[8,1,1],[1,8,1],[1,1,8]])
    assert abs(miou(toy)-100*8/12)<1e-12
    specs=[('Source',SHARED/'phase400_20261031','evaluations/Source/METRICS.json'),
           ('Phase411',SHARED/'phase411_20261101/layerwise','evaluations/M/step_1000/METRICS.json'),
           ('Phase416',SHARED/'phase416_20261101/aux_separate_clip','evaluations/M/step_1000/METRICS.json')]
    for phase in [417,418]:
        root=SHARED/f'phase{phase}_20261101'
        complete=read(root/'COMPLETE.json');assert complete['passed']
        accepted=read(root/'INDEPENDENT_ACCEPTANCE.json');assert accepted['passed']
        specs += [(m,root/m,'evaluations/M/step_1000/METRICS.json') for m in complete['reports']]
    reports={};roots={};groupmaps={};totals={}
    for model,root,rel in specs:
        rep=read(root/rel);reports[model]=rep;roots[model]=root
        groups={g['product']:g for g in rep['per_group']};assert len(groups)==32
        groupmaps[model]=groups;cm=np.sum([g['confusion'] for g in groups.values()],axis=0)
        assert cm.tolist()==rep['confusion'] and abs(miou(cm)-rep['mIoU'])<1e-9
        totals[model]=cm
    base='Phase416';basecm=totals[base];products=list(groupmaps[base])
    for model in reports:
        assert set(groupmaps[model])==set(products)
        for product in products:
            g=groupmaps[model][product];ref=groupmaps[base][product]
            assert g['group_id']==ref['group_id']
            assert np.array_equal(np.sum(g['confusion'],axis=1),np.sum(ref['confusion'],axis=1))
    rows=read(SHARED/'phase400_20261031/INPUT_LOCK.json')['target_rows']
    meta={r['product']:r for r in rows if r['split']=='development_val'};assert len(meta)==32
    classes=[];channels=[];oracles=[];scene=[];strata=[];modelrows=[]
    for model,cm in totals.items():
        ii=ious(cm);support=cm.sum(1);pred=cm.sum(0);tp=np.diag(cm)
        modelrows.append({'model':model,'mIoU':miou(cm),'delta_vs_Phase416':miou(cm)-miou(basecm),'IoU':ii.tolist(),'confusion':cm.tolist()})
        for i in range(3):
            classes.append({'model':model,'class':NAMES[i],'support':int(support[i]),'pixel_share_pct':float(100*support[i]/support.sum()),'TP':int(tp[i]),'FN':int(support[i]-tp[i]),'FP':int(pred[i]-tp[i]),'IoU':float(ii[i]),'precision':float(100*tp[i]/pred[i]),'recall':float(100*tp[i]/support[i]),'contribution_to_mIoU_gap_from_100_pp':float((100-ii[i])/3),'contribution_to_delta_vs_Phase416_pp':float((ii[i]-ious(basecm)[i])/3)})
        for a,b in itertools.permutations(range(3),2):
            channels.append({'model':model,'truth':NAMES[a],'prediction':NAMES[b],'pixels':int(cm[a,b]),'within_truth_pct':float(100*cm[a,b]/support[a]),'share_of_all_errors_pct':float(100*cm[a,b]/(cm.sum()-tp.sum())),'delta_pixels_vs_Phase416':int(cm[a,b]-basecm[a,b])})
            fixed=cm.copy();fixed[a,a]+=fixed[a,b];fixed[a,b]=0
            oracles.append({'model':model,'repair_truth':NAMES[a],'repair_prediction':NAMES[b],'oracle_mIoU':miou(fixed),'oracle_gain_pp':miou(fixed)-miou(cm),'note':'GT oracle sensitivity only; gains not additive, not achievable performance'})
        buckets={}
        for product,g in groupmaps[model].items():
            gc=np.asarray(g['confusion']);bc=np.asarray(groupmaps[base][product]['confusion']);supp=gc.sum(1);key=cohort(supp)
            buckets.setdefault(key,[]).append(gc)
            fixed=cm-gc+bc
            perfect=cm-gc+np.diag(supp)
            scene.append({'model':model,'product':product,'group_id':g['group_id'],'cohort':key,'scene_mIoU_present_union_classes':miou(gc),'delta_errors_vs_Phase416':int((gc.sum()-np.trace(gc))-(bc.sum()-np.trace(bc))),'pooled_gain_if_scene_replaced_by_Phase416_pp':miou(fixed)-miou(cm),'pooled_oracle_gain_if_scene_perfect_pp':miou(perfect)-miou(cm),'confusion':gc.tolist(),'delta_confusion_vs_Phase416':(gc-bc).tolist(),'brightness_mean':meta[product].get('observable_spectra',{}).get('brightness_mean'),'dark_fraction':meta[product].get('observable_spectra',{}).get('dark_fraction')})
        for key,mats in buckets.items():
            c=np.sum(mats,axis=0);ii=ious(c)
            strata.append({'model':model,'GT_class_presence':key,'groups':len(mats),'valid_pixels':int(c.sum()),'pooled_mIoU_present_union_classes':miou(c),'IoU':[None if np.isnan(v) else float(v) for v in ii],'confusion':c.tolist(),'note':'subset mIoU can have different active class set; not directly comparable across strata'})
    for name,data in [('MODELS.csv',modelrows),('CLASS_DIAGNOSIS.csv',classes),('CONFUSION_CHANNELS.csv',channels),('ORACLE_SENSITIVITY.csv',oracles),('SCENE_IMPACT.csv',scene),('GT_STRATA.csv',strata)]:table(name,data)
    # Score strata are diagnostic masks based on GT and RGB, never routing inputs.
    score_rows=[];acc={};verified=0
    for index,product in enumerate(products):
        g=groupmaps[base][product];path=roots[base]/g['score_file']
        assert sha(path)==g['score_SHA'];INPUTS[str(path)]=g['score_SHA']
        with np.load(path,allow_pickle=False) as z:truth=z['truth'].copy()
        valid=truth!=255;assert np.bincount(truth[valid],minlength=3).tolist()==np.sum(g['confusion'],axis=1).tolist()
        rp=Path(meta[product]['rgb_path']);assert sha(rp)==meta[product]['prepared_sha256']['_rgb.npy'];INPUTS[str(rp)]=sha(rp)
        rgb=np.load(rp,allow_pickle=False);assert rgb.shape==truth.shape+(3,) and np.isfinite(rgb).all()
        bright=rgb.mean(2);masks={NAMES[c]:truth==c for c in range(3)}
        # Boundaries only between two observed, differently labelled pixels.
        edge=np.zeros_like(valid)
        diff=valid[:-1]&valid[1:]&(truth[:-1]!=truth[1:]);edge[:-1]|=diff;edge[1:]|=diff
        diff=valid[:,:-1]&valid[:,1:]&(truth[:,:-1]!=truth[:,1:]);edge[:,:-1]|=diff;edge[:,1:]|=diff
        dist=distance_transform_edt(~edge) if edge.any() else np.full(truth.shape,np.inf)
        masks['shadow_boundary_le3']=masks['shadow']&(dist<=3)
        masks['shadow_interior_gt3']=masks['shadow']&(dist>3)
        cloud_dist=distance_transform_edt(~masks['cloud']) if masks['cloud'].any() else np.full(truth.shape,np.inf)
        masks['shadow_near_cloud_le32']=masks['shadow']&(cloud_dist<=32)
        masks['shadow_far_cloud_gt32']=masks['shadow']&(cloud_dist>32)
        for c in ['surface','cloud']:
            if masks[c].any():
                q=float(np.quantile(bright[masks[c]],.1));dark=masks[c]&(bright<=q)
                masks[c+'_darkest_decile']=dark;masks[c+'_remaining']=masks[c]&~dark
        for model in reports:
            mg=groupmaps[model][product];p=roots[model]/mg['score_file'];assert sha(p)==mg['score_SHA'];INPUTS[str(p)]=mg['score_SHA']
            with np.load(p,allow_pickle=False) as z:
                assert set(z.files)=={'shadow','truth'} and np.array_equal(z['truth'],truth)
                score=z['shadow'].copy()
            assert score.shape==truth.shape and np.isfinite(score).all() and score.min()>=0 and score.max()<=1
            verified+=1
            for label,mask in masks.items():
                values=score[mask];n=values.size
                if not n:continue
                total=float(values.sum(dtype=np.float64));high=int(np.count_nonzero(values>=.5))
                row={'model':model,'product':product,'stratum':label,'pixels':int(n),'mean_shadow_score':total/n,'fraction_score_ge_0_5_pct':100*high/n}
                score_rows.append(row);v=acc.setdefault((model,label),[0,0.,0]);v[0]+=n;v[1]+=total;v[2]+=high
        save('STATUS.json',{'stage':'CPU_score_strata','groups_completed':index+1,'groups_total':32,'verified_score_files':verified,'seconds':time.monotonic()-START})
    score_summary=[{'model':m,'stratum':s,'pixels':v[0],'mean_shadow_score':v[1]/v[0],'fraction_score_ge_0_5_pct':100*v[2]/v[0],'note':'score>=0.5 is fixed diagnostic, NOT saved 3-class argmax prediction; masks overlap'} for (m,s),v in acc.items()]
    table('SCORE_STRATA.csv',score_rows);table('SCORE_SUMMARY.csv',score_summary)
    # Endpoint changes and loss are reconciled exactly; no resampling or fitting.
    for model in reports:
        assert abs(sum(r['contribution_to_delta_vs_Phase416_pp'] for r in classes if r['model']==model)-(miou(totals[model])-miou(basecm)))<1e-9
    save('INPUT_MANIFEST.json',INPUTS)
    save('ACCEPTANCE.json',{'passed':True,'models':len(reports),'groups':32,'score_files_verified':verified,'pooled_CM_and_mIoU_recomputed':True,'GT_and_group_support_paired':True,'class_delta_decomposition_exact':True,'new_training_updates':0,'new_model_forwards':0,'uses_torch':False,'limitations':['No full hard prediction maps saved: cannot count per-pixel fixes/regressions or locate all multiclass errors','Score threshold diagnostics are not model argmax metrics','GT-conditioned strata and perfect-repair probes are diagnostic oracles, not deployable rules','Single seed exposed development; descriptive, not confirmatory or causal','RGB darkness is relative within scene and GT class, not material identity']})
    save('COMPLETE.json',{'passed':True,'CPU_wall_seconds':time.monotonic()-START,'new_model_forwards':0,'new_training_updates':0})
    (R/'EXIT_CODE').write_text('0')

if __name__=='__main__':
    try:main()
    except BaseException:save('FAILURE.json',{'traceback':traceback.format_exc()});(R/'EXIT_CODE').write_text('1');raise

"""Paired descriptive error flow; development selection is not confirmation."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--adapted',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve summary')
    source=json.loads(a.source.read_text());adapted=json.loads(a.adapted.read_text())
    other={s['product']:s for s in adapted['scenes']};assert set(other)=={s['product'] for s in source['scenes']}
    rows=[]
    for s in source['scenes']:
        t=other[s['product']];sc=np.asarray(s['confusion']);tc=np.asarray(t['confusion'])
        assert s['group_id']==t['group_id'] and (sc.sum(1)==tc.sum(1)).all()
        rows.append(dict(product=s['product'],group_id=s['group_id'],cohort=s['cohort'],d=s['d'],
            h1_public_support=s['h1_public_support'],actual_shadow_pixels=s['actual_shadow_pixels'],
            delta_miou_pp=t['miou_percent']-s['miou_percent'],
            delta_shadow_iou_pp=None if s['iou_percent'][2] is None or t['iou_percent'][2] is None else t['iou_percent'][2]-s['iou_percent'][2],
            delta_shadow_to_surface_pixels=int(tc[2,0]-sc[2,0]),delta_shadow_to_cloud_pixels=int(tc[2,1]-sc[2,1]),
            delta_surface_to_shadow_pixels=int(tc[0,2]-sc[0,2]),source_features=s['source_observable_features']))
    out=dict(scope='exploratory_only_no_primary_test_no_65b_authorization',paired_scenes=rows,
             interpretation='Development checkpoint selection and label support limit inference; do not report preregistered confirmation passing.')
    fit=[r for r in rows if r['cohort']=='fit_representatives'];dev=[r for r in rows if r['cohort']=='development_val']
    for name,pool in [('fit',fit),('development',dev)]:
        h=[r for r in pool if r['h1_public_support'] and r['d'] is not None and r['delta_shadow_iou_pp'] is not None]
        d=np.array([r['d'] for r in h]);gain=np.array([r['delta_shadow_iou_pp'] for r in h])
        r=float(np.corrcoef(d,gain)[0,1]) if len(h)>2 and d.std()>0 and gain.std()>0 else None
        out[name]=dict(groups=len(pool),h1_supported_with_valid_shadow=len(h),descriptive_pearson_r=r,
            mean_delta_miou_pp=float(np.mean([r['delta_miou_pp'] for r in pool])))
    x=np.asarray([r['source_features'] for r in fit]);y=np.array([r['delta_shadow_to_surface_pixels']>0 for r in fit])
    yd=np.array([r['delta_shadow_to_surface_pixels']>0 for r in dev])
    out['h2']=dict(fit_positive=int(y.sum()),fit_negative=int((~y).sum()),development_positive=int(yd.sum()),development_negative=int((~yd).sum()))
    if min(y.sum(),(~y).sum())>=8 and np.isfinite(x).all():
        risk=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,solver='lbfgs',random_state=65))
        risk.fit(x,y);prob=risk.predict_proba([r['source_features'] for r in dev])[:,1]
        out['h2']['development_probabilities']=prob.tolist()
        if min(yd.sum(),(~yd).sum())>=8:out['h2']['descriptive_development_auroc']=float(roc_auc_score(yd,prob))
        else:out['h2']['interpretation']='Development class support insufficient; no AUROC claim'
    else:out['h2']['interpretation']='Fit support/input insufficient; risk model not fitted'
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False))


if __name__=='__main__':main()

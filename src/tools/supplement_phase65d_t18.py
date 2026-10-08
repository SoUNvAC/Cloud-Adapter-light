"""Post-hoc reference-Surface brightness strata, never correction selection."""
import argparse,json
import numpy as np
from analyze_phase65d import indices,load,T18
from phase65d_metrics import curve,score_summary
from phase65b_evidence_probe import authorized
from prepare_phase65a_explore import digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=authorized(a.root);out=root/'t18_surface_supplement.json'
    if out.exists():raise FileExistsError('Preserve supplementary diagnostic')
    report=root/'diagnostic_report.json';lock=root/'calibration_lock.json'
    before=digest(lock);rows,_=indices(root,'phase65b_evidence')
    r=rows[0][T18];truth=load(root,r,'truth');brightness=load(root,r,'brightness')
    surface=truth==0;shadow=truth==2
    boundary=float(np.quantile(brightness[surface],.1))
    strata={'surface_all':surface,'surface_lowest_brightness_decile':surface&(brightness<=boundary)}
    result=dict(product=T18,status='posthoc_descriptive_only',
        reason='Preregistered absolute brightness <0.08 reference-Surface stratum has only 3 pixels; insufficient for dark-Surface inference.',
        relative_stratum_definition='Lowest RGB-brightness decile of reference Surface, boundary ties included; image/reference only, no model scores used to choose stratum.',
        relative_brightness_boundary=boundary,brightness=score_summary(brightness[surface]),
        limitations='Relative low brightness is not independently verified dark land. Pixel counts are not independent samples. No threshold, model, or calibration refit.',
        original_report_sha256=digest(report),calibration_lock_sha256=before,models={})
    for model,rm in zip(['source','msre'],rows):
        r=rm[T18];model_result={}
        for score in ['score','native_score','margin']:
            values=load(root,r,score);entry={'distributions':{'shadow':score_summary(values[shadow])},'ranking':{}}
            for name,mask in strata.items():
                entry['distributions'][name]=score_summary(values[mask])
                c=curve(values[shadow|mask],truth[shadow|mask]==2)
                entry['ranking'][name]={k:v for k,v in c.items() if k not in ['precision','recall','fpr','thresholds']}
            model_result[score]=entry
        result['models'][model]=model_result
    if digest(lock)!=before:raise ValueError('Calibration lock changed')
    out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print('Supplement complete; calibration unchanged',before,flush=True)

if __name__=='__main__':main()

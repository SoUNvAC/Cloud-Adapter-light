"""Independent pooled CPU checks and small shared delivery sealing; no forwards."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
import run_phase89 as p
from summarize_phase89 import main as summarize

def main():
    root=p.ROOT;d=root/'delivery';r=p.read(d/'phase89_report.json');p.read(d/'CPU_VERIFIED.json')
    lock,b,models=p.locked();prep=p.read(root/'prepared/index.json')['rows'];checks=[]
    # Deliberately no area/calibrate/decisions/confusion/curve helpers here.
    for version,truthkey,maskkey in [('original_main','truth','valid'),('original_common','truth','common'),('revised_common','revised','common')]:
        ys=[];scores={n:[] for n in p.NAMES};matrices={n:np.zeros((3,3),np.int64) for n in p.NAMES}
        for row in prep:
            product=row['product'];truth=p.array(row,truthkey);mask=p.array(row,maskkey)
            if not mask.any():continue
            y=np.asarray(truth[mask]);ys.append(y);prob=p.probabilities(product)
            ms=prob['MsRE'][mask];nonshadow=np.argmax(ms[:,:2],axis=1)
            for name in p.NAMES:
                if name in ['Source','MsRE']:
                    values=prob[name][mask];score=values[:,2];pred=np.argmax(values,axis=1)
                else:
                    key='msre_source' if name=='L2' else 'msre_source_spectral'
                    score=prob[name][mask];threshold=float(np.exp(-np.logaddexp(0.,-models[key]['threshold'])))
                    pred=np.where(score>=threshold,2,nonshadow)
                scores[name].append(np.asarray(score,np.float32))
                for i in range(3):
                    for j in range(3):matrices[name][i,j]+=np.count_nonzero((y==i)&(pred==j))
        y=np.concatenate(ys)==2
        for name in p.NAMES:
            q=r['versions'][version]['policies'][name];score=np.concatenate(scores[name]);value=float(average_precision_score(y,score))
            assert abs(value-q['pooled']['ap'])<1e-12
            assert np.array_equal(matrices[name],q['pooled']['confusion'])
            for metric,key in [('fpr','macro_fpr'),('recall','macro_recall'),('miou_percent','macro_miou_percent')]:
                values=np.array([t[metric] for t in q['rows'] if t[metric] is not None],np.float64)
                assert abs(values.mean()-q[key]['mean'])<1e-12
                draws=np.random.default_rng(89011).integers(len(values),size=(10000,len(values)))
                boot=values[draws].mean(1)
                np.testing.assert_allclose(np.percentile(boot,[2.5,97.5]),q[key]['ci95'],rtol=0,atol=1e-12)
                assert abs(float(np.percentile(boot,95))-q[key]['upper95_one_sided'])<1e-12
            checks.append(dict(version=version,policy=name,pooled_sklearn_AP=value,pooled_confusion_exact=True,macro_mean_CI_verified=True))
            print('independent pooled CPU',version,name,flush=True)
        del scores,ys,score,y
    p.write(d/'POOLED_CPU_VERIFIED.json',dict(created_utc=p.now(),rows=checks,checks=12,no_forward_no_fit=True))
    p.write(d/'input_and_runtime_index.json',dict(prepared=p.read(root/'prepared/index.json'),
        inference={n:p.read(root/('scores_'+n)/'index.json') for n in ['Source','MsRE']},calibration=p.read(root/'calibrated/index.json'),
        preparation_seconds=sum(x['seconds'] for x in prep),source_attempt='run02; prior XML-selection preparation failure retained without forward',
        human_original_sha256=p.digest(p.HUMAN),human_B='not_executed'))
    summarize(d)
    p.write(d/'DELIVERY_SHA256.json',{str(x.relative_to(d)):p.digest(x) for x in sorted(d.iterdir()) if x.is_file()})
    p.write(root/'COMPLETED.json',dict(completed_utc=p.now(),delivery_manifest_sha256=p.digest(d/'DELIVERY_SHA256.json'),
        models=['Source','MsRE_2000','L2','L3'],groups=9,network_forwards=648,zero_training_fitting_tuning_downloads=True,
        external_reference_consistency_not_formal_confirmation=True,human_B='not_executed',no_work_dirs_transfer=True))

if __name__=='__main__':main()

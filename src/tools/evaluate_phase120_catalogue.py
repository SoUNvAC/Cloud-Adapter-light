"""Phase120 Catalogue support-first, five-policy CPU evaluation."""
import gc,sys
from pathlib import Path
import numpy as np
from phase120_common import *
from phase120_metrics import deliver

def main():
    setup();assert (ROOT/'train_C/EXIT_CODE').read_text().strip()=='0' and (ROOT/'train_D/EXIT_CODE').read_text().strip()=='0'
    for p in ('C','D'):
        assert sha(ROOT/(p+'_decoder_final.safetensors'))==load(ROOT/'TRAINING_SEAL.json')['models'][p]
        assert (ROOT/(p+'_Catalogue')/'EXIT_CODE').read_text().strip()=='0'
    oldroot=ROOT.parent/'phase119_20261015';plan=load(ROOT/'INPUT_PATCH_PLAN.json')['rows']
    rows=[r for r in plan if r['pool']=='Catalogue'];support=[];masks={}
    for row in rows:
        pid=row['product_id'];indices=None;common=None;bindings={}
        for p in ('R','N','C','D'):
            source=oldroot/p if p in ('R','N') else ROOT/(p+'_Catalogue')
            file=source/'scores/Catalogue'/(pid+'.npz')
            assert sha(file)==load(source/'SCORES_SEALED.json')['scores_sha256']['Catalogue/'+pid]
            bindings[p]=sha(file)
            with np.load(file,allow_pickle=False) as z:
                idx=z['flat_indices'];valid=z['valid']
            if indices is None:indices=idx;common=valid.copy()
            else:assert np.array_equal(indices,idx);common &= valid
        assert len(common)==row['group']['common_valid']
        e=OLD/'scores'/('O_'+pid+'.npz');assert sha(e)==row['old_score_sha256']
        with np.load(e,allow_pickle=False) as z:assert np.isfinite(z['scores']).all()
        path=ROOT/'common_support'/('Catalogue_'+pid+'.npy');path.parent.mkdir(exist_ok=True)
        with path.open('xb') as f:np.save(f,common,allow_pickle=False)
        support.append({'product':pid,'group_id':row['group']['group_id'],'original':len(common),'common':int(common.sum()),
            'loss':int((~common).sum()),'mask_sha256':sha(path),'score_bindings':bindings,'E_score_sha256':sha(e)})
        masks[pid]=common
    write(ROOT/'Catalogue_COMMON_SUPPORT_LOCK.json',{'before_reference_semantics_join':True,'settings':['E','R','N','C','D'],'rows':support})
    groups={}
    for row in rows:
        pid=row['product_id'];mask=masks[pid];record={'products':pid}
        with np.load(OLD/'scores'/('O_'+pid+'.npz'),allow_pickle=False) as z:
            truth=z['truth'];record['E']=z['scores'][mask]
        _,valid,original=catalog_image(row['group'],read_truth=True);assert np.array_equal(truth,original)
        record['truth']=truth[mask]==2
        for p in ('R','N','C','D'):
            source=oldroot/p if p in ('R','N') else ROOT/(p+'_Catalogue')
            with np.load(source/'scores/Catalogue'/(pid+'.npz'),allow_pickle=False) as z:record[p]=z['scores'][mask]
        groups[row['group']['group_id']]=record;gc.collect()
    per,summary,cmp=deliver(ROOT/'aggregate','Catalogue',groups)
    # All five are recomputed on this round's shared support. Reproduce old E/R/N only when unchanged.
    import csv
    expected=list(csv.DictReader((REPO/'src/research_results/phase119_20261015/Catalogue_summary.csv').open()))
    loss=sum(r['loss'] for r in support)
    if loss==0:
        for old in expected:
            new=next(v for v in summary if v['setting']==old['setting'])
            assert abs(new['macro_AP_pp']-float(old['macro_AP_pp']))<1e-8
            assert abs(new['pooled_AP_pp']-float(old['pooled_AP_pp']))<1e-8
    write(ROOT/'Catalogue_CPU_ACCEPTANCE.json',{'status':'passed','groups':32,'AP_groups':7,'support_loss':loss,
        'original_E_R_N_reproduced':loss==0,'independent_AP_tie_implementation_and_scalar_bootstrap':True,
        'seed':120015,'resamples':10000,'model_calls':0,'threshold':.5,'before_any_PixBox_new_inference':True})
    print(summary,cmp,flush=True)
if __name__=='__main__':main()

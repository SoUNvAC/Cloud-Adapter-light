"""Reuse frozen R scores and seal both evaluation supports without references."""
import numpy as np
from phase122_common import *
def main():
    setup();assert load(ROOT/'THRESHOLD_LOCK.json')['models_and_thresholds_before_evaluation']
    plan=load(ROOT/'INPUT_PATCH_PLAN.json')['rows'];old=ROOT.parent/'phase119_20261015/R';hashes={};support=[]
    for row in [r for r in plan if r['pool']=='Catalogue']:
        pid=row['product_id'];f=old/'scores/Catalogue'/(pid+'.npz')
        assert sha(f)==load(old/'SCORES_SEALED.json')['scores_sha256']['Catalogue/'+pid]
        with np.load(f,allow_pickle=False) as z:r={k:z[k].copy() for k in ('scores','valid','flat_indices')}
        dest=ROOT/'R_Catalogue_reuse/scores/Catalogue'/(pid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as file:np.savez_compressed(file,**r)
        hashes['Catalogue/'+pid]=sha(dest)
        cf=ROOT/'C_Catalogue/scores/Catalogue'/(pid+'.npz')
        assert sha(cf)==load(ROOT/'C_Catalogue/SCORES_SEALED.json')['scores_sha256']['Catalogue/'+pid]
        with np.load(cf,allow_pickle=False) as z:
            assert np.array_equal(r['flat_indices'],z['flat_indices']);common=r['valid']&z['valid']
        mask=ROOT/'common_support'/('Catalogue_'+pid+'.npy');mask.parent.mkdir(exist_ok=True)
        with mask.open('xb') as file:np.save(file,common,allow_pickle=False)
        support.append(dict(product=pid,group_id=row['group']['group_id'],original=len(common),common=int(common.sum()),loss=int((~common).sum()),
            mask_sha256=sha(mask),R_original_sha256=sha(f),R_repacked_sha256=sha(dest),C_sha256=sha(cf)))
    write(ROOT/'R_Catalogue_reuse/SCORES_SEALED.json',dict(scores_sha256=hashes,model_calls=0,source='frozen Phase119 R; unchanged float32 confidence'))
    write(ROOT/'Catalogue_COMMON_SUPPORT_LOCK.json',dict(before_reference_join=True,rows=support))
    write(ROOT/'ALL_SCORES_SEALED.json',dict(training=sha(ROOT/'TRAINING_SEAL.json'),thresholds=sha(ROOT/'THRESHOLD_LOCK.json'),
        settings={p:sha(ROOT/p/'SCORES_SEALED.json') for p in ('R_Calibration','C_Calibration','C_Catalogue','C_PixBox','R_Catalogue_reuse')},
        Catalogue_support=sha(ROOT/'Catalogue_COMMON_SUPPORT_LOCK.json'),evaluation_labels_joined=False))
if __name__=='__main__':main()

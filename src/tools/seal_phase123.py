"""Seal triple common support before any new evaluation-reference join."""
import numpy as np
from phase123_common import *
def main():
    setup();assert load(ROOT/'THRESHOLD_LOCK.json')['models_and_thresholds_before_evaluation']
    rows=load(ROOT/'INPUT_PATCH_PLAN.json')['rows'];supports=[]
    for row in [v for v in rows if v['pool']=='Catalogue']:
        pid=row['product_id'];arrays=[];sources=[]
        for base,stage in [(PREVIOUS,'R_Catalogue_reuse'),(PREVIOUS,'C_Catalogue'),(ROOT,'C_Catalogue')]:
            p=base/stage/'scores/Catalogue'/(pid+'.npz')
            assert sha(p)==load(base/stage/'SCORES_SEALED.json')['scores_sha256']['Catalogue/'+pid]
            with np.load(p,allow_pickle=False) as z:arrays.append((z['flat_indices'].copy(),z['valid'].copy()))
            sources.append(dict(path=str(p),sha256=sha(p)))
        assert all(np.array_equal(a[0],arrays[0][0]) for a in arrays)
        common=arrays[0][1]&arrays[1][1]&arrays[2][1]
        supports.append(dict(product=pid,group_id=row['group']['group_id'],original=len(common),common=int(common.sum()),loss=int((~common).sum()),sources=sources))
    write(ROOT/'Catalogue_COMMON_SUPPORT_LOCK.json',dict(before_reference_join=True,rows=supports))
    write(ROOT/'ALL_SCORES_SEALED.json',dict(training=sha(ROOT/'TRAINING_SEAL.json'),thresholds=sha(ROOT/'THRESHOLD_LOCK.json'),
        settings={p:sha(ROOT/p/'SCORES_SEALED.json') for p in ('C_Calibration','C_Catalogue','C_PixBox')},
        previous_seal_sha256=sha(PREVIOUS/'ALL_SCORES_SEALED.json'),Catalogue_support=sha(ROOT/'Catalogue_COMMON_SUPPORT_LOCK.json'),evaluation_labels_joined=False))
if __name__=='__main__':main()

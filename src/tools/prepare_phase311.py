"""Reproduce original image-only teacher caches and fixed cropped supports, zero forwards."""
import sys
import numpy as np
from phase311_common import *
from support_phase311 import partitions
from phase302_images import rgb,image_guard
def main():
    def deny(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            name=str(args[0]).lower();assert not any(k in name for k in ('masks.zip','alt_masks','classification_tags','phase306','phase307','pixbox'))
    sys.addaudithook(deny)
    rows=load(ROOT/'FIT_IMAGES.json')['rows'];steps=load(ROOT/'CROP_SCHEDULE.json')['rows']
    image_guard(rows,ROOT/'PREPARE_ACCESS.jsonl')
    oldlock=load(PL/'PSEUDO_LOCK.json');old={r['product']:r for r in oldlock['rows']};expected={int(r['attempt']):r for r in read(PL/'CROP_REPRODUCTION.csv')}
    prior=load(SP/'MASK_LOCK.json');priorproducts={r['product']:r for r in prior['rows']};priorcrops={int(r['attempt']):r for r in read(SP/'CROP_SUPPORT.csv')}
    products=[];crops=[]
    for i,row in enumerate(rows):
        pid=row['product'];entry=old[pid];sp=Path(entry['scores_path']);tp=PL/'teacher'/(pid+'.npz')
        assert sha(sp)==entry['scores_sha256'] and sha(tp)==entry['sha256']
        s=np.load(sp);assert array_sha(s)==entry['scores_array_sha256']
        image,valid=rgb(row,ROOT/'IMAGE_ACCESS.jsonl');A,P,pseudo=partitions(s,valid)
        with np.load(tp) as z:assert np.array_equal(z['valid'],valid) and np.array_equal(z['pseudo'],pseudo) and np.array_equal(z['pred'],s.argmax(0))
        assert int(valid.sum())==entry['valid'] and counts(pseudo)==entry['counts']
        oldentry=priorproducts[pid];dest=SP/oldentry['mask_file'];assert sha(dest)==oldentry['mask_sha256']
        assert array_sha(A)==oldentry['A_sha256'] and array_sha(P)==oldentry['P_sha256']
        with np.load(dest) as z:assert np.array_equal(A,z['A']) and np.array_equal(P,z['P']) and np.array_equal(valid,z['valid'])
        products.append(dict(product=pid,valid=int(valid.sum()),A=int(A.sum()),P=int(P.sum()),classes=counts(pseudo),A_sha256=array_sha(A),P_sha256=array_sha(P),mask_file=str(dest),mask_sha256=sha(dest),teacher_file=str(tp),teacher_sha256=sha(tp),scores_path=str(sp),scores_sha256=sha(sp),scores_array_sha256=array_sha(s),image_member_sha256=row['image_member_sha256']))
        for step in (x for x in steps if x['fit_index']==i):
            t,l=step['top'],step['left'];a=A[t:t+512,l:l+512].copy();p=P[t:t+512,l:l+512].copy();lab=pseudo[t:t+512,l:l+512].copy()
            if step['horizontal_flip']:a=a[:,::-1].copy();p=p[:,::-1].copy();lab=lab[:,::-1].copy()
            oldcrop=priorcrops[step['attempt']];assert array_sha(a)==oldcrop['A_sha256'] and array_sha(p)==oldcrop['P_sha256'] and array_sha(lab)==oldcrop['pseudo_sha256']
            cs=counts(lab);ex=expected[step['attempt']];assert cs==[int(ex[c]) for c in C]
            assert not (a&p).any() and sum(cs)==int(a.sum())
            crops.append(dict(**step,product=pid,A=int(a.sum()),P=int(p.sum()),Surface=cs[0],Cloud=cs[1],Shadow=cs[2],A_sha256=array_sha(a),P_sha256=array_sha(p),pseudo_sha256=array_sha(lab),update=bool(a.any())))
        print('cached prepare',i+1,flush=True)
    crops=sorted(crops,key=lambda x:x['attempt']);assert len(crops)==1000 and sum(x['update'] for x in crops)==955
    assert sum(x['valid'] for x in products)==104448397 and sum(x['A'] for x in products)==75469158
    assert np.sum([x['classes'] for x in products],axis=0).tolist()==[40622307,33554491,1292360]
    assert [sum(x[c]>0 for x in crops) for c in C]==[524,644,458]
    table(ROOT/'PRODUCT_SUPPORT.csv',products);table(ROOT/'CROP_SUPPORT.csv',crops)
    audits=[1,101,500,900]
    assert sum(x['P'] for x in products)==8201741 and sum(x['P'] for x in crops if x['update'])==19937970
    assert audits==prior['gradient_audit_attempts']
    write(ROOT/'PRIOR_MASK_REPRODUCTION.json',dict(status='passed',all_100_product_and_1000_crop_array_SHA_identical=True,original_mask_lock_SHA=sha(SP/'MASK_LOCK.json'),original_crop_support_SHA=sha(SP/'CROP_SUPPORT.csv'),original_product_support_SHA=sha(SP/'PRODUCT_SUPPORT.csv'),read_only_masks_reused=True))
    write(ROOT/'MASK_LOCK.json',dict(status='locked',rows=products,crops_sha256=sha(ROOT/'CROP_SUPPORT.csv'),products_sha256=sha(ROOT/'PRODUCT_SUPPORT.csv'),original_pseudo_lock_sha256=sha(PL/'PSEUDO_LOCK.json'),schedule_sha256=sha(ROOT/'CROP_SCHEDULE.json'),P_definition='image_valid & sum(s)>1e-12 & argmax(s)==0 & max(s/sum(s))<.90',A_definition='image_valid & sum(s)>1e-12 & max(s/sum(s))>=.90',disjoint=True,truth_reads=False,valid=104448397,A=75469158,classes=[40622307,33554491,1292360],P_unique_product_pixels=sum(x['P'] for x in products),P_cumulative_all_crops=sum(x['P'] for x in crops),P_cumulative_updated_crops=sum(x['P'] for x in crops if x['update']),P_nonempty_all_steps=sum(x['P']>0 for x in crops),P_nonempty_updated_steps=sum(x['P']>0 and x['update'] for x in crops),gradient_audit_attempts=audits,expected_updates=955,expected_skips=45,additional_teacher_reconstruction=0))
if __name__=='__main__':main()

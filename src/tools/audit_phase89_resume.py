"""Post disk-full integrity inventory; no new prediction, overwrites or cleanup."""
import hashlib,json
from pathlib import Path
import numpy as np
import run_phase89 as p

OUT=p.SHARED/'phase89_resume_audit_20261011.json'

def main():
    assert not OUT.exists(),'Preserve prior audit'
    result=dict(created_utc=p.now(),errors=[],objects=[],prepared=[],blocks=[],missing_blocks=[],receipts_without_output=[],
                no_forward_no_fit_no_deletion=True,prior_process_alive=False)
    err=result['errors']
    def attempt(role,fn):
        try:return fn()
        except Exception as e:err.append(dict(role=role,error=type(e).__name__+': '+str(e)));return None
    frozen=attempt('frozen_chain',p.locked)
    if frozen:
        lock,b,models=frozen
        for name,sha in [('Source',p.SOURCE_SHA),('MsRE',p.ADAPTED_SHA)]:
            attempt('checkpoint_'+name,lambda name=name,sha=sha:check_sha(p.CK[name],sha))
        for obj in lock['objects']:
            actual=attempt('object_'+obj['object'],lambda obj=obj:check_sha(Path(obj['path']),obj['sha256']))
            result['objects'].append(dict(object=obj['object'],matches=actual is not None))
        index=attempt('prepared_index',lambda:p.read(p.ROOT/'prepared/index.json'))
        if index:
            for r in index['rows']:
                for name,f in r['files'].items():
                    actual=attempt('prepared_'+r['product']+'_'+name,lambda f=f:check_sha(p.ROOT/f['path'],f['sha256']))
                    result['prepared'].append(dict(product=r['product'],field=name,matches=actual is not None))
                spec=attempt('prepared_spectra_load',lambda r=r:np.load(p.ROOT/r['files']['spectra20']['path'],mmap_mode='r'))
                for model in ['Source','MsRE']:
                    d=p.ROOT/('scores_'+model)/r['product'];h,w=r['shape60'];means=np.empty((h,w,3),np.float32);good=0
                    for y in range(0,h,340):
                        for x in range(0,w,340):
                            hh=min(340,h-y);ww=min(340,w-x);key=f'{y:04d}_{x:04d}'
                            receipt=d/(key+'_forward_receipt.json');output=d/(key+'.npy')
                            loc=dict(model=model,product=r['product'],key=key)
                            if not output.exists():
                                result['missing_blocks'].append(loc)
                                if receipt.exists():result['receipts_without_output'].append(loc)
                                continue
                            def block():
                                rec=p.read(receipt);a=np.load(output,allow_pickle=False)
                                assert a.shape==(hh*3,ww*3,3) and a.dtype==np.float32
                                assert np.isfinite(a).all() and (a>=0).all() and (a<=1).all()
                                assert np.allclose(a.sum(-1),1,rtol=0,atol=3e-7)
                                assert rec['y60']==y and rec['x60']==x and rec['true_shape20']==[hh*3,ww*3]
                                assert spec is not None
                                rgb=np.zeros((1020,1020,3),np.float32)
                                rgb[:hh*3,:ww*3]=spec[y*3:(y+hh)*3,x*3:(x+ww)*3,[2,1,0]]*np.float32(255.)
                                assert hashlib.sha256(rgb.tobytes()).hexdigest()==rec['input_rgb_sha256']
                                means[y:y+hh,x:x+ww]=p.area(a,3).astype(np.float32)
                                return dict(**loc,sha256=p.digest(output),bytes=output.stat().st_size,
                                    receipt_sha256=p.digest(receipt),structural_and_input_identity_checks_passed=True,
                                    pre_crash_output_sha_available=False)
                            got=attempt('block_'+model+'_'+r['product']+'_'+key,block)
                            if got:result['blocks'].append(got);good+=1
                    oldmean=d/'prob60.npy'
                    if oldmean.exists():
                        def aggregate():
                            assert good==36
                            assert np.array_equal(np.load(oldmean),means)
                            return p.digest(oldmean)
                        attempt('existing_60m_aggregate_'+model+'_'+r['product'],aggregate)
    result['status']='integrity_errors_stop' if err else 'structurally_readable_inventory_not_precrash_output_checksum_certification'
    result['summary']=dict(errors=len(err),objects=len(result['objects']),prepared_files=len(result['prepared']),
        existing_structurally_valid_blocks=len(result['blocks']),missing_blocks=len(result['missing_blocks']),
        receipt_without_output=len(result['receipts_without_output']))
    p.write(OUT,result);print(json.dumps(result['summary']),flush=True)

def check_sha(path,expected):
    actual=p.digest(path);assert actual==expected,(str(path),actual,expected);return actual

if __name__=='__main__':main()

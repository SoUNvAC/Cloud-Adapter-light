"""Separate payload check when crash damaged receipts; no inference or metrics."""
import json
import numpy as np
import run_phase89 as p

def main():
    out=p.SHARED/'phase89_probability_payload_audit_20261011.json';assert not out.exists()
    p.locked();rows=[];groups=[]
    for r in p.read(p.ROOT/'prepared/index.json')['rows']:
        h,w=r['shape60'];d=p.ROOT/'scores_Source'/r['product'];means=np.empty((h,w,3),np.float32);good=0
        for y in range(0,h,340):
            for x in range(0,w,340):
                key=f'{y:04d}_{x:04d}';path=d/(key+'.npy')
                if not path.exists():continue
                row=dict(product=r['product'],key=key,path=str(path),sha256=p.digest(path),bytes=path.stat().st_size,payload_valid=False)
                try:
                    a=np.load(path,allow_pickle=False);hh=min(340,h-y);ww=min(340,w-x)
                    assert a.shape==(hh*3,ww*3,3) and a.dtype==np.float32
                    assert np.isfinite(a).all() and (a>=0).all() and (a<=1).all() and np.allclose(a.sum(-1),1,rtol=0,atol=3e-7)
                    means[y:y+hh,x:x+ww]=p.area(a,3).astype(np.float32);row['payload_valid']=True;good+=1
                except Exception as e:row['error']=type(e).__name__+': '+str(e)
                rows.append(row)
        aggregate=dict(product=r['product'],valid_payloads=good,existing_aggregate=(d/'prob60.npy').exists(),aggregate_matches=None)
        if good==36 and aggregate['existing_aggregate']:
            try:aggregate['aggregate_matches']=bool(np.array_equal(np.load(d/'prob60.npy'),means))
            except Exception as e:aggregate['error']=type(e).__name__+': '+str(e)
        groups.append(aggregate)
    result=dict(created_utc=p.now(),rows=rows,groups=groups,payload_count=len(rows),valid_payloads=sum(r['payload_valid'] for r in rows),
                invalid_payloads=sum(not r['payload_valid'] for r in rows),not_a_precrash_SHA_certification=True,no_forward_no_fit_no_deletion=True)
    p.write(out,result);print(json.dumps({k:v for k,v in result.items() if k not in ['rows','groups']}))

if __name__=='__main__':main()

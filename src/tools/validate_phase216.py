"""Independent scalar CPU recomputation from original confusion and losses."""
from pathlib import Path
import csv,json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase216/phase216_20261023';RAW=OUT/'delivery'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def read(n):return list(csv.DictReader((OUT/n).open(encoding='utf-8-sig')))
def score(c):
    c=np.asarray(c);vals=[]
    for k in range(3):
        den=sum(c[k,:])+sum(c[:,k])-c[k,k]
        if den:vals.append(c[k,k]/den)
    return float(sum(vals)/len(vals)*100)
def main():
    manifest=js(RAW/'DELIVERY_MANIFEST.json')
    assert all((RAW/n).stat().st_size==r['bytes'] and hashlib.sha256((RAW/n).read_bytes()).hexdigest()==r['sha256'] for n,r in manifest.items())
    assert js(RAW/'REMOTE_CPU_ACCEPTANCE.json')['status']=='passed'
    # Independently bind every delivered source ordinal to the official ROI metadata.
    official=list(csv.DictReader((ROOT/'outputs/phase65/source_metadata/acquired/val_metadata.csv').open(encoding='utf-8-sig')))
    support=js(RAW/'EVAL_INPUT_LOCK.json')['source'];assert len(support)==len(official)==535
    assert all(r['scene']==official[int(Path(r['name']).stem)]['roi_id'] for r in support)
    cases={}
    for r in read('PER_GROUP.csv'):
        key=(r['arm'],float(r['alpha']),r['pool']);c=np.asarray(json.loads(r['confusion']),np.int64);cases.setdefault(key,{})[r['group']]=c
        assert abs(score(c)-float(r['mIoU']))<1e-10
        assert int(c.sum())==int(r['valid_pixels'])
    poolrows={(r['arm'],float(r['alpha']),r['pool']):r for r in read('ALL_POOL_METRICS.csv')}
    for key,gs in cases.items():
        assert abs(score(sum(gs.values()))-float(poolrows[key]['mIoU']))<1e-10
    sourcebase=cases['C',0,'source'];assert len(sourcebase)==107
    recomputed={};selected={}
    for arm in ['C','D']:
        for a in [0,.25,.5,.75,1]:
            gs=cases[arm,a,'source'];loss=sorted([score(sourcebase[g])-score(gs[g]) for g in gs],reverse=True)
            recomputed[arm,a]={'pool_loss_pp':score(sum(sourcebase.values()))-score(sum(gs.values())),'T11_pp':sum(loss[:11])/11,'worst_ROI_loss_pp':loss[0]}
        for b in [1,3,5]:selected[arm,b]=max(a for a in [0,.25,.5,.75,1] if recomputed[arm,a]['pool_loss_pp']<=b and recomputed[arm,a]['T11_pp']<=5)
    lock=js(RAW/'SOURCE_SELECTION_LOCK.json');assert all(r['alpha']==selected[r['arm'],r['budget_pp']] for r in lock['rows'])
    for r in read('ALL_CANDIDATES.csv'):
        for n,v in recomputed[r['arm'],float(r['alpha'])].items():assert abs(float(r[n])-v)<1e-10
    proof=js(OUT/'BOOTSTRAP_PROOF.json');checks=0;cirows={(int(r['budget_pp']),r['metric']):r for r in read('PAIRED_CI.csv')};full_ci_checks=0
    def batch_score(c):
        values=[]
        for k in range(3):
            denom=c[:,k,:].sum(1)+c[:,:,k].sum(1)-c[:,k,k]
            values.append(np.divide(c[:,k,k],denom,out=np.full(len(c),np.nan),where=denom>0))
        return np.nanmean(np.stack(values,axis=1),axis=1)*100
    # Independently regenerate all index draws, then scalar-check the first100 for each metric/budget.
    for b in [1,3,5]:
        ca,da=selected['C',b],selected['D',b]
        for pool,num in [('test',10),('source',107)]:
            ids=sorted(cases['C',ca,pool]);assert len(ids)==num
            ix=np.random.default_rng(216023).integers(0,num,size=(10000,num))
            c=cases['C',ca,pool];d=cases['D',da,pool];cstack=np.stack([c[g] for g in ids]);dstack=np.stack([d[g] for g in ids])
            if pool=='test':full={'test_mIoU_D_minus_C_pp':batch_score(dstack[ix].sum(1))-batch_score(cstack[ix].sum(1))}
            else:
                cl=np.array([score(sourcebase[g])-score(c[g]) for g in ids]);dl=np.array([score(sourcebase[g])-score(d[g]) for g in ids])
                full={'source_pool_loss_D_minus_C_pp':batch_score(cstack[ix].sum(1))-batch_score(dstack[ix].sum(1)),'source_T11_D_minus_C_pp':np.sort(dl[ix],axis=1)[:,-11:].mean(1)-np.sort(cl[ix],axis=1)[:,-11:].mean(1)}
            for n,v in full.items():
                lo,hi=np.quantile(v,[.025,.975]);r=cirows[b,n];assert abs(lo-float(r['CI95_low']))<1e-10 and abs(hi-float(r['CI95_high']))<1e-10;full_ci_checks+=1
            for j,indices in enumerate(ix[:100]):
                names=[ids[i] for i in indices];c=cases['C',ca,pool];d=cases['D',da,pool]
                if pool=='test':actual={'test_mIoU_D_minus_C_pp':score(sum(d[g] for g in names))-score(sum(c[g] for g in names))}
                else:
                    lc=sorted((score(sourcebase[g])-score(c[g]) for g in names),reverse=True);ld=sorted((score(sourcebase[g])-score(d[g]) for g in names),reverse=True)
                    actual={'source_pool_loss_D_minus_C_pp':score(sum(c[g] for g in names))-score(sum(d[g] for g in names)),'source_T11_D_minus_C_pp':sum(ld[:11])/11-sum(lc[:11])/11}
                for name,v in actual.items():assert abs(proof[str(b)+'/'+name]['first100'][j]-v)<1e-10;checks+=1
    result={'status':'passed','original_confusions_integer_support':True,'source535_ROI_bindings_match_official_metadata':True,'pooled_and_group_mIoU_independently_recomputed':True,'all_source_choices_independently_recomputed':True,'paired_scalar_bootstrap_checks':checks,'full10000_percentile_checks':full_ci_checks,'remote_heads_ledger_acceptance':True,'new_model_calls':0,'independent_code_SHA':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'INDEPENDENT_CPU_ACCEPTANCE.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()

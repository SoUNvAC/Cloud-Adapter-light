"""Four distances in exactly one frozen DINO layer and one global PCA."""
from pathlib import Path
import json,csv,time,hashlib
import numpy as np
from math_phase441 import scene_moments,combine,pca_from_moments,gaussian,sliced
R=Path('/home/scv/shared/phase443_20261103');OLD=Path('/home/scv/shared/phase441_20261103')
def save(n,x):(R/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    start=time.monotonic();rows=json.loads((R/'INPUTS.json').read_text());names=sorted({x['group'] for x in rows});groups=np.arange(len(names));mapping={s:i for i,s in enumerate(names)};domains=np.array([0 if s.startswith('A:') else 1 for s in names])
    arrays=[];gg=[];class_indices=[[],[],[]];offset=0;parity=[]
    for row in rows:
        assert sum(row['counts'])==np.prod(row['shape']),'Invalid pixels require explicit area weights'
        with np.load(R/'official/features'/f"{row['index']:03d}.npz") as z:a=z['final']
        assert a.shape==(1024,384);arrays.append(a);gg.append(np.full(1024,mapping[row['group']],int))
        for s in row['selected']:
            if s['kind']=='pure':class_indices[s['category']].append(offset+s['token'])
        offset+=1024
    x=np.concatenate(arrays).astype(np.float64);g=np.concatenate(gg);del arrays,gg
    BASE=Path('/home/scv/shared/phase442_20261103')
    with np.load(R/'COMMON_PCA.npz') as p:center=p['center'];basis=p['basis']
    assert sha(R/'COMMON_PCA.npz')==sha(BASE/'COMMON_PCA.npz')
    explained=json.loads((BASE/'LOCK.json').read_text())['explained_variance']
    z=(x-center)@basis;del x
    indices=[np.arange(len(z))]+[np.array(v) for v in class_indices]
    boots=np.load(R/'SCENE_BOOTSTRAP.npy');directions=np.load(R/'SW_DIRECTIONS.npy')
    result=[]
    baseline={v['scope']:v for v in json.loads((BASE/'SUMMARY.json').read_text())}
    for name,idx in zip(['global','surface','cloud','shadow'],indices):
        a=z[idx];ga=g[idx];m=scene_moments(a,ga,groups)
        mu,ca=combine(*m,weights=(domains==0));nu,cb=combine(*m,weights=(domains==1))
        bm,bc=combine(*m,weights=boots*(domains==0));bn,bd=combine(*m,weights=boots*(domains==1))
        f=float(gaussian(mu,ca,nu,cb));bs=gaussian(bm,bc,bn,bd)
        out=dict(scope=name,F=f,F_lo=float(np.quantile(bs,.025)),F_hi=float(np.quantile(bs,.975)),
            mean_distance=float(np.linalg.norm(mu-nu)),source_scenes=int(((m[2]>0)&(domains==0)).sum()),target_scenes=int(((m[2]>0)&(domains==1)).sum()),
            source_tokens=int((domains[ga]==0).sum()),target_tokens=int((domains[ga]==1).sum()))
        sw=sliced(a,ga,groups,domains,boots,directions);out.update(SW1_exact=sw['SW1_exact'],SW1_binned=sw['SW1'],SW1_lo=sw['SW1_CI'][0],SW1_hi=sw['SW1_CI'][1],SW1_bin_error_bound=sw['SW1_bin_error_bound'])
        original=baseline[name]
        with np.load(BASE/(name+'_MOMENTS.npz')) as old:
            assert np.array_equal(idx,old['selected_indices']) and np.array_equal(m[2],old['scene_counts'])
            delta=bs-old['F_bootstrap']
        out.update(F_before=original['F'],F_delta=f-original['F'],F_reduction_percent=100*(1-f/original['F']),F_delta_lo=float(np.quantile(delta,.025)),F_delta_hi=float(np.quantile(delta,.975)),SW1_before=original['SW1_exact'],SW1_delta=sw['SW1_exact']-original['SW1_exact'])
        result.append(out);np.savez_compressed(R/(name+'_MOMENTS.npz'),source_mean=mu,source_cov=ca,target_mean=nu,target_cov=cb,scene_means=m[0],scene_seconds=m[1],scene_counts=m[2],F_bootstrap=bs,selected_indices=idx)
        save('STATUS.json',dict(completed=name,seconds=time.monotonic()-start))
    with (R/'MAIN.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    save('SUMMARY.json',result)
    save('LOCK.json',dict(model='Official DINOv3-S+/16',layer='block12 final LayerNorm patch tokens',PCA_sha=sha(R/'COMMON_PCA.npz'),PCA_refit=False,explained_variance_original=explained,input_sha=sha(R/'INPUTS.json'),alignment_sha=sha(R/'ALIGNMENT.npz'),weight_lock=json.loads((R/'official/LOCK.json').read_text()),groups=names,bootstrap='Identical Phase442 scene resamples, paired F differences; alignment and PCA fixed',conditional='Identical Phase442 purity>=90% indices; target shadow5 scenes161 tokens',analysis_sha=sha(__file__)))
    save('COMPLETE.json',dict(passed=True,forwards=567,network_updates=0,PCA_refit=False,seconds=time.monotonic()-start))
if __name__=='__main__':main()

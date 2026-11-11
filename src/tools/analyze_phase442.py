"""Four distances in exactly one frozen DINO layer and one global PCA."""
from pathlib import Path
import json,csv,time,hashlib
import numpy as np
from math_phase441 import scene_moments,combine,pca_from_moments,gaussian,sliced
R=Path('/home/scv/shared/phase442_20261103');OLD=Path('/home/scv/shared/phase441_20261103')
def save(n,x):(R/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    start=time.monotonic();rows=json.loads((R/'INPUTS.json').read_text());names=sorted({x['group'] for x in rows});groups=np.arange(len(names));mapping={s:i for i,s in enumerate(names)};domains=np.array([0 if s.startswith('A:') else 1 for s in names])
    arrays=[];gg=[];class_indices=[[],[],[]];offset=0;parity=[]
    for row in rows:
        assert sum(row['counts'])==np.prod(row['shape']),'Invalid pixels require explicit area weights'
        with np.load(R/'official/features'/f"{row['index']:03d}.npz") as z:a=z['final']
        assert a.shape==(1024,384);arrays.append(a);gg.append(np.full(1024,mapping[row['group']],int))
        with np.load(OLD/'official/features'/f"{row['index']:03d}.npz") as z:
            old=z['final'];indices=np.array([s['token'] for s in row['selected']],dtype=np.int64)
            assert np.array_equal(a[indices],old);parity.append(True)
        for s in row['selected']:
            if s['kind']=='pure':class_indices[s['category']].append(offset+s['token'])
        offset+=1024
    x=np.concatenate(arrays).astype(np.float64);g=np.concatenate(gg);del arrays,gg
    moments=scene_moments(x,g,groups);center,basis,explained,eigen=pca_from_moments(*moments,domains)
    np.savez_compressed(R/'COMMON_PCA.npz',center=center,basis=basis,eigenvalues=eigen)
    z=(x-center)@basis;del x,moments
    indices=[np.arange(len(z))]+[np.array(v) for v in class_indices]
    represented=np.stack([np.bincount(g[i],minlength=len(groups))>0 for i in indices])
    rng=np.random.default_rng(44201);boots=[]
    while len(boots)<256:
        w=np.zeros(len(groups))
        for d in [0,1]:
            ids=groups[domains==d];w[ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))
        if all((w*represented[k]*(domains==d)).sum()>0 for k in range(4) for d in [0,1]):boots.append(w)
    boots=np.stack(boots);np.save(R/'SCENE_BOOTSTRAP.npy',boots)
    rng=np.random.default_rng(44130);directions=rng.normal(size=(64,64));directions/=np.linalg.norm(directions,axis=0);np.save(R/'SW_DIRECTIONS.npy',directions)
    result=[]
    for name,idx in zip(['global','surface','cloud','shadow'],indices):
        a=z[idx];ga=g[idx];m=scene_moments(a,ga,groups)
        mu,ca=combine(*m,weights=(domains==0));nu,cb=combine(*m,weights=(domains==1))
        bm,bc=combine(*m,weights=boots*(domains==0));bn,bd=combine(*m,weights=boots*(domains==1))
        f=float(gaussian(mu,ca,nu,cb));bs=gaussian(bm,bc,bn,bd)
        out=dict(scope=name,F=f,F_lo=float(np.quantile(bs,.025)),F_hi=float(np.quantile(bs,.975)),
            mean_distance=float(np.linalg.norm(mu-nu)),source_scenes=int(((m[2]>0)&(domains==0)).sum()),target_scenes=int(((m[2]>0)&(domains==1)).sum()),
            source_tokens=int((domains[ga]==0).sum()),target_tokens=int((domains[ga]==1).sum()))
        sw=sliced(a,ga,groups,domains,boots,directions);out.update(SW1_exact=sw['SW1_exact'],SW1_binned=sw['SW1'],SW1_lo=sw['SW1_CI'][0],SW1_hi=sw['SW1_CI'][1],SW1_bin_error_bound=sw['SW1_bin_error_bound'])
        result.append(out);np.savez_compressed(R/(name+'_MOMENTS.npz'),source_mean=mu,source_cov=ca,target_mean=nu,target_cov=cb,scene_means=m[0],scene_seconds=m[1],scene_counts=m[2],F_bootstrap=bs,selected_indices=idx)
        save('STATUS.json',dict(completed=name,seconds=time.monotonic()-start))
    with (R/'MAIN.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    save('SUMMARY.json',result);save('LOCK.json',dict(model='Official pretrained DINOv3-S+/16',layer='block12 followed by official final LayerNorm; no adapter/CNN/head',
        pca='Single global PCA fitted with equal domains and equal ROIs/scenes, all valid tokens; reused unchanged for all four rows',dimensions=64,whiten=False,explained_variance=explained,
        PCA_sha=sha(R/'COMMON_PCA.npz'),weight_lock=json.loads((R/'official/LOCK.json').read_text()),groups=names,
        global_population='All 1024 valid patch tokens per image, includes mixed patches, no class balancing',
        conditional='Reuse Phase441 purity>=90%, maximum128 per ROI/class; represented scenes equally weighted within each domain/class',
        existing_feature_parity_all567_exact=all(parity),bootstrap='256 ROI/scene resamples; PCA held fixed',
        raw_class_priors=json.loads((OLD/'PROTOCOL.json').read_text())['pi_raw'],
        limitations='Different existing RGB pipelines retained; target pure-shadow only5 groups161 tokens and24.42% shadow area',
        analysis_sha=sha(__file__)))
    lines=['# Phase442：固定DINOv3、固定层、固定PCA的四项距离','',f'官方DINOv3-S+/16，第12个Transformer块之后的最终LayerNorm特征。全局拟合一套64维PCA，两域/场景等权，不白化，解释方差{explained:.2%}。四项均复用此PCA与64个切片方向。','',
        '|范围|高斯距离F [95%场景区间]|切片W1（精确）|源/目标场景|源/目标token|','|---|---|---:|---|---|']
    cn=dict(global_='全局',surface='地表',cloud='云',shadow='云影')
    for v in result:
        label='全局' if v['scope']=='global' else cn[v['scope']]
        lines.append(f"|{label}|{v['F']:.4f} [{v['F_lo']:.4f}, {v['F_hi']:.4f}]|{v['SW1_exact']:.4f}|{v['source_scenes']}/{v['target_scenes']}|{v['source_tokens']}/{v['target_tokens']}|")
    lines+=['','全局使用全部有效token，保留真实类别构成及混合边界；三类条件统计沿用90%纯度与每场景每类128上限。全局距离不是三类距离的加权平均。',
        'CloudSEN535图按107个ROI处理，Catalogue32组。没有分类器、没有网络更新；重新提取全量token567次前向，旧抽样token逐项严格复现。',
        '云影条件结果只覆盖目标5组161token、约24.42%原始云影面积。两域既有RGB管线保持原样。区间按场景重抽样，固定PCA；切片区间采用512箱近似并提供精确点值/误差上界。',
        'COMMON_PCA.npz、各范围MOMENTS.npz、抽样索引、场景重抽样与方向文件可用于复算；完整token特征留远端D盘。']
    (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');save('COMPLETE.json',dict(passed=True,forwards=567,network_updates=0,same_PCA_all_four=True,prior_feature_parity=True,seconds=time.monotonic()-start))
if __name__=='__main__':main()

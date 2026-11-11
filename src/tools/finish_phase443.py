from pathlib import Path
import json,hashlib,csv
import numpy as np
from scipy.stats import wasserstein_distance
R=Path('/home/scv/shared/phase443_20261103');B=Path('/home/scv/shared/phase442_20261103')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    for n in ['COMMON_PCA.npz','INPUTS.json','SCENE_BOOTSTRAP.npy','SW_DIRECTIONS.npy']:assert sha(R/n)==sha(B/n),n
    old=json.loads((B/'official/LOCK.json').read_text());new=json.loads((R/'official/LOCK.json').read_text())
    assert old['state_sha']==new['state_sha'] and old['checkpoint_sha']==new['checkpoint_sha']
    complete=json.loads((R/'official/COMPLETE.json').read_text());assert complete['state_unchanged'] and complete['forwards']==567
    results=json.loads((R/'SUMMARY.json').read_text());checks=[]
    for v in results:
        with np.load(R/(v['scope']+'_MOMENTS.npz')) as p,np.load(B/(v['scope']+'_MOMENTS.npz')) as b:
            a=p['source_cov'];c=p['target_cov'];va,ua=np.linalg.eigh(a);vc,uc=np.linalg.eigh(c)
            ra=(ua*np.sqrt(np.maximum(va,0)))@ua.T;rc=(uc*np.sqrt(np.maximum(vc,0)))@uc.T
            f=np.square(p['source_mean']-p['target_mean']).sum()+np.trace(a)+np.trace(c)-2*np.linalg.svd(ra@rc,compute_uv=False).sum()
            assert abs(f-v['F'])<1e-7 and np.array_equal(p['selected_indices'],b['selected_indices'])
            assert np.allclose(np.quantile(p['F_bootstrap']-b['F_bootstrap'],[.025,.975]),[v['F_delta_lo'],v['F_delta_hi']])
            checks.append(dict(scope=v['scope'],SVD_error=float(abs(f-v['F']))))
    signatures=json.loads((R/'official/SIGNATURES.json').read_text());assert len(signatures)==567
    for s in signatures:assert sha(R/'official/features'/f"{s['index']:03d}.npz")==s['feature_sha']
    stats=json.loads((R/'APPEARANCE_STATS.json').read_text());appearance={}
    for setting in ['before','after']:
        agg={}
        for d in ['A','B']:
            values=[v[setting] for v in stats if v['domain']==d]
            mu=np.mean([v['mean'] for v in values],axis=0);second=np.mean([v['second'] for v in values],axis=0)
            agg[d]=dict(mean=mu.tolist(),std=np.sqrt(np.maximum(second-mu*mu,0)).tolist(),endpoint_fraction=float(np.mean([v['endpoint_fraction'] for v in values])))
        q={d:np.concatenate([v[setting]['quantiles'] for v in stats if v['domain']==d]) for d in ['A','B']}
        appearance[setting]=dict(domains=agg,mean_gap=(np.array(agg['B']['mean'])-agg['A']['mean']).tolist(),approximate_W1=[float(wasserstein_distance(q['A'][:,c],q['B'][:,c])) for c in range(4)])
    appearance['order']=['R','G','B','RGB weighted intensity'];appearance['W1_definition']='Approximate equal-image/scene mixture of257 per-image sample quantiles; means/std use all resized pixels'
    appearance['out_of_fit_range']={d:np.mean([v['outside_fit_range'] for v in stats if v['domain']==d],axis=0).tolist() for d in ['A','B']}
    (R/'APPEARANCE_SUMMARY.json').write_text(json.dumps(appearance,indent=2))
    (R/'INDEPENDENT_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,fixed_PCA_byte_identical=True,fixed_weights=True,fixed_indices_bootstraps_directions=True,all567_feature_SHA=True,checks=checks),indent=2))
    lines=['# Phase443：颜色与亮度对齐对照','', '官方DINOv3-S+/16、最后一层归一化patch特征；Phase442的PCA文件、中心、64维基、抽样索引与场景重抽样逐字节不变。567次前向、零网络更新，独立SVD与特征SHA验收通过。','', '|范围|F原始|F对齐后|变化率|切片W1原始|切片W1对齐后|配对F差95%区间|','|---|---:|---:|---:|---:|---:|---|']
    labels=['全局','地表','云','云影']
    for name,v in zip(labels,results):lines.append(f"|{name}|{v['F_before']:.4f}|{v['F']:.4f}|{-v['F_reduction_percent']:+.2f}%|{v['SW1_before']:.4f}|{v['SW1_exact']:.4f}|[{v['F_delta_lo']:.4f}, {v['F_delta_hi']:.4f}]|")
    lines+=['','两域RGB通道分别通过4097档分位数映射到共同中间分布。每域所有图共用映射，不按图、不按类别调整；空间内容、掩膜不变。重复分位数用对应目标均值确定性合并，无抖动；输入未额外clip，样本极值外映射到端点并单列比例。', '颜色对齐只控制RGB边缘分布，不保证联合颜色/空间纹理一致。整体颜色也受类别构成影响，因此距离变化不能直接等同纯成像因素的因果贡献，更不等同分割IoU提升。', 'PCA从未重新拟合；原88.34%解释方差对应原始输入，不宣称对新特征仍保持该比例。256次配对场景bootstrap固定映射和PCA，不包含颜色映射拟合的不确定性。', '目标纯云影仅5场景161token，覆盖24.42%影面积。全局包含混合patch，三类保持90%纯度筛选。', '', '## 输入对齐检查（R/G/B/加权RGB亮度，0–255输入单位）']
    for k in ['before','after']:lines.append(f"{k}：近似分布W1="+', '.join(f'{v:.3f}' for v in appearance[k]['approximate_W1'])+'；均值差B−A='+', '.join(f'{v:.3f}' for v in appearance[k]['mean_gap']))
    lines+=['','颜色映射/统计见ALIGNMENT.npz、ALIGNMENT_PROTOCOL.json、APPEARANCE_SUMMARY.json；四项原始矩与抽样见各MOMENTS.npz。完整特征留远端D盘。']
    (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(R/'READOUT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(results=results,appearance=appearance,accepted=True)))
if __name__=='__main__':main()

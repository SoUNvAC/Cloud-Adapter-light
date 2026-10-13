"""Fixed reverse-increment analysis. No fits, forwards, downloads or decision metrics."""
import argparse,collections,hashlib,json,math,shutil,time
import numpy as np
from phase104_common import REPO,read,write,sha,now,csvwrite
from phase65d_recovery_math import logit_score,predict_logit
from phase65d_metrics import curve
from phase89_cpu_ap import average_precision_score as independent_ap

RESULT=REPO/'src/research_results/phase106_20261013'
OUT=REPO/'outputs/phase106/phase106_20261013'
DATA=REPO/'data/pixbox_s2_cmix_5036991/phase106_20261013'
CONTROL_SHA='c2a83964073918015460e0ad89a9857b60b6a764e0f8562652472b224915f5ad'
from pathlib import Path
REMOTE=Path('/home/scv/shared/data/pixbox_s2_cmix_5036991')

def freeze():
    assert not RESULT.exists(),'Preserve any prior attempt'
    control=REPO/'outputs/phase67/phase67_20261010/delivery/control_lock.json'
    assert sha(control)==CONTROL_SHA
    prior=REPO/'src/research_results/phase105_20261013'
    seal=read(prior/'SCORES_SEALED.json');scores=REPO/'data/pixbox_s2_cmix_5036991/phase105_20261013/point_scores_no_labels.json'
    assert sha(scores)==seal['score_sha256']
    RESULT.mkdir(parents=True);OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    (RESULT/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n',encoding='ascii')
    write(RESULT/'model_lock.json',dict(control_sha256=CONTROL_SHA,model_name='Source_spectral',
        model=read(control)['models']['Source_spectral']),True)
    write(RESULT/'input_score_lock.json',dict(utc=now(),status='frozen_before_B_scoring',
        protocol_sha256=sha(REPO/'src/research_plans/PHASE106_PIXBOX_REVERSE_INCREMENT_20261013.md'),
        code_sha256=sha(Path(__file__)),model_lock_sha256=sha(RESULT/'model_lock.json'),
        Source_cache_sha256=sha(scores),point_inputs_sha256=read(prior/'CPU_tree_acceptance.json')['point_inputs_sha256'],
        reference_sha256=read(prior/'task_lock.json')['phase103_reference_sha256'],
        inputs='float32 [original Source logit,B08,B11,B12]; Fortran layout; original float32 logit output',
        A='original cached Source probability, not reconstructed',B='original153 Phase67 Source_spectral logit',
        bootstrap=dict(seed=106013,resamples=10000,unit='paired related groups'),
        post_hoc=True,forward_count=0,fit_count=0,download_bytes=0,threshold_scans=0),True)
    print('Phase106 frozen; no B scores or reference join yet')

def score():
    start=time.monotonic();root=REMOTE/'phase106_20261013';root.mkdir(exist_ok=False)
    lock=read(RESULT/'input_score_lock.json');assert sha(Path(__file__))==lock['code_sha256']
    assert sha(RESULT/'model_lock.json')==lock['model_lock_sha256']
    source=REMOTE/'phase105_20261013/point_scores_no_labels.json';inputs=REMOTE/'phase104_20261013/point_inputs_private.json'
    assert sha(source)==lock['Source_cache_sha256'] and sha(inputs)==lock['point_inputs_sha256']
    assert np.__version__=='1.26.4'
    rows=read(source)['rows'];meta={r['point_id']:r for r in read(inputs)['rows']}
    assert len(rows)==len(meta)==9500 and {r['point_id'] for r in rows}==set(meta)
    valid=[r for r in rows if r['input_valid']]
    x=np.asfortranarray(np.asarray([[logit_score(r['Source']),*[meta[r['point_id']]['bands'][b]['TOA'] for b in ('B08','B11','B12')]] for r in valid],np.float32))
    assert x.flags.f_contiguous and len(x)==9498
    model=read(RESULT/'model_lock.json')['model'];b=predict_logit(x,model)
    # Separate scalar input standardization and explicit linear sum, without predict_logit.
    cpu=[]
    for r in valid:
        probability=max(1e-6,min(1-1e-6,r['Source']))
        values=[np.float32(math.log(probability)-math.log1p(-probability))]
        for band in ('B08','B11','B12'):
            dn=[v for line in meta[r['point_id']]['bands'][band]['DN'] for v in line]
            values.append(np.float32(sum(float(np.float32(v)/np.float32(10000)) for v in dn)/len(dn)))
        standard=[(float(v)-m)/s for v,m,s in zip(values,model['mean'],model['scale'])]
        value=standard[0]*model['theta'][0]
        for i in range(1,4):value+=standard[i]*model['theta'][i]
        cpu.append(np.float32(value+model['theta'][-1]))
    cpu=np.asarray(cpu,np.float32);error=np.abs(b.astype(float)-cpu.astype(float))
    tolerance=np.abs(np.spacing(b)).astype(float)
    assert np.all(error<=tolerance),'Independent float32 logit replay exceeds one ULP'
    # Original vectorized Phase90 Fortran computation must be bitwise identical.
    z=(x-np.array(model['mean']))/np.array(model['scale'])
    assert np.array_equal(b,(z@np.array(model['theta'][:-1])+model['theta'][-1]).astype(np.float32))
    assert np.isfinite(b).all()
    bs=dict(zip((r['point_id'] for r in valid),map(float,b)));scores=[];shared={}
    for r in rows:
        m=meta[r['point_id']];assert r['input_valid']==m['input_valid']
        assert all(r[k]==m[k] for k in ('product_id','group_id','row20','col20'))
        q={k:r[k] for k in ('point_id','product_id','group_id','row20','col20','input_valid')}
        q.update(A=r['Source'],B=bs.get(r['point_id']))
        cell=(q['product_id'],q['row20'],q['col20']);value=(q['A'],q['B'])
        if cell in shared:assert shared[cell]==value
        shared[cell]=value;scores.append(q)
    write(root/'point_scores_no_labels.json',dict(rows=scores),True)
    write(root/'CPU_score_acceptance.json',dict(status='passed',points=9500,valid=9498,
        scalar_float32_max_abs_error=float(error.max()),scalar_bitwise_equal=int(np.count_nonzero(b==cpu)),
        scalar_tolerance='one float32 ULP; actual original vectorized Fortran output preserved',
        Phase90_Fortran_replay_bitwise_equal=True,shared20m_score_identity=True,seconds=time.monotonic()-start,
        numpy=np.__version__,new_forwards=0,new_fits=0,new_download_bytes=0),True)
    write(root/'SCORES_SEALED.json',dict(utc=now(),score_sha256=sha(root/'point_scores_no_labels.json'),
        CPU_acceptance_sha256=sha(root/'CPU_score_acceptance.json'),input_lock_sha256=sha(RESULT/'input_score_lock.json'),
        no_label_access=True),True)
    print('Phase106 B scores sealed; no labels accessed')

def evaluate():
    lock=read(RESULT/'input_score_lock.json');seal=read(OUT/'SCORES_SEALED.json')
    path=DATA/'point_scores_no_labels.json';assert sha(path)==seal['score_sha256']
    assert sha(OUT/'CPU_score_acceptance.json')==seal['CPU_acceptance_sha256']
    assert sha(RESULT/'input_score_lock.json')==seal['input_lock_sha256']
    write(RESULT/'LOCAL_SCORE_SEAL.json',dict(utc=now(),score_sha256=sha(path),before_reference_join=True),True)
    ref=REPO/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json';assert sha(ref)==lock['reference_sha256']
    refs={r['point_ids'][0]:r for r in read(ref)};scores=read(path)['rows'];assert len(scores)==len(refs)==9500
    groups=collections.defaultdict(list);joined=[]
    for r in scores:
        q=refs[r['point_id']];assert all(str(r[k])==str(q[k]) for k in ('product_id','group_id','row20','col20'))
        row=dict(**r,reference=q['reference']);joined.append(row)
        if r['input_valid'] and q['reference']!='ignore':groups[r['group_id']].append(row)
    allrows=[r for rr in groups.values() for r in rr]
    assert len(groups)==10 and len(allrows)==8567 and sum(r['reference']=='positive_ground_cloud_shadow' for r in allrows)==379
    assert sum(r['reference']=='ignore' for r in joined)==931 and sum(not r['input_valid'] for r in joined)==2
    csvwrite(DATA/'joined_reference_scores_private.csv',joined)
    table=[];independent_delta=[];max_error=0
    def ap(rr,name):
        nonlocal max_error
        y=np.array([r['reference']=='positive_ground_cloud_shadow' for r in rr]);s=np.array([r[name] for r in rr],np.float32)
        # AP only: original tied-score summation, no FPR or threshold scan.
        if y.any() and (~y).any():
            order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order]
            ends=np.flatnonzero(np.r_[ss[1:]!=ss[:-1],True]);tp=np.cumsum(yy,dtype=np.int64)[ends]
            precision=tp/(ends+1);v=float(np.sum(np.diff(np.r_[0,tp])*precision/y.sum()))
            ind=independent_ap(y,s)
        else:v=ind=None
        if v is not None:max_error=max(max_error,abs(v-ind));assert abs(v-ind)<1e-12
        return v,ind
    for gid,rr in sorted(groups.items()):
        a,ai=ap(rr,'A');b,bi=ap(rr,'B');delta=(b-a)*100 if a is not None else None
        if delta is not None:independent_delta.append((bi-ai)*100)
        table.append(dict(group_id=gid,tile=rr[0]['product_id'].split('_T')[1][:5],
            positive=sum(r['reference']=='positive_ground_cloud_shadow' for r in rr),negative=sum(r['reference']!='positive_ground_cloud_shadow' for r in rr),
            A_AP=a,B_AP=b,delta_AP_pp=delta,change='NA' if delta is None else 'improved' if delta>1 else 'declined' if delta< -1 else 'stable'))
    eligible=[r for r in table if r['delta_AP_pp'] is not None];assert len(eligible)==8 and all(r['positive']>=5 and r['negative']>=20 for r in eligible)
    delta=np.array([r['delta_AP_pp'] for r in eligible]);indices=np.random.default_rng(106013).integers(0,8,(10000,8))
    ci=np.quantile(delta[indices].mean(1),[.025,.975]);scalar=sorted(sum(independent_delta[i] for i in idx)/8 for idx in indices)
    def quantile(q):
        t=q*9999;i=int(t);return scalar[i]+(t-i)*(scalar[min(i+1,9999)]-scalar[i])
    assert np.max(np.abs(ci-[quantile(.025),quantile(.975)]))<1e-12
    result=dict(mean_delta_AP_pp=float(delta.mean()),CI95_pp=ci.tolist(),width_pp=float(ci[1]-ci[0]),seed=106013,resamples=10000,
        n_related_groups=8,post_hoc=True,improved=sum(r['change']=='improved' for r in eligible),stable=sum(r['change']=='stable' for r in eligible),
        declined=sum(r['change']=='declined' for r in eligible),worst_group=min(eligible,key=lambda r:r['delta_AP_pp'])['tile'],worst_delta_AP_pp=float(delta.min()),
        interpretation='增量不确定，不能解释为等效' if ci[0]<=0<=ci[1] else '本批采样点上观察到正增量' if ci[0]>0 else '本批采样点上观察到负增量')
    totals=[dict(score=n,macro_AP_percent=float(np.mean([r[n+'_AP'] for r in eligible]))*100,pooled_AP_percent=ap(allrows,n)[0]*100) for n in ('A','B')]
    csvwrite(RESULT/'two_score_summary.csv',totals);csvwrite(RESULT/'per_group_results.csv',table)
    write(RESULT/'delta_interval.json',result,True)
    write(RESULT/'CPU_acceptance.json',dict(status='passed',independent_AP_max_abs_error=max_error,independent_scalar_bootstrap=True,
        NA_groups=2,all_point_ids=9500,positive=379,negative=8188,reference_ignored=931,invalid_input=2,
        remote_score_acceptance=read(OUT/'CPU_score_acceptance.json'),new_network_forwards=0,new_fits=0,new_download_bytes=0),True)
    rows='\n'.join(f"|{r['score']}|{r['macro_AP_percent']:.3f}|{r['pooled_AP_percent']:.3f}|" for r in totals)
    text=f'''# Phase106 事后反向增量分析\n\n## 1 数据结果\n\n固定PixBox16产品/10组，379正类、8188负类，8组AP；931 ignore、2输入无效不变。A为Phase105原Source概率，B为原153组Phase67 Source_spectral的float32 logit；不以G7或反推概率替代A。\n\n|评分|逐组平均AP(%)|池化AP(%)|\n|---|---:|---:|\n{rows}\n\nB−A逐组平均AP差{result['mean_delta_AP_pp']:.3f}pp；配对相关组bootstrap95%描述区间[{ci[0]:.3f},{ci[1]:.3f}]pp，宽{result['width_pp']:.3f}pp；10000次、seed106013。改善/稳定/下降{result['improved']}/{result['stable']}/{result['declined']}；最差{result['worst_group']}：{result['worst_delta_AP_pp']:.3f}pp。19GBQ/48NUG的AP为NA；逐组完整数值见per_group_results.csv。\n\n零新增网络前向、拟合、下载、阈值扫描与决策指标；独立CPU评分、AP并列及区间验收通过。\n\n## 2 结论分析\n\n{result['interpretation']}。PixBox性能已暴露，这是事后补充分析，不称新的独立确认，也不与Phase105事后合并成联合确认。结论限于当前固定读出与选择性采样参考，不能证明光谱特异性、物理机制、适配必要性或全场景误报受控。未新增RGB对照或修改标签/阈值；未知基础预训练独立性、参考语义及抽样限制保留。\n'''
    (RESULT/'README.md').write_text(text,encoding='utf-8',newline='\n')
    for n in ('CPU_score_acceptance.json','SCORES_SEALED.json'):shutil.copyfile(OUT/n,RESULT/n)
    write(RESULT/'phase105_background_only.json',dict(not_reevaluated=True,prior_primary=read(REPO/'src/research_results/phase105_20261013/primary_endpoint.json')),True)
    for p in RESULT.iterdir():
        if p.is_file():shutil.copyfile(p,OUT/p.name)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','score','evaluate']);a=p.parse_args()
    {'freeze':freeze,'score':score,'evaluate':evaluate}[a.mode]()

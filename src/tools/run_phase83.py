"""Frozen Shadow gate with exactly two fit-only conditional nonshadow readouts."""
import hashlib,io,sys,time,zipfile,subprocess,csv
from pathlib import Path
import numpy as np
from phase80_common import REPO,DATA,ROOT as PREVIOUS,PREP,MANIFEST_SHA,read,write,digest,base_check,deadline
from run_phase65d_recovery import records,prepare_verify,load
from phase65d_recovery_math import objective,standardize,logit_score,predict_logit
from phase65d_metrics import measures
from run_phase81 import group_row,checks
ROOT=DATA/'phase83_20261011'
RUNCONTROL=Path('/home/scv/shared/phase83_control_20261011')
PHASE81=DATA/'phase81_20261011'
PROTOCOL='src/research_plans/PHASE83_CONDITIONAL_READOUT_20261011.md'

def interval(values):
    x=np.asarray(values,np.float64);assert len(x) and np.isfinite(x).all()
    bs=x[np.random.default_rng(83011).integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),upper95_one_sided=float(np.quantile(bs,.95)),n_groups=len(x),seed=83011,resamples=10000,descriptive_only=True)

def conditional_u(prob):
    assert prob.shape[0]==3 and np.isfinite(prob).all()
    return np.log(np.maximum(prob[1].astype(np.float64),1e-6))-np.log(np.maximum(prob[0].astype(np.float64),1e-6))

def fit_readout(x,y,w):
    from scipy.optimize import minimize
    assert np.isfinite(x).all() and np.isfinite(w).all() and np.isin(y,[0,1]).all()
    w=w/w.sum();mean,scale=standardize(x,w);z=(x-mean)/scale
    zero=np.flatnonzero(np.sum((x-mean)**2*w[:,None],axis=0)==0).tolist()
    result=minimize(objective,np.zeros(x.shape[1]+1),args=(z,y,w),jac=True,method='L-BFGS-B',
        bounds=[(0,None)]+[(None,None)]*x.shape[1],options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
    out=dict(mean=mean.tolist(),scale=scale.tolist(),theta=result.x.tolist(),zero_variance_columns=zero,
        success=bool(result.success),message=str(result.message),iterations=int(result.nit),objective=float(result.fun),
        optimizer='L-BFGS-B',maxiter=1000,ftol=1e-12,gtol=1e-8,penalty=.001,threshold=.5,operator='>=',score_dtype='float64')
    assert np.isfinite(result.x).all() and np.isfinite(result.fun)
    return out

def predict_cloud(x,m):
    logits=((x-np.array(m['mean']))/np.array(m['scale']))@np.array(m['theta'][:-1])+m['theta'][-1]
    prob=np.exp(-np.logaddexp(0,-logits));assert np.isfinite(prob).all()
    return (prob>=.5).astype(np.uint8)

def state_sha(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def csvwrite(name,rows):
    with (ROOT/name).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def extract():
    import torch
    from run_phase80_training import setup
    setup()
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmseg.registry import MODELS,DATASETS
    done=read(PREVIOUS/'training_completed.json');ck=Path(done['best_checkpoint'])
    assert done['best_iteration']==3500 and digest(ck)==done['best_checkpoint_sha256']=='e6106b765feee929f98e3c1daa026feb7061384f4abdb7a604bb3b52782fe5e1'
    cfg=Config.fromfile(str(REPO/'src/configs/protocol/phase80_lora_rgb.py'))
    model=MODELS.build(cfg.model);model.init_weights();model.load_state_dict(torch.load(ck,map_location='cpu')['state_dict'],strict=True)
    model.requires_grad_(False);model.eval().cuda();torch.manual_seed(65);np.random.seed(65)
    before=state_sha(model);timings={};rows=[]
    for cohort in ('fit_representatives','development_val'):
        dest=ROOT/cohort;dest.mkdir();selected=records(cohort)
        old={r['product']:r for r in read(PREVIOUS/cohort/'index.json')['rows']}
        assert set(selected)==set(old)
        ds=DATASETS.build(dict(cfg.val_dataloader.dataset,cohort=cohort));torch.cuda.synchronize();start=time.monotonic()
        for i in range(len(ds)):
            sample=ds[i];product=Path(sample['data_samples'].img_path).name.removesuffix('_rgb.npy');r=selected[product];prepare_verify(r)
            assert not any(m.training for m in model.modules())
            with torch.no_grad():p=model.test_step(pseudo_collate([sample]))[0]
            probs=p.seg_logits.data.softmax(0).cpu().numpy().astype(np.float32)
            assert np.array_equal(probs[2],load(PREVIOUS,old[product],'score')),('Shadow probability drift',product)
            assert np.array_equal(p.seg_logits.data[:2].argmax(0).cpu().numpy(),load(PREVIOUS,old[product],'nonshadow'))
            assert np.allclose(probs.sum(0),1,atol=2e-7)
            path=dest/(product+'_probabilities.npy');np.save(path,probs,allow_pickle=False)
            rows.append(dict(product=product,group_id=r['group_id'],cohort=cohort,path=str(path.relative_to(ROOT)),sha256=digest(path),old_cache=old[product],prepared_sha256=r['prepared_sha256']))
            print('Phase83 frozen LoRA probabilities',cohort,i+1,len(ds),product,flush=True)
        torch.cuda.synchronize();timings[cohort]=time.monotonic()-start
    after=state_sha(model);assert before==after and not any(p.grad is not None for p in model.parameters())
    write(ROOT/'probability_index.json',dict(rows=rows,checkpoint_sha256=digest(ck),checkpoint_bytes=ck.stat().st_size,
        state_before_sha256=before,state_after_sha256=after,all_parameters_frozen=True,eval=True,forwards=185,
        times_seconds=timings,time_definition='includes prepared SHA/read, forward/postprocessing and full probability cache write/SHA',
        probability_definition='Phase80 identical seg_logits.data.softmax(0); float32 three-class aggregated Mask2Former scores',
        parameters=sum(p.numel() for p in model.parameters()),confirmation_access=False))

def full_features(r,index,z):
    prepare_verify(r);truth=np.load(PREP/r['mask_path']);valid=truth!=255
    blob=z.read('subscenes/'+r['product']+'.npy');assert hashlib.sha256(blob).hexdigest()==r['member_sha256']['image']
    raw=np.load(io.BytesIO(blob),allow_pickle=False);spectral=raw[...,[7,11,12]][valid]
    assert np.isfinite(spectral).all() and np.all(spectral!=0),'Original L+S support drift'
    p=ROOT/index['path'];assert digest(p)==index['sha256'];prob=np.load(p,mmap_mode='r')
    assert prob.shape==(3,)+truth.shape
    x=np.column_stack([conditional_u(prob)[valid],spectral]).astype(np.float64)
    oldx=np.column_stack([logit_score(prob[2][valid]),spectral]).astype(np.float32)
    return truth,valid,x,oldx

def fit():
    idx={r['product']:r for r in read(ROOT/'probability_index.json')['rows'] if r['cohort']=='fit_representatives'}
    rng=np.random.default_rng(83011);xs=[];ys=[];sizes=[];audit=[];dest=ROOT/'sample_indices';dest.mkdir()
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for product,r in records('fit_representatives').items():
            truth,valid,x,_=full_features(r,idx[product],z);y=truth[valid];eligible=np.flatnonzero((y==0)|(y==1))
            if not len(eligible):audit.append(dict(product=product,excluded='no_valid_reference_nonshadow'));continue
            sample=rng.choice(eligible,size=min(4096,len(eligible)),replace=False);path=dest/(product+'.npy');np.save(path,sample,allow_pickle=False)
            xs.append(x[sample]);ys.append(y[sample]);sizes.append(len(sample))
            audit.append(dict(product=product,group_id=r['group_id'],eligible_count=len(eligible),samples=len(sample),cloud_samples=int((y[sample]==1).sum()),
                index_path=str(path.relative_to(ROOT)),index_sha256=digest(path),index_space='flattened original valid reference mask',
                probability_sha256=idx[product]['sha256'],image_sha256=r['member_sha256']['image'],prepared_sha256=r['prepared_sha256']))
    x=np.concatenate(xs);y=np.concatenate(ys).astype(np.float64);w=np.concatenate([np.full(n,1/(len(sizes)*n)) for n in sizes]);models={}
    write(ROOT/'fit_input_audit.json',dict(rows=audit,seed=83011,rng_order='manifest representative insertion order, single PCG64 generator',groups=len(sizes),sample_count=len(y),
        sample_x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),sample_y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),same_indices_both_models=True,selection_independent_of_predicted_Shadow=True))
    for n,cols in [('R1',[0]),('R4',[0,1,2,3])]:
        start=time.monotonic();m=fit_readout(x[:,cols],y,w);m.update(columns=cols,seconds=time.monotonic()-start);write(ROOT/(n+'_fit.json'),m)
        assert m['success'],('Fixed optimizer failed; no retry',n,m);models[n]=m
    write(ROOT/'readout_lock.json',dict(models=models,fit_input_sha256=digest(ROOT/'fit_input_audit.json'),
        old_shadow_lock_sha256=digest(PREVIOUS/'calibration_lock.json'),old_shadow_model=read(PREVIOUS/'calibration_lock.json')['models']['L+S'],
        frozen_before_development_evaluation=True,no_network_training=True,no_threshold_search=True,no_development_fit=True))

def evaluate():
    from run_phase82_analysis import aggregate
    import run_phase82_analysis as agg
    agg.interval=interval
    reference=read(PHASE81/'phase81_report.json');lock=read(ROOT/'readout_lock.json');m=lock['old_shadow_model'];rows={'R1':[],'R4':[]};invariants=[]
    idx={r['product']:r for r in read(ROOT/'probability_index.json')['rows'] if r['cohort']=='development_val'}
    oldscores={r['product']:r for r in read(PREVIOUS/'development_input_audit.json')['rows']}
    refs={r['product']:r for r in reference['policies']['LL']['rows']}
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for product,r in records('development_val').items():
            truth,valid,x,oldx=full_features(r,idx[product],z);t=truth[valid];score=predict_logit(oldx,m)
            oldprob=load(PREVIOUS,idx[product]['old_cache'],'score')[valid]
            priorx=oldx.copy();priorx[:,0]=logit_score(oldprob).astype(np.float32)
            assert np.array_equal(score,predict_logit(priorx,m)),('frozen L+S pixel score drift',product)
            shadow=score>=m['threshold'];old=refs[product]
            for n in rows:
                cloud=predict_cloud(x[:,lock['models'][n]['columns']],lock['models'][n]);pred=np.where(shadow,2,cloud).astype(np.uint8)
                row=group_row(n,r,t,pred,old['ap']);checks(row,old)
                assert np.array_equal(pred==2,shadow)
                assert row['shadow_to_surface']+row['shadow_to_cloud']==old['shadow_to_surface']+old['shadow_to_cloud']
                rows[n].append(row)
            invariants.append(dict(product=product,score_exact=True,Shadow_mask_exact=True,all_shadow_metrics_exact=True,leak_total_exact=True,valid_pixels=len(t)))
            print('Phase83 locked development readouts',len(invariants),32,product,flush=True)
    policies={n:aggregate(reference['policies'][n]['rows'],reference['policies'][n]['pooled_ap']) for n in ('LL','MM','LM')}
    for n in rows:policies[n]=aggregate(rows[n],reference['policies']['LL']['pooled_ap'])
    pairs=[];comparisons={}
    for a,b in [('R4','LL'),('R4','R1'),('R1','LL')]:
        by={r['product']:r for r in policies[b]['rows']};d=[]
        for r in policies[a]['rows']:
            v=r['miou_percent']-by[r['product']]['miou_percent'];d.append(v)
            pairs.append(dict(comparison=a+' minus '+b,product=r['product'],group_id=r['group_id'],delta_mIoU_pp=v))
        comparisons[a+' minus '+b]=dict(macro_mIoU_pp=interval(d),pooled_mIoU_pp=policies[a]['miou_percent']-policies[b]['miou_percent'],
            worst_group_delta_pp=min(d),worst_products=[r['product'] for r,v in zip(policies[a]['rows'],d) if v==min(d)],
            improving=sum(v>0 for v in d),stable=sum(v==0 for v in d),declining=sum(v<0 for v in d))
    c=comparisons['R4 minus LL'];gate=dict(pooled_gain_at_least_1pp=c['pooled_mIoU_pp']>=1,macro_loss_at_most_1pp=c['macro_mIoU_pp']['mean']>=-1,all_invariants_passed=True)
    report=dict(policies=policies,comparisons=comparisons,gate=gate,retain_R4=all(gate.values()),formal_confirmation=False,primary='R4 minus LL',fixed_increment='R4 minus R1')
    write(ROOT/'phase83_report.json',report);write(ROOT/'invariant_checks.json',dict(status='passed',rows=invariants))
    csvwrite('paired_differences.csv',pairs)
    flat=[];table=[]
    for n,p in policies.items():
        table.append(dict(policy=n,macro_mIoU_percent=p['macro_miou_percent']['mean'],pooled_mIoU_percent=p['miou_percent'],
            Surface_IoU_percent=p['iou_percent'][0],Cloud_IoU_percent=p['iou_percent'][1],Shadow_IoU_percent=p['iou_percent'][2],
            macro_AP_percent=p['macro_ap']['mean']*100,macro_recall_percent=p['macro_recall']['mean']*100,macro_FPR_percent=p['macro_fpr']['mean']*100,
            FPR_upper95_percent=p['macro_fpr']['upper95_one_sided']*100,Surface_to_Cloud=p['confusion'][0][1],Cloud_to_Surface=p['confusion'][1][0],
            Shadow_to_Surface=p['shadow_to_surface'],Shadow_to_Cloud=p['shadow_to_cloud']))
        for r in p['rows']:
            item={k:v for k,v in r.items() if k not in ('confusion','reference_support','iou_percent')}
            for i,cl in enumerate(('Surface','Cloud','Shadow')):
                item[cl+'_reference_pixels']=r['reference_support'][i];item[cl+'_IoU_percent']=r['iou_percent'][i]
                for j,out in enumerate(('Surface','Cloud','Shadow')):item[cl+'_to_'+out]=r['confusion'][i][j]
            flat.append(item)
    csvwrite('summary.csv',table);csvwrite('development_groups.csv',flat)
    write(ROOT/'confusion_matrices.json',{n:dict(pooled=p['confusion'],rows=[dict(product=r['product'],confusion=r['confusion']) for r in p['rows']]) for n,p in policies.items()})
    lines=['# Phase83：冻结云影决策的条件读出','','## 1 数据结果','','153 fit代表拟合R1／R4，32 development代表评价。无网络训练；新增185次原LoRA三类概率推理。云影评分、阈值、掩膜和指标与LL完全相同。','',
        '|策略|宏mIoU %|池化mIoU %|Surface IoU %|Cloud IoU %|Shadow IoU %|','|---|---:|---:|---:|---:|---:|']
    for t in table:lines.append('|'+t['policy']+'|'+'|'.join(f'{t[k]:.4f}' for k in ['macro_mIoU_percent','pooled_mIoU_percent','Surface_IoU_percent','Cloud_IoU_percent','Shadow_IoU_percent'])+'|')
    for n,c in comparisons.items():
        v=c['macro_mIoU_pp'];lines+=['',f"{n}：宏mIoU {v['mean']:+.4f}个百分点，配对组bootstrap95%区间 {v['ci95']}；池化mIoU {c['pooled_mIoU_pp']:+.4f}；最差逐组 {c['worst_group_delta_pp']:+.4f}，{c['worst_products']}。"]
    lines+=['','## 2 结论探讨','',f"固定R4晋级门：{gate}；保留R4：{report['retain_R4']}。未通过则关闭本轮固定R4读出，不换胜者、不调阈值。R1仅保留探索记录。",'',
        f"LL／R1／R4的云影指标相同，宏FPR {policies['LL']['macro_fpr']['mean']*100:.4f}%，单侧95%上界 {policies['LL']['macro_fpr']['upper95_one_sided']*100:.4f}%；另保留Phase81原上界 {reference['policies']['LL']['macro_fpr']['upper95_one_sided']*100:.4f}%。读出改善不使误报预算自动通过。",'',
        '区间仅描述已反复使用development、单一冻结模型的组间差异，不是独立确认。LM需要两网络且不是保证可达到的上界。缺席类别仍按union>0取平均；逐组表列有效类别数，避免把mIoU突变解释成同等比例像元出错。', '',
        '部署仅原LoRA网络＋原L+S云影浅层评分器＋R1或R4读出；权重留远端work_dirs。推理时间与checkpoint大小见probability_index.json，浅层参数见readout_lock.json；checkpoint大小包含原训练保存格式。']
    (ROOT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    write(ROOT/'COMPLETED.json',dict(status='completed',report_sha256=digest(ROOT/'phase83_report.json'),shallow_fits=2,network_training=0,new_forwards=185,confirmations=False,work_dirs_sync=False))

def main():
    deadline();base_check();assert not ROOT.exists();ROOT.mkdir()
    bindings=read(REPO/'src/research_plans/PHASE83_INPUT_BINDINGS_20261011.json')
    for name,parent in [('phase80',PREVIOUS),('phase81',PHASE81)]:
        for rel,h in bindings[name].items():assert digest(parent/rel)==h,rel
    import run_phase82_analysis as baseline
    baseline.RUNCONTROL=RUNCONTROL;baseline.baseline()
    write(ROOT/'input_configuration_lock.json',dict(protocol_sha256=digest(REPO/PROTOCOL),code_sha256=digest(Path(__file__)),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        phase81_report_sha256=digest(PHASE81/'phase81_report.json'),phase80_lock_sha256=digest(PREVIOUS/'calibration_lock.json'),manifest_sha256=MANIFEST_SHA,
        code_files={p:digest(REPO/p) for p in [PROTOCOL,'src/tools/run_phase83.py','src/tools/verify_phase83.py','src/tools/test_phase83.py','src/tools/run_phase83.sh','src/research_plans/PHASE83_INPUT_BINDINGS_20261011.json',
            'src/tools/phase65d_recovery_math.py','src/tools/phase65d_metrics.py','src/tools/run_phase65d_recovery.py','src/tools/run_phase80_training.py','src/tools/run_phase82_analysis.py','src/tools/run_phase81.py']},
        previous=bindings,seed=83011,fit_representatives=153,development=32,no_confirmation=True,network_training=False))
    extract();fit();evaluate()
    print('Phase83 complete',flush=True)

if __name__=='__main__':main()

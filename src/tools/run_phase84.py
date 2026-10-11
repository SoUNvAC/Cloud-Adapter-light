"""One frozen mask-feature extraction, exactly two fixed linear readouts."""
import copy,csv,hashlib,inspect,io,json,subprocess,time,zipfile
from pathlib import Path
import numpy as np
import run_phase83 as p83
from phase80_common import REPO,DATA,ROOT as P80,PREP,MANIFEST_SHA,read,write,digest,base_check,deadline
from run_phase65d_recovery import records,load
from phase65d_recovery_math import objective,predict_logit
from run_phase81 import group_row,checks
from phase65d_metrics import measures
ROOT=DATA/'phase84_20261011'
CONTROL=Path('/home/scv/shared/phase84_control_20261011')
PROTOCOL='src/research_plans/PHASE84_FROZEN_SPATIAL_READOUT_20261011.md'
NAMES=('F_real','F_shuffle')

def interval(values):
    x=np.asarray(values,float);assert len(x) and np.isfinite(x).all()
    bs=x[np.random.default_rng(84011).integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),upper95_one_sided=float(np.quantile(bs,.95)),n_groups=len(x),resamples=10000,seed=84011,descriptive_only=True)

def gate(pool_ll,pool_r4,pool_shuffle,macro_ll,invariants):
    c=dict(pooled_gain_over_LL_at_least_1pp=pool_ll>=1,pooled_gain_over_R4_at_least_half_pp=pool_r4>=.5,
           pooled_gain_over_shuffle_at_least_half_pp=pool_shuffle>=.5,macro_loss_vs_LL_at_most_1pp=macro_ll>=-1,all_shadow_invariants=invariants)
    return dict(checks=c,retain_F_real=all(c.values()),formal_confirmation=False)

def csvwrite(name,rows):
    with (ROOT/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def baseline():
    import run_phase82_analysis as old
    old.RUNCONTROL=CONTROL;old.baseline()
    ref=read(p83.ROOT/'phase83_report.json');lock=read(p83.ROOT/'readout_lock.json');idx={r['product']:r for r in read(p83.ROOT/'probability_index.json')['rows']}
    refs={r['product']:r for r in ref['policies']['R4']['rows']};verified=[]
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for product,r in records('development_val').items():
            truth,valid,x,oldx=p83.full_features(r,idx[product],z);score=predict_logit(oldx,lock['old_shadow_model'])
            cloud=p83.predict_cloud(x,lock['models']['R4']);pred=np.where(score>=lock['old_shadow_model']['threshold'],2,cloud).astype(np.uint8)
            row=group_row('R4',r,truth[valid],pred,refs[product]['ap']);assert row==refs[product],('R4 baseline drift',product)
            verified.append(product)
    write(CONTROL/'R4_reproduction.json',dict(status='all_32_R4_rows_exact',products=verified,reference_report_sha256=digest(p83.ROOT/'phase83_report.json')))

class Extractor:
    """Capture actual mask resize arguments, then use original postprocess_result.

    Both interpolation stages are retained. No low-resolution linear projection.
    """
    def __init__(self):
        import torch
        from run_phase80_training import setup
        setup()
        from mmengine.config import Config
        from mmseg.registry import MODELS
        done=read(P80/'training_completed.json');ck=Path(done['best_checkpoint'])
        assert done['best_iteration']==3500 and digest(ck)==done['best_checkpoint_sha256']=='e6106b765feee929f98e3c1daa026feb7061384f4abdb7a604bb3b52782fe5e1'
        self.cfg=Config.fromfile(str(REPO/'src/configs/protocol/phase80_lora_rgb.py'));assert self.cfg.model.test_cfg.mode=='whole'
        self.model=MODELS.build(self.cfg.model);self.model.init_weights();self.model.load_state_dict(torch.load(ck,map_location='cpu')['state_dict'],strict=True)
        self.model.requires_grad_(False);self.model.eval().cuda();torch.manual_seed(65);np.random.seed(65)
        self.before=p83.state_sha(self.model);self.rows=[];self.expected=None;self.times={}
        self.indices={r['product']:r for r in read(p83.ROOT/'probability_index.json')['rows']}
        torch.cuda.reset_peak_memory_stats();self.ck=ck

    def dataset(self,cohort):
        from mmseg.registry import DATASETS
        return DATASETS.build(dict(self.cfg.val_dataloader.dataset,cohort=cohort))

    def forward(self,sample,product):
        import torch
        from mmengine.dataset import pseudo_collate
        from torch.nn import functional as fn
        captures={};calls=[];head=self.model.decode_head;original=head.predict_by_feat;interpolate=fn.interpolate
        def pixel_hook(module,args,out):
            assert 'mask_features' not in captures,'More than one pixel decoder forward'
            assert isinstance(out,(tuple,list));captures['mask_features']=out[0]
        def predict_trace(cls,masks,metas):
            captures['last_mask']=masks[-1];captures['metas']=copy.deepcopy(metas)
            result=original(cls,masks,metas);captures['head_output']=result
            return result
        signature=inspect.signature(interpolate)
        def resize_trace(tensor,*args,**kwargs):
            result=interpolate(tensor,*args,**kwargs)
            target=captures.get('last_mask')
            if target is not None and tensor.data_ptr()==target.data_ptr() and tuple(tensor.shape)==tuple(target.shape):
                bound=signature.bind(tensor,*args,**kwargs);bound.apply_defaults();params=dict(bound.arguments);params.pop('input')
                calls.append(dict(input_shape=list(tensor.shape),output_shape=list(result.shape),parameters=params))
            return result
        hook=head.pixel_decoder.register_forward_hook(pixel_hook);head.predict_by_feat=predict_trace;fn.interpolate=resize_trace
        assert not any(m.training for m in self.model.modules())
        try:
            with torch.no_grad():out=self.model.test_step(pseudo_collate([sample]))[0]
        finally:
            fn.interpolate=interpolate;head.predict_by_feat=original;hook.remove()
        assert len(calls)==1,('Unexpected mask-to-head interpolation chain',calls)
        assert calls[0]['parameters']['mode']=='bilinear' and calls[0]['parameters']['align_corners'] is False
        mask=captures['mask_features'];assert mask.ndim==4 and mask.shape[0]==1
        spec=dict(layer='decode_head.pixel_decoder output[0] (mask_features)',shape=list(mask.shape),channels=int(mask.shape[1]),
            head_mask_resize=calls[0],postprocess='original model.postprocess_result with identical sample metainfo (padding/crop, flip restoration, resize)',
            output_shape=list(out.seg_logits.data.shape[1:]),model_align_corners=self.model.align_corners)
        if self.expected is None:self.expected=spec
        else:assert spec==self.expected,('Feature geometry drift',product,spec,self.expected)
        prior=self.indices[product];p=p83.ROOT/prior['path'];assert digest(p)==prior['sha256']
        probabilities=out.seg_logits.data.softmax(0).cpu().numpy().astype(np.float32)
        assert np.array_equal(probabilities,np.load(p,mmap_mode='r')),('Three-class probability drift',product)
        with torch.no_grad():
            # This recovers the exact semantic head output through the same chain,
            # independently checking metadata before applying it to feature channels.
            checked=self.model.postprocess_result(captures['head_output'],[copy.deepcopy(out)])[0].seg_logits.data
            assert torch.equal(checked,out.seg_logits.data),'Postprocessing chain does not reproduce semantic output'
            first=interpolate(mask,**calls[0]['parameters'])
            aligned=self.model.postprocess_result(first,[copy.deepcopy(out)])[0].seg_logits.data
        assert aligned.shape[0]==spec['channels'] and aligned.shape[1:]==out.seg_logits.data.shape[1:] and torch.isfinite(aligned).all()
        self.rows.append(dict(product=product,three_class_probability_sha256=prior['sha256'],probabilities_exact=True,semantic_postprocess_exact=True))
        return aligned

    def finish(self):
        import torch
        import resource
        after=p83.state_sha(self.model);assert after==self.before and len(self.rows)==185
        assert all(p.grad is None for p in self.model.parameters())
        write(ROOT/'extraction_audit.json',dict(feature=self.expected,rows=self.rows,forwards=len(self.rows),times_seconds=self.times,
            state_before_sha256=self.before,state_after_sha256=after,checkpoint_sha256=digest(self.ck),checkpoint_bytes=self.ck.stat().st_size,
            parameters=sum(p.numel() for p in self.model.parameters()),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            memory_definition='CUDA whole run peak since reset after model load; Linux process peak RSS includes CPU fitting',
            same_forward_feature_and_probability=True,no_shortcut_interpolation=True,eval=True,all_parameters_frozen=True))

def extract_fit(engine):
    import torch
    previous=read(p83.ROOT/'fit_input_audit.json');audit={r['product']:r for r in previous['rows']};selected=records('fit_representatives');dest=ROOT/'fit_features';dest.mkdir()
    rng=np.random.default_rng(84011);xs=[];ys=[];fs=[];perms=[];rows=[];sizes=[];start=time.monotonic();forward_seconds=0
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        ds=engine.dataset('fit_representatives')
        for i in range(len(ds)):
            sample=ds[i];product=Path(sample['data_samples'].img_path).name.removesuffix('_rgb.npy');r=selected[product];a=audit[product]
            truth,valid,x,_=p83.full_features(r,engine.indices[product],z)
            torch.cuda.synchronize();t0=time.monotonic();f=engine.forward(sample,product);torch.cuda.synchronize();forward_seconds+=time.monotonic()-t0
            if 'excluded' in a:rows.append(a);del f;continue
            indices=p83.ROOT/a['index_path'];assert digest(indices)==a['index_sha256'];pos=np.load(indices);assert np.isin(truth[valid][pos],[0,1]).all()
            actual=np.flatnonzero(valid)[pos];rr,cc=np.unravel_index(actual,truth.shape)
            feature=f[:,torch.as_tensor(rr,device='cuda'),torch.as_tensor(cc,device='cuda')].T.contiguous().cpu().numpy().astype(np.float32)
            assert np.isfinite(feature).all();perm=rng.permutation(len(pos));path=dest/(product+'.npy');np.save(path,feature,allow_pickle=False)
            pp=dest/(product+'_permutation.npy');np.save(pp,perm,allow_pickle=False)
            xs.append(x[pos]);ys.append(truth[valid][pos]);fs.append(feature);perms.append(perm);sizes.append(len(pos))
            rows.append(dict(product=product,group_id=r['group_id'],feature_path=str(path.relative_to(ROOT)),feature_sha256=digest(path),shape=list(feature.shape),
                permutation_path=str(pp.relative_to(ROOT)),permutation_sha256=digest(pp),sample_index_path=str(indices),sample_index_sha256=a['index_sha256']))
            del f;print('Phase84 fit sampled features',i+1,len(ds),product,flush=True)
    x=np.concatenate(xs);y=np.concatenate(ys).astype(float);f=np.concatenate(fs);del xs,ys,fs
    assert hashlib.sha256(x.tobytes()).hexdigest()==previous['sample_x_sha256'] and hashlib.sha256(y.tobytes()).hexdigest()==previous['sample_y_sha256']
    engine.times['fit_total_with_base_features_seconds']=time.monotonic()-start;engine.times['fit_forward_and_alignment_seconds']=forward_seconds
    write(ROOT/'feature_lock.json',dict(feature=engine.expected,rows=rows,original_fit_audit_sha256=digest(p83.ROOT/'fit_input_audit.json'),
        seed=84011,permutation_rule='joint F vectors within each group; fresh PCG64(84011) per cohort; manifest representative order; no labels or base columns permuted',
        samples=len(y),groups=len(sizes),base_sample_x_sha256=previous['sample_x_sha256'],sample_y_sha256=previous['sample_y_sha256'],fit_saved_only_sampled_features=True))
    return x,y,f,perms,sizes

def fit_models(x,y,f,perms,sizes):
    from scipy.optimize import minimize
    old=read(p83.ROOT/'readout_lock.json');base=old['models']['R4'];w=np.concatenate([np.full(n,1/(len(sizes)*n)) for n in sizes]);w/=w.sum()
    mu=np.sum(f.astype(float)*w[:,None],0);var=np.sum((f.astype(float)-mu)**2*w[:,None],0);scale=np.sqrt(var);zero=np.flatnonzero(scale==0).tolist();scale[scale==0]=1
    mean=np.r_[base['mean'],mu];scales=np.r_[base['scale'],scale];models={}
    for name in NAMES:
        start=time.monotonic()
        ff=f if name=='F_real' else np.concatenate([part[p] for part,p in zip(np.split(f,np.cumsum(sizes)[:-1]),perms)])
        z=(np.column_stack([x,ff])-mean)/scales
        result=minimize(objective,np.zeros(z.shape[1]+1),args=(z,y,w),jac=True,method='L-BFGS-B',
            bounds=[(0,None)]+[(None,None)]*z.shape[1],options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
        m=dict(mean=mean.tolist(),scale=scales.tolist(),theta=result.x.tolist(),zero_variance_F_columns=zero,success=bool(result.success),
            message=str(result.message),iterations=int(result.nit),objective=float(result.fun),seconds=time.monotonic()-start,
            penalty=.001,optimizer='L-BFGS-B',maxiter=1000,ftol=1e-12,gtol=1e-8,threshold=.5,operator='>=',score_dtype='float64')
        write(ROOT/(name+'_fit.json'),m);assert m['success'] and np.isfinite(result.x).all() and np.isfinite(result.fun),('Fixed fit failed; no retry',name,m)
        models[name]=m;del z,ff;print('Phase84 fixed optimizer',name,m['iterations'],m['seconds'],flush=True)
    write(ROOT/'readout_lock.json',dict(models=models,feature_lock_sha256=digest(ROOT/'feature_lock.json'),original_phase83_lock_sha256=digest(p83.ROOT/'readout_lock.json'),
        original_shadow_model=old['old_shadow_model'],base_standardization_exact=True,shared_F_standardization='permutations preserve each equally weighted group distribution',
        frozen_before_development=True,development_shuffle_seed=84011,development_permutation_uses_labels=False,threshold_search=False,network_training=False))

def aligned_logits(base,f,valid_positions,permutation,m):
    """Feature interpolation is already complete; dense readout in bounded chunks."""
    count=len(base);out=np.empty(count,np.float64);mu=np.asarray(m['mean']);scale=np.asarray(m['scale']);theta=np.asarray(m['theta'])
    for start in range(0,count,32768):
        end=min(start+32768,count);where=valid_positions[start:end] if permutation is None else valid_positions[permutation[start:end]]
        # Joint channels are fetched at the same source pixel for every F vector.
        ff=f.reshape(f.shape[0],-1)[:,where].T
        x=np.column_stack([base[start:end],ff]);out[start:end]=((x-mu)/scale)@theta[:-1]+theta[-1]
    assert np.isfinite(out).all();return out

def evaluate(engine):
    import torch
    old=read(p83.ROOT/'phase83_report.json');lock=read(ROOT/'readout_lock.json');models=lock['models'];gate_model=lock['original_shadow_model']
    refs={r['product']:r for r in old['policies']['LL']['rows']};rows={n:[] for n in NAMES};rng=np.random.default_rng(84011);dest=ROOT/'development';dest.mkdir();audits=[]
    selected=records('development_val');start=time.monotonic();forward_seconds=0
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        ds=engine.dataset('development_val')
        for i in range(len(ds)):
            sample=ds[i];product=Path(sample['data_samples'].img_path).name.removesuffix('_rgb.npy');r=selected[product]
            truth,valid,x,oldx=p83.full_features(r,engine.indices[product],z);t=truth[valid];positions=np.flatnonzero(valid)
            torch.cuda.synchronize();t0=time.monotonic();aligned=engine.forward(sample,product);torch.cuda.synchronize();forward_seconds+=time.monotonic()-t0
            f=aligned.cpu().numpy();del aligned
            permutation=rng.permutation(len(t));pp=dest/(product+'_permutation.npy');np.save(pp,permutation,allow_pickle=False)
            score=predict_logit(oldx,gate_model);shadow=score>=gate_model['threshold'];files={};check=np.linspace(0,len(t)-1,min(512,len(t)),dtype=np.int64)
            cp=dest/(product+'_check_base.npy');np.save(cp,x[check],allow_pickle=False)
            files['check_base']=dict(path=str(cp.relative_to(ROOT)),sha256=digest(cp))
            for name in NAMES:
                perm=None if name=='F_real' else permutation
                logits=aligned_logits(x,f,positions,perm,models[name]);cloud=(np.exp(-np.logaddexp(0,-logits))>=.5).astype(np.uint8);pred=np.where(shadow,2,cloud).astype(np.uint8)
                row=group_row(name,r,t,pred,refs[product]['ap']);checks(row,refs[product]);assert np.array_equal(pred==2,shadow)
                assert row['shadow_to_surface']+row['shadow_to_cloud']==refs[product]['shadow_to_surface']+refs[product]['shadow_to_cloud'];rows[name].append(row)
                fp=dest/(product+'_'+name+'_logits.npy');np.save(fp,logits,allow_pickle=False)
                loc=positions[check] if perm is None else positions[perm[check]];sampled=f.reshape(f.shape[0],-1)[:,loc].T
                path=dest/(product+'_'+name+'_check_F.npy');np.save(path,sampled,allow_pickle=False)
                files[name]=dict(logits_path=str(fp.relative_to(ROOT)),logits_sha256=digest(fp),check_F_path=str(path.relative_to(ROOT)),check_F_sha256=digest(path))
                if name=='F_shuffle':
                    # A full-resolution, unpermuted projection of this same model
                    # allows independent verification of every shuffled vector's
                    # linear contribution without saving dense high-dimensional F.
                    m=models[name]
                    base_part=((x-np.array(m['mean'][:4]))/np.array(m['scale'][:4]))@np.array(m['theta'][:4])+m['theta'][-1]
                    unpermuted=aligned_logits(x,f,positions,None,m)-base_part
                    error=float(np.max(np.abs(logits-(base_part+unpermuted[permutation]))));assert error<=1e-6
                    pr=dest/(product+'_shuffle_unpermuted_F_projection.npy');np.save(pr,unpermuted,allow_pickle=False)
                    files[name].update(unpermuted_F_projection_path=str(pr.relative_to(ROOT)),unpermuted_F_projection_sha256=digest(pr),shuffle_all_pixel_error=error)
                m=models[name];expected=((np.column_stack([x[check],sampled])-np.array(m['mean']))/np.array(m['scale']))@np.array(m['theta'][:-1])+m['theta'][-1]
                assert np.max(np.abs(logits[check]-expected))<=1e-6
            audits.append(dict(product=product,group_id=r['group_id'],files=files,check_indices=check.tolist(),permutation_path=str(pp.relative_to(ROOT)),permutation_sha256=digest(pp),
                original_probability_index=engine.indices[product],image_sha256=r['member_sha256']['image'],valid_pixels=len(t),all_shadow_invariants=True))
            del f,x;print('Phase84 locked development evaluation',i+1,len(ds),product,flush=True)
    engine.times['development_total_seconds']=time.monotonic()-start;engine.times['development_forward_and_alignment_seconds']=forward_seconds
    write(ROOT/'development_audit.json',dict(rows=audits,numerical_check_points_per_group=512,check_rule='linspace over original valid raster order; reference categories not used',tolerance=1e-6))
    engine.finish()
    from run_phase82_analysis import aggregate
    import run_phase82_analysis as agg
    agg.interval=interval
    policies={n:aggregate(old['policies'][n]['rows'],old['policies'][n]['pooled_ap']) for n in ('LL','R4','LM')}
    for n in NAMES:policies[n]=aggregate(rows[n],old['policies']['LL']['pooled_ap'])
    comparisons={};pairs=[]
    for a,b in [('F_real','R4'),('F_real','F_shuffle'),('F_real','LL')]:
        by={r['product']:r for r in policies[b]['rows']};d=[]
        for r in policies[a]['rows']:
            v=r['miou_percent']-by[r['product']]['miou_percent'];d.append(v);pairs.append(dict(comparison=a+' minus '+b,product=r['product'],group_id=r['group_id'],delta_mIoU_pp=v))
        comparisons[a+' minus '+b]=dict(macro_mIoU_pp=interval(d),pooled_mIoU_pp=policies[a]['miou_percent']-policies[b]['miou_percent'],worst_group_delta_pp=min(d),
            worst_products=[r['product'] for r,v in zip(policies[a]['rows'],d) if v==min(d)],improving=sum(v>0 for v in d),stable=sum(v==0 for v in d),declining=sum(v<0 for v in d))
    g=gate(comparisons['F_real minus LL']['pooled_mIoU_pp'],comparisons['F_real minus R4']['pooled_mIoU_pp'],comparisons['F_real minus F_shuffle']['pooled_mIoU_pp'],comparisons['F_real minus LL']['macro_mIoU_pp']['mean'],True)
    write(ROOT/'phase84_report.json',dict(policies=policies,comparisons=comparisons,gate=g,primary='F_real minus R4',capacity_control='F_real minus F_shuffle',formal_confirmation=False))
    csvwrite('paired_differences.csv',pairs);table=[];flat=[]
    for n,p in policies.items():
        table.append(dict(policy=n,macro_mIoU_percent=p['macro_miou_percent']['mean'],pooled_mIoU_percent=p['miou_percent'],
            Surface_IoU_percent=p['iou_percent'][0],Cloud_IoU_percent=p['iou_percent'][1],Shadow_IoU_percent=p['iou_percent'][2],
            Surface_to_Cloud=p['confusion'][0][1],Cloud_to_Surface=p['confusion'][1][0],Shadow_to_Surface=p['shadow_to_surface'],Shadow_to_Cloud=p['shadow_to_cloud'],
            macro_AP_percent=p['macro_ap']['mean']*100,macro_recall_percent=p['macro_recall']['mean']*100,macro_FPR_percent=p['macro_fpr']['mean']*100,FPR_upper95_percent=p['macro_fpr']['upper95_one_sided']*100))
        for r in p['rows']:
            item={k:v for k,v in r.items() if k not in ('confusion','reference_support','iou_percent')}
            for i,cl in enumerate(('Surface','Cloud','Shadow')):
                item[cl+'_reference_pixels']=r['reference_support'][i];item[cl+'_IoU_percent']=r['iou_percent'][i]
                for j,out in enumerate(('Surface','Cloud','Shadow')):item[cl+'_to_'+out]=r['confusion'][i][j]
            flat.append(item)
    csvwrite('summary.csv',table);csvwrite('development_groups.csv',flat)
    write(ROOT/'confusion_matrices.json',{n:dict(pooled=p['confusion'],rows=[dict(product=r['product'],confusion=r['confusion']) for r in p['rows']]) for n,p in policies.items()})
    write(ROOT/'invariant_checks.json',dict(status='passed',groups=32,strategies=2,all_Shadow_scores_masks_AP_recall_FPR_precision_IoU_and_total_misses_exact=True,geometry_chain_preserved=True,
        state_unchanged=True,base_sample_features_labels_hash_exact=True,base_standardization_exact=True,numerical_check_points=32*512,tolerance=1e-6))
    lines=['# Phase84：冻结空间特征的非云影读出','','## 1 数据结果','',
        f"153 fit代表沿用Phase83的626688个参考非云影抽样点；32 development代表。原LoRA3500冻结，零网络训练、185次特征提取、两个浅层拟合。实际mask_features形状/通道：{engine.expected['shape']}，全部通道使用。",'',
        '|策略|宏mIoU %|池化mIoU %|Surface IoU %|Cloud IoU %|Shadow IoU %|','|---|---:|---:|---:|---:|---:|']
    for r in table:lines.append('|'+r['policy']+'|'+'|'.join(f'{r[k]:.4f}' for k in ['macro_mIoU_percent','pooled_mIoU_percent','Surface_IoU_percent','Cloud_IoU_percent','Shadow_IoU_percent'])+'|')
    for n,c in comparisons.items():lines+=['',f"{n}: 宏mIoU {c['macro_mIoU_pp']['mean']:+.4f}个百分点，95%描述区间 {c['macro_mIoU_pp']['ci95']}；池化 {c['pooled_mIoU_pp']:+.4f}；最差逐组 {c['worst_group_delta_pp']:+.4f} {c['worst_products']}。"]
    lines+=['','## 2 结论探讨','',f"固定探索晋级门：{g['checks']}；保留真实F读出：{g['retain_F_real']}。任一失败即关闭本轮固定读出，不加层/通道/非线性/调参。打乱F仅容量对照，不能成为部署赢家。",'',
        f"云影评分/掩膜/全部指标与LL相同。宏FPR {policies['LL']['macro_fpr']['mean']*100:.4f}%，84011单侧95%上界 {policies['LL']['macro_fpr']['upper95_one_sided']*100:.4f}%；Phase83原上界 {old['policies']['LL']['macro_fpr']['upper95_one_sided']*100:.4f}%。不因读出改善宣称原误报约束通过。",'',
        '区间按32相关组配对bootstrap10000次/84011，仅描述反复探索development的结果，不是独立确认，也不是线性读出的方法创新证明。union>0类别平均规则未改，逐组表列有效类别数。', '',
        'F从同次前向pixel_decoder output[0]取得；捕获真实mask-to-head插值参数，再调用原postprocess_result保持裁边/翻转/第二次插值。原三类概率逐值不变；未将低分辨率线性读出替代两次插值。fit仅保存原抽样点F；dev逐景处理并仅存读出logits、置换及512固定检查点，未保存185景完整高维特征。', '',
        '输入/抽样/置换/参数与标准化锁、提取耗时、全程CUDA峰值和Linux CPU RSS峰值见JSON；权重仍远端work_dirs，不同步或打包。']
    (ROOT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    write(ROOT/'COMPLETED.json',dict(status='completed',report_sha256=digest(ROOT/'phase84_report.json'),new_forwards=185,shallow_fits=2,network_training=0,no_confirmations=True,no_work_dirs_sync=True))

def main():
    deadline();base_check();assert not ROOT.exists();ROOT.mkdir()
    bindings=read(REPO/'src/research_plans/PHASE84_INPUT_BINDINGS_20261011.json')
    for name,parent in [('phase80',P80),('phase83',p83.ROOT)]:
        for rel,h in bindings[name].items():assert digest(parent/rel)==h,rel
    baseline()
    files=[PROTOCOL,'src/research_plans/PHASE84_INPUT_BINDINGS_20261011.json','src/tools/run_phase84.py','src/tools/verify_phase84.py','src/tools/test_phase84.py','src/tools/run_phase84.sh',
        'src/tools/run_phase83.py','src/tools/run_phase82_analysis.py','src/tools/run_phase81.py','src/tools/phase65d_metrics.py','src/tools/phase65d_recovery_math.py','src/tools/run_phase65d_recovery.py','src/tools/run_phase80_training.py']
    write(ROOT/'input_configuration_lock.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),code_files={n:digest(REPO/n) for n in files},previous=bindings,
        seed=84011,fit_representatives=153,development_groups=32,no_confirmation=True,no_network_training=True))
    engine=Extractor();x,y,f,perms,sizes=extract_fit(engine);fit_models(x,y,f,perms,sizes);del x,y,f,perms,sizes
    evaluate(engine);print('Phase84 completed',flush=True)
if __name__=='__main__':main()

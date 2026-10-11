"""Phase85 one extraction/fit per selected seed; fixed historical crossed references."""
import argparse,hashlib,io,sys,zipfile,csv,time,subprocess
import numpy as np
from phase85_common import *
from run_phase65d_recovery import records,prepare_verify,load,features,brief
from phase65d_recovery_math import logit_score,fit_model,predict_logit,group_threshold
from phase65d_metrics import curve,confusion,measures,corrected
from verify_phase78_local import independent_AP
from run_phase81 import group_row,checks
from run_phase80_analysis import rows_index

def extract():
    import torch
    sys.path.insert(0,str(REPO/'src'))
    from run_phase85_training import setup
    setup()
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmseg.registry import MODELS,DATASETS
    done=read(ROOT/'training_completed.json');ck=Path(done['best_checkpoint'])
    assert digest(ck)==done['best_checkpoint_sha256']
    cfg=Config.fromfile(str(REPO/'src/configs/protocol/phase80_lora_rgb.py'))
    model=MODELS.build(cfg.model);model.init_weights()
    model.load_state_dict(torch.load(ck,map_location='cpu')['state_dict'],strict=True)
    model.train();assert all(permitted_lora(n) for n,p in model.named_parameters() if p.requires_grad)
    model.eval().cuda();torch.manual_seed(SEED);np.random.seed(SEED)
    for cohort in ['fit_representatives','development_val']:
        dest=ROOT/cohort;dest.mkdir()
        selected=records(cohort);ds=DATASETS.build(dict(cfg.val_dataloader.dataset,cohort=cohort))
        rows=[]
        for i in range(len(ds)):
            sample=ds[i];product=Path(sample['data_samples'].img_path).name.removesuffix('_rgb.npy')
            r=selected[product];prepare_verify(r);truth=np.load(PREP/r['mask_path'])
            with torch.no_grad():p=model.test_step(pseudo_collate([sample]))[0]
            native=p.seg_logits.data.cpu().numpy();score=p.seg_logits.data.softmax(0).cpu().numpy()[2].astype(np.float32)
            pred=p.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
            assert np.isfinite(native).all() and np.isfinite(score).all() and score.shape==truth.shape
            assert np.array_equal(pred,native.argmax(0))
            files={}
            for key,arr in dict(score=score,pred=pred,nonshadow=native[:2].argmax(0).astype(np.uint8)).items():
                path=dest/(product+'_'+key+'.npy');np.save(path,arr,allow_pickle=False)
                files[key]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path))
            rows.append(dict(product=product,group_id=r['group_id'],files=files,
                argmax_confusion=confusion(truth,pred).tolist()))
            print('Phase85 best LoRA forward',cohort,i+1,len(ds),product,flush=True)
        assert {r['product'] for r in rows}==set(selected)
        write(dest/'index.json',dict(rows=rows,manifest_sha256=MANIFEST_SHA,
            checkpoint_sha256=done['best_checkpoint_sha256'],score_definition='softmax of aggregated three-class Mask2Former scores'))

def fit():
    # Exact reuse of the old deterministic sampling positions and spectral columns.
    base_check();fitrows=records('fit_representatives')
    li=rows_index(ROOT,'fit_representatives/index.json')
    previous=read(RECOVERY/'fit_feature_index.json');old={r['product']:r for r in previous['rows']}
    assert set(old)==set(li)==set(fitrows)
    dest=ROOT/'fit_features';dest.mkdir();xs=[];ys=[];ws=[];audit=[]
    for product,r in fitrows.items():
        prepare_verify(r);truth=np.load(PREP/r['mask_path']);valid=truth!=255;y=truth[valid]==2
        prior=old[product];path=RECOVERY/prior['feature_path'];assert digest(path)==prior['feature_sha256']
        spectral=np.load(path,mmap_mode='r')[:,2:5]
        x=np.column_stack([logit_score(load(ROOT,li[product],'score')[valid]),spectral]).astype(np.float32)
        assert np.isfinite(x).all() and len(x)==prior['valid_pixels']
        seed=65081+int(hashlib.sha256(product.encode()).hexdigest()[:8],16)
        idx=np.random.default_rng(seed).choice(len(x),min(8192,len(x)),replace=False)
        assert hashlib.sha256(idx.tobytes()).hexdigest()==prior['sample_indices_sha256']
        assert int(y[idx].sum())==prior['sample_positive']
        p=dest/(product+'.npy');np.save(p,x,allow_pickle=False)
        audit.append(dict(product=product,group_id=r['group_id'],feature_path=str(p.relative_to(ROOT)),
            feature_sha256=digest(p),old_feature_sha256=prior['feature_sha256'],
            sample_indices_sha256=prior['sample_indices_sha256'],sample_count=len(idx),sample_positive=int(y[idx].sum()),valid_pixels=len(x)))
        xs.append(x[idx]);ys.append(y[idx]);ws.append(np.full(len(idx),1/(153*len(idx))))
    x=np.concatenate(xs).astype(np.float64);y=np.concatenate(ys).astype(float);w=np.concatenate(ws)
    assert hashlib.sha256(y.tobytes()).hexdigest()==previous['sample_y_sha256']
    write(ROOT/'fit_input_audit.json',dict(rows=audit,sample_count=len(x),sample_positive=int(y.sum()),
          sample_x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),sample_y_sha256=previous['sample_y_sha256'],equal_group_weights=True))
    models={}
    for name,cols in [('L+S',[0,1,2,3])]:
        model=fit_model(x[:,cols],y,w,1);neg=[]
        for r in audit:
            full=np.load(ROOT/r['feature_path'],mmap_mode='r')
            truth=np.load(PREP/fitrows[r['product']]['mask_path']);yy=truth[truth!=255]==2
            neg.append(predict_logit(full[:,cols],model)[~yy])
        model['threshold']=group_threshold(neg,.01)
        model['fit_macro_fpr']=float(np.mean([(v>=model['threshold']).mean() for v in neg]))
        assert model['fit_macro_fpr']<=.01+1e-10
        model['columns']=cols;models[name]=model;del neg
        print('Phase85 shallow frozen',name,model,flush=True)
    write(ROOT/'calibration_lock.json',dict(status='one_fit_only_model_frozen_before_development_shallow_evaluation',
        models=models,fit_input_audit_sha256=digest(ROOT/'fit_input_audit.json'),
        training_completed_sha256=digest(ROOT/'training_completed.json'),
        baseline_reproduction_sha256=digest(CONTROL85/'baseline_reproduction.json'),
        source_sha256=SOURCE_SHA,manifest_sha256=MANIFEST_SHA,original_split_sha256=SPLIT_SHA,
        recovery_lock_sha256=RECOVERY_SHA,phase67_lock_sha256=digest(CONTROL/'control_lock.json'),
        analysis_code_sha256=digest(Path(__file__)),math_sha256=digest(Path(__file__).with_name('phase65d_recovery_math.py')),
        full_fit_negative_threshold=True,fit_group_weights='1/153 per representative',development_fitting=False,
        no_confirmations_read=True,no_hyperparameter_search=True))



def baseline():
    base_check();assert not PARENT.exists();PARENT.mkdir()
    bindings=read(REPO/'src/research_plans/PHASE85_INPUT_BINDINGS_20261011.json')
    for name,parent in [('phase80',P80),('phase81',PHASE81)]:
        for rel,h in bindings[name].items():assert digest(parent/rel)==h,rel
    import run_phase82_analysis as old
    old.RUNCONTROL=CONTROL85
    old.baseline()  # CPU cache reproduction only; no Phase82 training or inference.
    files=['src/research_plans/PHASE85_LORA_SEED_REPLICATION_20261011.md','src/research_plans/PHASE85_INPUT_BINDINGS_20261011.json',
      'src/tools/phase85_common.py','src/tools/run_phase85_training.py','src/tools/run_phase85_analysis.py','src/tools/verify_phase85.py','src/tools/run_phase85.sh','src/tools/test_phase85.py',
      'src/tools/run_phase80_analysis.py','src/tools/run_phase80_training.py','src/tools/phase80_common.py','src/tools/run_phase82_analysis.py','src/tools/run_phase81.py',
      'src/tools/run_phase65d_recovery.py','src/tools/phase65d_metrics.py','src/tools/phase65d_recovery_math.py','src/tools/verify_phase78_local.py',
      'src/configs/protocol/phase80_lora_rgb.py','src/configs/protocol/phase65a_explore_rgb.py','src/configs/protocol/phase64_source_parent_rgb.py',
      'src/cloud_adapter/models/backbones/phase64_lora_dinov2.py','src/cloud_adapter/models/backbones/phase64_lora_layers.py','src/cloud_adapter/models/segmentors/frozen_encoder_decoder.py',
      'src/cloud_adapter/datasets/phase65_catalogue.py']
    write(PARENT/'input_configuration_lock.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
      code_sha256={n:digest(REPO/n) for n in files},bindings=bindings,training_seeds=[66,67],new_training_runs=2,new_shallow_fits=2,
      no_confirmation=True,no_alcd=True,no_work_dirs_transfer=True,shallow_sampling_seed='original 65081+product hash; same indices for both seeds',bootstrap_seed=85011))

def evaluate_seed():
    lock=read(ROOT/'calibration_lock.json');assert digest(Path(__file__))==lock['analysis_code_sha256']
    model=lock['models']['L+S'];assert set(lock['models'])=={'L+S'}
    li=rows_index(ROOT,'development_val/index.json');mi=rows_index(OLD,'development_val/index_msre.json')
    rows={n:[] for n in ['LL'+str(SEED),'LM'+str(SEED)]};pools=[];ys=[];invariants=[];audits=[]
    dest=ROOT/'development_scores';dest.mkdir()
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for product,r in records('development_val').items():
            prepare_verify(r);t0=np.load(PREP/r['mask_path']);v=t0!=255;t=t0[v]
            b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
            raw=np.load(io.BytesIO(b),allow_pickle=False)
            x=np.column_stack([logit_score(load(ROOT,li[product],'score')[v]),raw[...,7][v],raw[...,11][v],raw[...,12][v]]).astype(np.float32)
            score=predict_logit(x,model);ap=curve(score,t==2)['ap'];other=independent_AP(t==2,score)
            assert (ap is None and other is None) or abs(ap-other)<1e-12
            preds=[]
            for n,winner in [('LL'+str(SEED),load(ROOT,li[product],'nonshadow')[v]),('LM'+str(SEED),load(OLD,mi[product],'nonshadow')[v])]:
                pred=corrected(score,model['threshold'],winner);preds.append(pred)
                rows[n].append(group_row(n,r,t,pred,ap))
            assert np.array_equal(preds[0]==2,preds[1]==2);checks(rows['LL'+str(SEED)][-1],rows['LM'+str(SEED)][-1])
            assert sum(rows['LL'+str(SEED)][-1][k] for k in ['shadow_to_surface','shadow_to_cloud'])==sum(rows['LM'+str(SEED)][-1][k] for k in ['shadow_to_surface','shadow_to_cloud'])
            path=dest/(product+'.npy');np.save(path,score,allow_pickle=False)
            audits.append(dict(product=product,path=str(path.relative_to(ROOT)),sha256=digest(path),lora_cache=li[product],msre_cache=mi[product],image_sha256=r['member_sha256']['image']))
            pools.append(score);ys.append(t==2);invariants.append(dict(product=product,shadow_mask_exact=True,all_shadow_metrics_exact=True))
            print('Phase85 seed evaluation',SEED,len(ys),32,product,flush=True)
    pool=curve(np.concatenate(pools),np.concatenate(ys))['ap']
    assert abs(pool-independent_AP(np.concatenate(ys),np.concatenate(pools)))<1e-12
    write(ROOT/'development_input_audit.json',dict(rows=audits))
    write(ROOT/'seed_report.json',dict(seed=SEED,rows=rows,pooled_ap=pool,calibration_lock_sha256=digest(ROOT/'calibration_lock.json'),invariants=invariants))

def aggregate(rows,pool_ap):
    shadow=[r for r in rows if r['positive']];zero=[r for r in rows if not r['positive']]
    pooled=measures(np.sum([r['confusion'] for r in rows],axis=0))
    return dict(rows=rows,**pooled,pooled_ap=pool_ap,macro_ap=interval([r['ap'] for r in shadow]),
      macro_recall=interval([r['recall'] for r in shadow]),macro_precision=interval([r['precision'] for r in rows if r['precision'] is not None]),
      macro_fpr=interval([r['fpr'] for r in rows]),macro_miou_percent=interval([r['miou_percent'] for r in rows]),
      shadow_fpr=interval([r['fpr'] for r in shadow]),no_shadow_fpr=interval([r['fpr'] for r in zero]),
      fpr_above_one_percent_groups=sum(r['fpr']>.01 for r in rows),zero_recall_groups=[r['product'] for r in shadow if r['recall']==0],
      n_groups=len(rows),shadow_groups=len(shadow),no_shadow_groups=len(zero))

def csvwrite(name,rows):
    with (PARENT/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def report():
    refs=read(PHASE81/'phase81_report.json');policies={}
    for dest,src in [('MM','MM'),('LL65','LL'),('LM65','LM')]:
        rows=[dict(r,policy=dest) for r in refs['policies'][src]['rows']]
        policies[dest]=aggregate(rows,refs['policies'][src]['pooled_ap'])
    costs={};traces=[]
    for seed in [66,67]:
        parent=PARENT/('seed'+str(seed));sr=read(parent/'seed_report.json');done=read(parent/'training_completed.json')
        for n,rows in sr['rows'].items():policies[n]=aggregate(rows,sr['pooled_ap'])
        costs[str(seed)]={k:done[k] for k in ['seconds','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','best_iteration','best_checkpoint','best_checkpoint_sha256','trainable_parameters']}
        costs[str(seed)]['preflight']=read(parent/'preflight.json')
        traces.append(read(parent/'random_stream.json'))
    assert traces[0]['first_eight_batches']!=traces[1]['first_eight_batches'],'Seeds must have distinct actual random streams'
    names=['MM','LL65','LM65','LL66','LM66','LL67','LM67'];pairs=[];comparisons={};new_deltas=[]
    for seed in [66,67]:
        for a in ['LL'+str(seed),'LM'+str(seed)]:
            b='MM';d=[]
            for x,y in zip(policies[a]['rows'],policies[b]['rows']):
                assert x['group_id']==y['group_id']
                row=dict(comparison=a+' minus MM',product=x['product'],group_id=x['group_id'],positive=x['positive'],
                  delta_ap_pp=None if not x['positive'] else 100*(x['ap']-y['ap']),
                  delta_recall_pp=None if not x['positive'] else 100*(x['recall']-y['recall']),
                  delta_FPR_pp=100*(x['fpr']-y['fpr']),delta_mIoU_pp=x['miou_percent']-y['miou_percent'])
                pairs.append(row);d.append(row)
            aps=[r['delta_ap_pp'] for r in d if r['delta_ap_pp'] is not None];rec=[r['delta_recall_pp'] for r in d if r['delta_recall_pp'] is not None]
            comparisons[a+' minus MM']=dict(AP_pp=interval(aps),recall_pp=interval(rec),FPR_pp=interval([r['delta_FPR_pp'] for r in d]),
              macro_mIoU_pp=interval([r['delta_mIoU_pp'] for r in d]),pooled_mIoU_pp=policies[a]['miou_percent']-policies[b]['miou_percent'],
              worst_AP_pp=min(aps),worst_recall_pp=min(rec),worst_mIoU_pp=min(r['delta_mIoU_pp'] for r in d),
              improved=sum(v>1 for v in aps),stable=sum(abs(v)<=1 for v in aps),declined=sum(v< -1 for v in aps))
            if a.startswith('LL'):new_deltas.append(aps)
    # Average the two fixed-seed contrasts WITHIN group before one paired group resample.
    seedmean=np.mean(new_deltas,axis=0);assert len(seedmean)==7
    invariants=[]
    for seed in [66,67]:
        for x,y in zip(policies['LL'+str(seed)]['rows'],policies['LM'+str(seed)]['rows']):
            checks(x,y);invariants.append(dict(seed=seed,product=x['product'],passed=True))
    rules=gate(policies);rules['all_crossed_shadow_invariants']=len(invariants)==64
    r=dict(scope='conditional robustness vs one fixed MsRE on repeatedly used development',formal_confirmation=False,
      policies=policies,comparisons=comparisons,new_seed_mean_AP_gain_pp=interval(seedmean),new_seed_group_mean_AP_gains_pp=seedmean.tolist(),
      group_order=[x['group_id'] for x in policies['MM']['rows'] if x['positive']],gate=rules,retain_branch=all(rules.values()),
      costs=costs,seed65_in_promotion_mean=False,new_network_trainings=2,new_shallow_fits=2,independent_statistical_groups=7,work_dirs_transferred=False)
    write(PARENT/'phase85_report.json',r);write(PARENT/'invariant_checks.json',dict(status='passed',rows=invariants,actual_random_streams_different=True))
    write(PARENT/'confusion_matrices.json',{n:dict(pooled=p['confusion'],groups=[dict(product=g['product'],confusion=g['confusion']) for g in p['rows']]) for n,p in policies.items()})
    summary=[];flat=[]
    for n in names:
        p=policies[n];summary.append(dict(policy=n,macro_AP_percent=100*p['macro_ap']['mean'],pooled_AP_percent=100*p['pooled_ap'],
          macro_recall_percent=100*p['macro_recall']['mean'],macro_precision_percent=100*p['macro_precision']['mean'],
          macro_FPR_percent=100*p['macro_fpr']['mean'],FPR_upper95_percent=100*p['macro_fpr']['upper95_one_sided'],
          shadow_group_FPR_percent=100*p['shadow_fpr']['mean'],no_shadow_FPR_percent=100*p['no_shadow_fpr']['mean'],
          macro_mIoU_percent=p['macro_miou_percent']['mean'],pooled_mIoU_percent=p['miou_percent'],
          Surface_IoU_percent=p['iou_percent'][0],Cloud_IoU_percent=p['iou_percent'][1],Shadow_IoU_percent=p['iou_percent'][2],
          zero_recall_groups=len(p['zero_recall_groups']),FPR_above_budget_groups=p['fpr_above_one_percent_groups']))
        for row in p['rows']:
            item={k:v for k,v in row.items() if k not in ('confusion','iou_percent','reference_support')}
            for i,c in enumerate(['Surface','Cloud','Shadow']):
                item[c+'_reference_pixels']=row['reference_support'][i];item[c+'_IoU_percent']=row['iou_percent'][i]
                for j,b in enumerate(['Surface','Cloud','Shadow']):item[c+'_to_'+b]=row['confusion'][i][j]
            flat.append(item)
    csvwrite('summary.csv',summary);csvwrite('development_groups.csv',flat);csvwrite('paired_differences.csv',pairs)
    lines=['# Phase85：原LoRA方案的训练随机性复测','','## 1 数据结果','',
      '原Catalogue 153 fit组155产品训练，153代表浅层拟合，32 development代表选择/评价（7组有云影）。seed66/67各一次4000步、各一次L+S拟合；seed65仅历史参考。', '',
      '|组合|宏AP %|池化AP %|宏召回 %|宏FPR %|宏mIoU %|池化mIoU %|','|---|---:|---:|---:|---:|---:|---:|']
    for x in summary:lines.append('|'+x['policy']+'|'+ '|'.join(f"{x[k]:.4f}" for k in ['macro_AP_percent','pooled_AP_percent','macro_recall_percent','macro_FPR_percent','macro_mIoU_percent','pooled_mIoU_percent'])+'|')
    for seed in [66,67]:
        c=comparisons['LL'+str(seed)+' minus MM'];lines.append(f"\nseed{seed}选中步骤{costs[str(seed)]['best_iteration']}；相对MM宏AP增量{c['AP_pp']}；召回差{c['recall_pp']}；最差AP差{c['worst_AP_pp']:.6f}pp。")
    lines+=['',f"两个新seed组内AP增量平均后，在7组bootstrap：{r['new_seed_mean_AP_gain_pp']}。",'', '## 2 结论探讨','',f"固定继续门：{rules}；保留分枝：{r['retain_branch']}。",'',
      '这是相对单一固定MsRE的条件稳健性探索；两seed不能概括随机性总体，7相关组不能拆成14/21独立样本。development已反复探索，描述区间不是独立确认或正式非劣。无论结果不追加seed、不换checkpoint、不调阈值/超参数。', '',
      'LL与LM同seed云影评分/掩膜/指标逐值相同。LM需要本seed LoRA与原MsRE两套网络；训练/显存/两网权重成本见成本记录，不能把其成本当成单网。FPR点估计满足预算与单侧95%上界满足预算分别报告。', '',
      '模型和全部8步checkpoint保留远端work_dirs，仅shared小报告可交付；未读取Phase66/65b/65c/sealed/ALCD，不取新数据。']
    (PARENT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    write(PARENT/'COMPLETED.json',dict(status='completed',report_sha256=digest(PARENT/'phase85_report.json'),training_runs=2,shallow_fits=2))
    print('Phase85 completed retain_branch',r['retain_branch'],flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['baseline','seed','report']);args=a.parse_args();deadline()
    if args.action=='baseline':baseline()
    elif args.action=='seed':
        t=time.monotonic();extract();fit();evaluate_seed()
        write(ROOT/'analysis_cost.json',dict(seconds=time.monotonic()-t,selected_checkpoint_forwards=185,new_shallow_fits=1))
    else:report()
if __name__=='__main__':main()

"""Post-seal data delivery, cost accounting and explicit metric dictionary."""
import csv,hashlib,json,shutil,struct
from pathlib import Path
import numpy as np
from phase120_metrics import table
R=Path(__file__).resolve().parents[2];O=R/'src/research_results/phase122_20261015';U=R/'outputs/phase122/phase122_20261015'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def rows(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
    de=U/'delivery';controller=load(de/'CONTROLLER_STATUS.json');assert controller['status']=='completed_pending_local_reference_join'
    train=load(de/'train_C/TRAINING_RECEIPT.json');assert train['seed']==12201 and train['steps']==1000 and train['encoder_unchanged'] and train['decoder_updated']
    assert train['trainable_parameters']==14135988 and not train['teacher_state_sha256'] and train['training_calls']==1000
    model=U/'C_decoder_final.safetensors';assert sha(model)==train['final_model_sha256']==load(de/'TRAINING_SEAL.json')['models']['C']
    raw=model.read_bytes();n=struct.unpack('<Q',raw[:8])[0];header=json.loads(raw[8:8+n]);end=0;count=0
    for name,item in sorted(((k,v) for k,v in header.items() if k!='__metadata__'),key=lambda x:x[1]['data_offsets'][0]):
        a,b=item['data_offsets'];assert a==end;end=b
        v=np.frombuffer(raw[8+n+a:8+n+b],dtype={'F32':'<f4','I64':'<i8'}[item['dtype']]).reshape(item['shape'])
        assert np.isfinite(v).all();count+=1
    assert count==61 and end==len(raw)-8-n
    stages=('train_C','R_Calibration','C_Calibration','C_Catalogue','C_PixBox');cost=[]
    for stage in stages:
        s=load(de/stage/'RUN_STATUS.json')
        cost.append(dict(stage=stage,status=s['status'],calls=s['all_calls'],construction_calls=s['construction_calls'],training_calls=s.get('training_calls',0),
            gate_calls=s.get('gate_calls',0),main_calls=s.get('main_calls',0),warmup_calls=s.get('warmup_calls',0),elapsed_seconds=s['elapsed_seconds'],
            CPU_peak_bytes=s['rss_peak_bytes'],GPU_reserved_peak_bytes=s['gpu_reserved_peak_bytes'],host_min_free_bytes=s.get('host_min_free_bytes'),intermediate_peak_bytes=s.get('intermediate_peak_bytes',0)))
    calls=sum(v['calls'] for v in cost);assert calls==2612==controller['calls_completed'] and controller['elapsed_seconds']<=7200
    table(O/'actual_cost.csv',cost)
    cal=load(de/'THRESHOLD_LOCK.json');assert cal['models_and_thresholds_before_evaluation'] and cal['no_evaluation_reference_used']
    for name in ('TRAINING_SEAL.json','THRESHOLD_LOCK.json','PAIRED_CROP_SCHEDULE.json','FIT_INPUT_LOCK.json','ALL_SCORES_SEALED.json','CONTROLLER_STATUS.json','calibration_CPU_RESOURCE_RECEIPT.json'):
        shutil.copyfile(de/name,O/name)
    shutil.copyfile(de/'Catalogue_COMMON_SUPPORT_LOCK.json',O/'REMOTE_Catalogue_COMMON_SUPPORT_LOCK.json')
    for name in ('TRAINING_RECEIPT.json','PARAMETER_WHITELIST.json','INITIAL_GRADIENT_GATE.json'):shutil.copyfile(de/'train_C'/name,O/('C100_'+name))
    scene=[];checks=[]
    for stage in stages[1:]:
        info=load(de/stage/'MODEL_INFO.json');assert info['models']==1 and info['total_parameters']==18039132 and info['deployed_teacher_models']==0
        shutil.copyfile(de/stage/'SCORES_SEALED.json',O/(stage+'_SCORES_SEALED.json'))
        for v in load(de/stage/'SCENE_RECEIPTS.json'):
            assert v['coverage_gradient_exact'] and v['max_confidence_error']<=1e-5 and v['max_raw_mean_error']<=1e-5
            checks.append(v['max_confidence_error']);scene.append(v)
    assert len(scene)==154
    save(O/'INFERENCE_CPU_ACCEPTANCE.json',dict(status='passed',scenes=len(scene),max_independent_confidence_error=max(checks),
        deployed_parameters=18039132,teacher_in_deployment=0,raw_logit_mosaic=True,model_calls_in_verifier=0))
    save(O/'MODEL_RECONSTRUCTION.json',dict(policy='C100',seed=12201,filename=model.name,local_path=model.relative_to(R).as_posix(),
        remote_path='/home/scv/shared/phase122_20261015/'+model.name,sha256=sha(model),bytes=model.stat().st_size,finite_tensors=count,
        original_asset_lock='src/research_results/phase111_20261014/PRE_EXECUTION_LOCK.json',
        restore='original official RegNet with original author loader; overlay decoder only, strict=False, missing keys frozen encoder only',
        inputs=['B04','B03','B8A'],normalization='original per-band nonzero author channel_norm',
        inference=dict(patch_size=1000,overlap=300,temperature=1,confidence='softmax + float32(.001), clipped [.001,.999]'),
        frozen_thresholds=cal['thresholds'],comparison_domain='float64',Git_weights_uploaded=False))
    worst=[];readable=[];baseline={}
    for pool in ('Catalogue','PixBox'):
        per=rows(O/(pool+'_per_group.csv'));groups=sorted({v['group_id'] for v in per});by={(v['group_id'],v['setting']):v for v in per}
        for gid in groups:
            rr=[by[gid,s] for s in ('R_05','R_cal','C100_05','C100_cal')]
            tile=','.join(sorted({v.split('_T')[1][:5] for v in rr[0]['products'].split(',')}))
            row=dict(pool=pool,tile=tile,group_id=gid,positive=rr[0]['positive'],negative=rr[0]['negative'])
            for v in rr:
                for k in ('AP','recall','FPR','Surface_FPR','Cloud_FPR'):row[v['setting']+'_'+k+'_pp']=float(v[k])*100 if v[k]!='' else None
            readable.append(row)
        for a,b in (('C100_cal','R_cal'),('R_cal','R_05'),('C100_cal','C100_05')):
            def dif(k):return [((float(by[gid,a][k])-float(by[gid,b][k]))*100,gid) for gid in groups if by[gid,a][k]!='' and by[gid,b][k]!='']
            aa=dif('AP');rec=dif('recall');fp=dif('FPR')
            worst.append(dict(pool=pool,comparison=a+'-'+b,min_AP_delta_pp=min(aa)[0],min_AP_group=min(aa)[1],
                min_recall_delta_pp=min(rec)[0],min_recall_group=min(rec)[1],max_FPR_increase_pp=max(fp)[0],max_FPR_increase_group=max(fp)[1]))
        old=next(v for v in rows(R/'src/research_results/phase121_20261015'/(pool+'_summary.csv')) if v['setting']=='R')
        now=next(v for v in rows(O/(pool+'_four_workpoints.csv')) if v['setting']=='R_05')
        unchanged=load(O/(pool+'_CPU_ACCEPTANCE.json'))['support_loss']==0
        if unchanged:
            for a,b in (('macro_AP_pp','macro_AP_pp'),('pooled_AP_pp','pooled_AP_pp'),('macro_recall_pp','macro_recall_pp'),('macro_negative_FP_pp','macro_FPR_pp')):assert abs(float(old[a])-float(now[b]))<1e-8
        baseline[pool]=dict(support_unchanged=unchanged,R_05_reproduced=unchanged)
    table(O/'group_operating_readable.csv',readable);table(O/'worst_group_damage.csv',worst)
    save(O/'BASELINE_REPRODUCTION.json',baseline)
    save(O/'DATA_DICTIONARY.json',dict(units='summary and differences ending pp are percentage points; per_group AP/recall/FPR are fractions',
        bootstrap='10000 related-group resamples seed122015; percentile two-sided2.5/97.5 and one-sided95th percentile upper; descriptive',
        AP='one value per model regardless of threshold, no-positive groups NA',
        strata='Surface label0 and Cloud label1, no-denominator groups NA; group means within stratum-supported groups',
        comparison_json_minimum_fields='generic *_worst_delta_pp fields store numerical minimum; for FPR harm use max_FPR_increase_pp in worst_group_damage.csv',
        thresholds='float32 confidence exactly promoted to float64; comparison and stored nextafter boundary float64',
        no_guarantee='1% calibration target is not an evaluation-side constraint or per-group guarantee'))
    cpu=load(O/'LOCAL_CPU_RESOURCE_RECEIPT.json');calres=load(de/'calibration_CPU_RESOURCE_RECEIPT.json')
    complete=dict(phase=122,status='completed_exploratory',model_calls=calls,training_steps=1000,training_groups=100,calibration_groups=53,
        seed=12201,thresholds=cal['thresholds'],controller_wall_seconds=controller['elapsed_seconds'],new_downloads=0,model_retries=0,
        CPU_peak_bytes=max(cpu['CPU_peak_bytes'],calres['rss_peak_bytes'],load(O/'CALIBRATION_INDEPENDENT_CPU_ACCEPTANCE.json')['CPU_peak_bytes'],max(v['CPU_peak_bytes'] for v in cost)),
        GPU_reserved_peak_bytes=max(v['GPU_reserved_peak_bytes'] for v in cost),
        minimum_remote_host_free_bytes=min(v['host_min_free_bytes'] for v in cost if v['host_min_free_bytes'] is not None),
        intermediate_peak_bytes=max(load(U/'DELIVERY_RECEIPT.json')['intermediate_bytes_after_package'],max(v['intermediate_peak_bytes'] for v in cost)),
        local_intermediate_peak_bytes=cpu['intermediate_peak_bytes'],
        labels_uploaded=0,work_dirs_transfers=0,Git_weights_uploaded=False,original_failed_split_preserved=True,
        new_independent_confirmation_claim=False)
    save(O/'COMPLETION.json',complete)
    text=['# Phase122 数据结果','','仅新训练一个C100，seed12201、1000步，从原RegNet初始化，100训练组与53校准组沿用Phase94身份。两个阈值在模型封存后、评价前冻结；未使用Phase121 oracle阈值。',
        '', '## 1．数据结果','',f"冻结阈值：R={cal['thresholds']['R']!r}；C100={cal['thresholds']['C']!r}。实际判定在float64域进行，展示四位小数不能用于替代锁文件的精确边界。",
        '', '| 池 | 模型/工作点 | 宏AP % | 池化AP % | 宏召回 % | 平均误报 % | 单侧95%误报上界 % | 最大组误报 % |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for pool in ('Catalogue','PixBox'):
        for v in rows(O/(pool+'_four_workpoints.csv')):text.append('| '+pool+' | '+v['setting']+' | '+' | '.join(f'{float(v[k]):.4f}' for k in ('macro_AP_pp','pooled_AP_pp','macro_recall_pp','macro_FPR_pp','FPR_U95_pp','max_group_FPR_pp'))+' |')
    text+=['','| 池 | 固定比较 | 平均召回差 pp [95%区间] | 平均误报差 pp [95%区间] |','|---|---|---|---|']
    for pool in ('Catalogue','PixBox'):
        for v in load(O/(pool+'_comparisons.json'))['rows']:
            rr,ff=v['recall'],v['FPR'];text.append(f"| {pool} | {v['comparison']} | {rr['mean']:.4f} [{rr['CI95'][0]:.4f}, {rr['CI95'][1]:.4f}] | {ff['mean']:.4f} [{ff['CI95'][0]:.4f}, {ff['CI95'][1]:.4f}] |")
    text+=['','主要探索比较为Catalogue的C100/Tcal−R/Tcal宏召回差，必须与实际误报和上界共同解释。AP在同模型两工作点完全不变。',
        'Catalogue32组，7组有正参考；PixBox16产品／10相关组，8组有正参考。无正类仍参加误报统计；Surface／Cloud各自没有分母时为NA。完整逐组、分层、零召回、超1%组数和区间见CSV。',
        '1%是53组上的经验校准目标，评价侧既未强制相等FPR，也未改阈值。PixBox误报仅指所选参考负类点；组均约束不代表逐景安全。区间由10000次相关组bootstrap得到，seed122015，仅描述。',
        '', '### 成本与验收','',f'实际{calls}次模型调用，仅一次拟合，控制器墙钟{controller["elapsed_seconds"]:.1f}秒。两个校准边界的完整负类计数、并列和相邻更低边界均经独立CPU全量复核（校准分数保留受控shared，源代码／分数／阈值SHA回执已核对）；两池参考标签只在本地接入。',
        f'CPU峰值{complete["CPU_peak_bytes"]/1e9:.3f}GB，GPU allocator预留峰值{complete["GPU_reserved_peak_bytes"]/1e9:.3f}GB，宿主可用采样最低{complete["minimum_remote_host_free_bytes"]/1e9:.3f}GB。',
        '61个decoder张量均有限，模型SHA在本地及shared核对。原encoder与buffer不变；部署仍为单RegNet，总18,039,132参数，训练14,135,988参数；不上传GitHub权重、标签或大数组，不同步work_dirs。',
        '100／53组历史上已暴露，只是本次C100训练未使用53组；原评价池暴露、训练地点交集未知及参考局限保留。C100与旧C153训练样本不同，不能把两者差异归因于校准单一因素。',
        '', '## 2．结论分析','','见ANALYSIS.md；与数据表分开。一个后续建议见NEXT_STEP.md，未自动执行。']
    with (O/'README.md').open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
    print(json.dumps(complete,indent=2))
if __name__=='__main__':main()

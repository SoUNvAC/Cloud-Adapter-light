"""Finalize Phase121 receipts and data; keep scientific discussion separate."""
import csv,hashlib,json,shutil,struct
from pathlib import Path
import numpy as np
from phase121_metrics import POLICIES,PAIRS,table
R=Path(__file__).resolve().parents[2];O=R/'src/research_results/phase121_20261015';U=R/'outputs/phase121/phase121_20261015'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def rows(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
    delivery=U/'delivery';controller=load(delivery/'CONTROLLER_STATUS.json')
    assert controller['status']=='completed_pending_local_PixBox_reference_join'
    seal=load(delivery/'TRAINING_SEAL.json');receipts={};tensor_receipts=[]
    for arm in ('C','D'):
        tr=load(delivery/('train_'+arm)/'TRAINING_RECEIPT.json');receipts[arm]=tr
        assert tr['seed']==12002 and tr['steps']==1000 and tr['encoder_unchanged'] and tr['teachers_unchanged'] and tr['decoder_updated']
        assert tr['teacher_student_disjoint_parameters'] and tr['trainable_parameters']==14135988
        assert tr['schedule_sha256']==sha(delivery/'PAIRED_CROP_SCHEDULE.json')
        path=U/(arm+'_decoder_final.safetensors');assert sha(path)==tr['final_model_sha256']==seal['models'][arm]
        raw=path.read_bytes();n=struct.unpack('<Q',raw[:8])[0];header=json.loads(raw[8:8+n]);end=0;count=0
        for name,item in sorted(((k,v) for k,v in header.items() if k!='__metadata__'),key=lambda kv:kv[1]['data_offsets'][0]):
            a,b=item['data_offsets'];assert a==end;end=b
            dtype={'F32':'<f4','I64':'<i8'}[item['dtype']]
            tensor=np.frombuffer(raw[8+n+a:8+n+b],dtype=dtype).reshape(item['shape'])
            assert np.isfinite(tensor).all();count+=1
        assert end==len(raw)-8-n and count==61
        tensor_receipts.append(dict(arm=arm,policy=arm+'12002',filename=path.name,local_path=path.relative_to(R).as_posix(),remote_path='/home/scv/shared/phase121_20261015/'+path.name,tensors=count,all_finite=True,bytes=len(raw),sha256=sha(path)))
    for f in (delivery/'aggregate').iterdir():
        if f.suffix in ('.csv','.json'):
            assert not (O/f.name).exists();shutil.copyfile(f,O/f.name)
    stages=('train_C','train_D','C_Catalogue','D_Catalogue','C_PixBox','D_PixBox');cost=[]
    for stage in stages:
        s=load(delivery/stage/'RUN_STATUS.json')
        cost.append(dict(stage=stage,status=s['status'],calls=s['all_calls'],construction_calls=s['construction_calls'],
            training_calls=s.get('training_calls',0),gate_calls=s.get('gate_calls',0),main_calls=s.get('main_calls',0),
            warmup_calls=s.get('warmup_calls',0),elapsed_seconds=s['elapsed_seconds'],rss_peak_bytes=s['rss_peak_bytes'],
            gpu_reserved_peak_bytes=s['gpu_reserved_peak_bytes'],host_min_free_bytes=s.get('host_min_free_bytes'),
            intermediate_peak_bytes=s.get('intermediate_peak_bytes',0)))
    calls=sum(v['calls'] for v in cost);assert calls==6483==controller['calls_completed']==load(delivery/'ALL_SCORES_SEALED.json')['actual_calls']
    assert controller['elapsed_seconds']<=14400
    cpu=load(delivery/'Catalogue_CPU_RESOURCE_RECEIPT.json');assert cpu['model_calls']==0 and cpu['rss_peak_bytes']<=16*1024**3
    table(O/'actual_cost.csv',cost)
    checks=[];bench=[];first=load(O/'PRE_EXECUTION_LOCK.json')['benchmark_used']
    for arm in ('C','D'):
        for pool in ('Catalogue','PixBox'):
            info=load(delivery/(arm+'_'+pool)/'MODEL_INFO.json')
            assert info['total_parameters']==18039132 and info['models']==1 and info['deployed_teacher_models']==0
            for v in load(delivery/(arm+'_'+pool)/'SCENE_RECEIPTS.json'):
                assert v['coverage_gradient_exact'] and v['max_confidence_error']<=1e-5 and v['max_raw_mean_error']<=1e-5
                checks.append(v['max_confidence_error'])
                if pool=='PixBox' and v['product_id']==first:
                    bench.append({k:v[k] for k in ('policy','product_id','repetition','model_calls','model_GPU_seconds',
                        'read_to_point_seconds_excluding_SHA_and_independent_validation','max_confidence_error')})
                    if v['repetition']==2:assert v['repeat_full_array_max_error']<=1e-5
    assert len(bench)==4 and len(checks)==98
    table(O/'single_scene_deployment.csv',bench)
    for name in ('TRAINING_SEAL.json','PAIRED_CROP_SCHEDULE.json','FIT_INPUT_LOCK.json','Catalogue_COMMON_SUPPORT_LOCK.json',
                 'Catalogue_CPU_ACCEPTANCE.json','Catalogue_CPU_RESOURCE_RECEIPT.json','CONTROLLER_STATUS.json','ALL_SCORES_SEALED.json'):
        shutil.copyfile(delivery/name,O/name)
    for arm in ('C','D'):
        for name in ('PARAMETER_WHITELIST.json','TRAINING_RECEIPT.json','INITIAL_GRADIENT_GATE.json'):
            shutil.copyfile(delivery/('train_'+arm)/name,O/(arm+'_'+name))
    save(O/'LOCAL_MODEL_TENSOR_ACCEPTANCE.json',dict(status='passed',models=tensor_receipts,Omni_model_calls=0))
    save(O/'MODEL_RECONSTRUCTION.json',dict(original_asset_lock='src/research_results/phase111_20261014/PRE_EXECUTION_LOCK.json',
        architecture='original author fastai DynamicUnet regnety_004, 3 inputs and 4 outputs',
        input_order=['B04','B03','B8A'],normalization='original per-band nonzero author channel_norm',
        decoder_overlays=tensor_receipts,allowed_state_keys='C_PARAMETER_WHITELIST.json / D_PARAMETER_WHITELIST.json',
        reconstruction='load original RegNet official safetensors with original author loader; load only decoder overlay strict=False; missing keys must be frozen encoder only',
        inference=dict(patch_size=1000,overlap=300,temperature=1,confidence='author softmax + float32(.001), clipped [0.001,0.999]',threshold=.5),
        model_storage='local outputs/phase121/phase121_20261015 and remote shared/phase121_20261015; no GitHub model weights'))
    save(O/'FINAL_CPU_ACCEPTANCE.json',dict(status='passed',two_fixed_fits=True,within_group_seed_averaging=True,
        support_first=True,old_five_baselines_reproduced=True,oracle_direct_boundary_acceptance=True,
        AP_ties_independent=True,bootstrap_scalar_independent=True,max_independent_confidence_error=max(checks),
        full_scene_receipts=len(checks),model_calls_in_finalizer=0))
    completion=dict(phase=121,status='completed_exploratory',calls=calls,elapsed_wall_seconds=controller['elapsed_seconds'],
        training_steps={'C12002':1000,'D12002':1000},new_downloads=0,failed_model_calls=0,
        CPU_peak_bytes=max(cpu['rss_peak_bytes'],max(v['rss_peak_bytes'] for v in cost)),
        GPU_reserved_peak_bytes=max(v['gpu_reserved_peak_bytes'] for v in cost),
        host_min_free_bytes=min(v['host_min_free_bytes'] for v in cost if v['host_min_free_bytes'] is not None),
        intermediate_peak_bytes=max(cpu['intermediate_peak_bytes'],max(v['intermediate_peak_bytes'] for v in cost)),
        decoder_files=tensor_receipts,models_local_and_shared_only=True,Git_model_weights_uploaded=False,
        work_dirs_transfers=0,original_failed_split_preserved=True,new_confirmation_claim=False,
        CPU_preflight_import_command_error_calls=0)
    save(O/'COMPLETION.json',completion)
    supports=[];worst=[]
    for pool in ('Catalogue','PixBox'):
        rr=rows(O/(pool+'_per_group.csv'));friendly=[]
        for row in rr:
            if row['E_AP']=='':continue
            tile=','.join(sorted({v.split('_T')[1][:5] for v in row['products'].split(',')}))
            friendly.append(dict(tile=tile,group_id=row['group_id'],positive=row['positive'],negative=row['negative'],
                **{p+'_AP_pp':float(row[p+'_AP'])*100 for p in POLICIES},
                **{k:float(v) for k,v in row.items() if '_AP_pp' in k and v!=''}))
        table(O/(pool+'_AP_groups_readable.csv'),friendly)
        for a,b in PAIRS:
            key=a+'_minus_'+b+'_AP_pp';apv=[(float(v[key]),v['group_id']) for v in rr if v[key]!='']
            rec=[((float(v[a+'_recall'])-float(v[b+'_recall']))*100,v['group_id']) for v in rr if v[a+'_recall']!='']
            worst.append(dict(pool=pool,comparison=a+'-'+b,worst_delta_AP_pp=min(apv)[0],worst_AP_group=min(apv)[1],
                worst_delta_recall_pp=min(rec)[0],worst_recall_group=min(rec)[1]))
        su=rows(O/(pool+'_summary.csv'))[0]
        supports.append(dict(pool=pool,all_groups=int(su['all_groups']),AP_groups=int(su['AP_groups']),positive=int(su['pooled_positive']),negative=int(su['pooled_negative']),
            support_loss=load(O/(pool+'_CPU_ACCEPTANCE.json'))['support_loss']))
    table(O/'worst_group_damage.csv',worst);save(O/'SUPPORT_SUMMARY.json',dict(pools=supports))
    text=['# Phase121 数据结果','','新seed12002的C（监督）与D（监督＋固定蒸馏）各1000步，均从原RegNet独立初始化；153 fit代表，两个末步封存后评价。旧seed12001完整复用。',
        '','## 1．数据结果','','固定confidence≥0.5；E为双模型教师，R为原RegNet，N为原ConvNeXt。',
        '', '| 数据池 | 模型 | 宏AP % | 池化AP % | 组均召回 % | 组均负类误报 % | 最大组误报 % |','|---|---|---:|---:|---:|---:|---:|']
    for pool in ('Catalogue','PixBox'):
        for v in rows(O/(pool+'_summary.csv')):
            text.append('| '+pool+' | '+v['setting']+' | '+' | '.join(f'{float(v[k]):.4f}' for k in ('macro_AP_pp','pooled_AP_pp','macro_recall_pp','macro_negative_FP_pp','worst_group_negative_FP_pp'))+' |')
    text+=['','| 数据池 | 比较 | 平均ΔAP pp | 95%描述区间 pp | 改善/稳定/下降 |','|---|---|---:|---|---|']
    for pool in ('Catalogue','PixBox'):
        for v in load(O/(pool+'_comparisons.json'))['rows']:
            ci=v['CI95_pp'];text.append(f"| {pool} | {v['comparison']} | {v['mean_delta_AP_pp']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {v['improved_gt1pp']}/{v['stable_within1pp']}/{v['declined_lt_minus1pp']} |")
    text+=['','双seed行先在相关组内平均两个seed的AP差，再对组等权；10000次配对组bootstrap，seed121015。不是两倍场景数，也不合成部署集成；区间条件于已观测两个seed。无正类AP=NA，仍参与误报；±1pp分改善/稳定/下降。',
        '', '### 同预算oracle诊断','','各池分别使用本池参考寻找唯一全局阈值，使相关组平均负类误报≤1%，取first unsafe negative tie之上的float64 nextafter边界。仅排序诊断，不是可部署校准收益，不作新冻结阈值。',
        '', '| 数据池 | 模型 | oracle宏召回 % | 实际组均误报 % | 最大组误报 % | 零召回正类组 |','|---|---|---:|---:|---:|---:|']
    for pool in ('Catalogue','PixBox'):
        for v in rows(O/(pool+'_oracle.csv')):
            text.append(f"| {pool} | {v['setting']} | {float(v['macro_recall_pp']):.4f} | {float(v['macro_negative_FP_pp']):.4f} | {float(v['max_group_negative_FP_pp']):.4f} | {v['zero_recall_positive_groups']} |")
    text+=['','全部逐组AP／运行指标、oracle逐组表及最差损伤另存CSV，未隐藏稀疏54TWN组。Catalogue32组仅7组可算AP，PixBox10组仅8组可算AP。PixBox是所选点参考，不能估计全景FPR。组均1%约束也不代表每组≤1%。',
        '', '### 成本和边界','',f'实际{calls}次模型调用；C/D各1000步，无额外拟合，无下载。墙钟{controller["elapsed_seconds"]:.1f}秒。CPU峰值{completion["CPU_peak_bytes"]/1e9:.3f}GB，GPU预留峰值{completion["GPU_reserved_peak_bytes"]/1e9:.3f}GB，宿主可用采样最低{completion["host_min_free_bytes"]/1e9:.3f}GB，中间文件采样峰值{completion["intermediate_peak_bytes"]/1e9:.3f}GB。',
        '单学生部署18,039,132参数，decoder训练14,135,988参数，教师只参与D训练；encoder与教师状态保持不变。首景单模型CUDA事件及带核验计时保留，非新速度基准。模型本地outputs及远端shared保留并核SHA，不上传GitHub权重，不传work_dirs。',
        'CPU预检曾有一次命令级导入路径错误，改用tools路径后全部哈希通过，未产生任何模型构建、前向或拟合；无模型重跑。',
        '', '## 2．结论探讨','','见 ANALYSIS.md；此处只列测量和方法边界。旧池已暴露，训练地点交集未知，标签质量与原支持门局限保留，不改写旧主检验失败。']
    with (O/'README.md').open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
    print(json.dumps(completion,indent=2))
if __name__=='__main__':main()

"""Finalize fixed Phase120 receipts and data tables; interpretation kept separate."""
import csv,json,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'src/research_results/phase120_20261015';U=R/'outputs/phase120/phase120_20261015'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def table(p,rs):
    with p.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def main():
    delivery=U/'delivery';controller=load(delivery/'CONTROLLER_STATUS.json')
    assert controller['status']=='completed_pending_local_PixBox_reference_join'
    for p in ('C','D'):
        t=load(delivery/('train_'+p)/'TRAINING_RECEIPT.json')
        assert t['steps']==1000 and t['encoder_unchanged'] and t['teachers_unchanged'] and t['decoder_updated']
        assert sha(U/(p+'_decoder_final.safetensors'))==t['final_model_sha256']
    for file in (delivery/'aggregate').iterdir():
        if file.suffix in ('.csv','.json'):shutil.copyfile(file,O/file.name)
    stages=['train_C','train_D','C_Catalogue','D_Catalogue','C_PixBox','D_PixBox','R_PixBox']
    records=[];all_calls=18
    for stage in stages:
        s=load(delivery/stage/'RUN_STATUS.json');calls=s['all_calls'];all_calls+=calls
        records.append({'stage':stage,'status':s['status'],'calls':calls,'construction_calls':s['construction_calls'],
            'training_calls':s.get('training_calls',0),'gate_calls':s.get('gate_calls',0),'main_inference_calls':s.get('main_calls',0),
            'warmup_calls':s.get('warmup_calls',0),'elapsed_seconds':s['elapsed_seconds'],
            'rss_peak_bytes':s['rss_peak_bytes'],'gpu_reserved_peak_bytes':s['gpu_reserved_peak_bytes'],
            'gpu_observed_peak_bytes':s.get('gpu_observed_peak_bytes'),'host_min_free_bytes':s.get('host_min_free_bytes'),
            'intermediate_peak_bytes':s.get('intermediate_peak_bytes')})
    failed=load(delivery/'train_D_attempt_001_failed/RUN_STATUS.json')
    assert failed['calls']==18 and failed['training_calls']==failed['gate_calls']==0 and failed['step']==0
    records.append({'stage':'train_D_attempt_001_failed','status':failed['status'],'calls':18,'construction_calls':18,
        'training_calls':0,'gate_calls':0,'main_inference_calls':0,'warmup_calls':0,'elapsed_seconds':failed['elapsed_seconds'],
        'rss_peak_bytes':failed['rss_peak_bytes'],'gpu_reserved_peak_bytes':failed['gpu_reserved_peak_bytes'],
        'gpu_observed_peak_bytes':None,'host_min_free_bytes':failed.get('host_min_free_bytes'),'intermediate_peak_bytes':failed.get('intermediate_peak_bytes')})
    assert all_calls==controller['calls_completed']==load(delivery/'ALL_SCORES_SEALED.json')['actual_calls']
    assert all_calls<=8000 and controller['elapsed_seconds']<=14400
    cpu=load(delivery/'Catalogue_CPU_REPLAY_RESOURCE.json');assert cpu['status']=='passed' and cpu['all_five_pool_and_group_metrics_reproduced'] and cpu['model_calls']==0
    records.append({'stage':'Catalogue_CPU_guarded_replay','status':'passed','calls':0,'construction_calls':0,'training_calls':0,'gate_calls':0,'main_inference_calls':0,'warmup_calls':0,'elapsed_seconds':cpu['elapsed_seconds'],'rss_peak_bytes':cpu['rss_peak_bytes'],'gpu_reserved_peak_bytes':0,'gpu_observed_peak_bytes':None,'host_min_free_bytes':cpu['host_free_after'],'intermediate_peak_bytes':None})
    table(O/'actual_cost.csv',records)
    bench=load(O/'PRE_EXECUTION_LOCK.json')['benchmark_used'];br=[];checks=[]
    for p in ('C','D','R'):
        info=load(delivery/(p+'_PixBox')/'MODEL_INFO.json')
        assert info['models']==1 and info['total_parameters']==18039132 and info['deployed_teacher_models']==0
        rs=load(delivery/(p+'_PixBox')/'SCENE_RECEIPTS.json')
        for r in rs:
            if r['product_id']==bench:br.append({k:r[k] for k in ['policy','product_id','repetition','model_calls','model_GPU_seconds',
                'read_to_point_seconds_excluding_SHA_and_independent_validation','excluded_inference_CPU_validation_seconds',
                'post_inference_CPU_validation_seconds','max_confidence_error','max_raw_mean_error']})
        for pool in (('Catalogue','PixBox') if p in ('C','D') else ('PixBox',)):
            for r in load(delivery/(p+'_'+pool)/'SCENE_RECEIPTS.json'):
                assert r['coverage_gradient_exact'] and r['max_confidence_error']<=1e-5 and r['max_raw_mean_error']<=1e-5
                if 'repeat_full_array_max_error' in r:assert r['repeat_full_array_max_error']<=1e-5
                checks.append(r['max_confidence_error'])
    assert len(br)==6;table(O/'single_scene_deployment.csv',br)
    # Fresh R benchmark must agree with sealed Phase119 R point confidences.
    old=load(R/'data/pixbox_s2_cmix_5036991/phase119_20261015/R/scores/PixBox'/bench/'UNLABELLED_POINT_SCORES.json')['points']
    new=load(R/'data/pixbox_s2_cmix_5036991/phase120_20261015/R_PixBox/scores/PixBox'/bench/'UNLABELLED_POINT_SCORES.json')['points']
    assert len(old)==len(new)
    difference=max(abs(a['shadow_confidence']-b['shadow_confidence']) for a,b in zip(old,new) if a['valid'] and b['valid'])
    assert difference<=1e-5
    acceptance={'status':'passed','failed_D_attempt_preserved':True,'failed_training_steps':0,'failed_construction_calls':18,
        'C_training_steps':1000,'D_training_steps':1000,'no_C_retraining':True,'no_checkpoint_or_threshold_selection':True,
        'fresh_R_benchmark_vs_Phase119_max_score_error':difference,'all_new_full_scene_receipts':len(checks),
        'max_independent_CPU_confidence_error':max(checks),'three_benchmark_repeat_pairs':True,
        'common_support_before_label_join':True,'both_CPU_AP_and_bootstrap_verified':True,'inference_temperature':1,
        'single_deployed_RegNet_graph_unchanged':True,'Omni_calls_in_this_finalizer':0,'actual_model_calls':all_calls}
    save(O/'FINAL_CPU_ACCEPTANCE.json',acceptance)
    for n in ['TRAINING_SEAL.json','PAIRED_CROP_SCHEDULE.json','FIT_INPUT_LOCK.json','Catalogue_COMMON_SUPPORT_LOCK.json','Catalogue_CPU_ACCEPTANCE.json','CONTROLLER_STATUS.json','ALL_SCORES_SEALED.json']:
        shutil.copyfile(delivery/n,O/n)
    for p in ('C','D'):
        for n in ['PARAMETER_WHITELIST.json','TRAINING_RECEIPT.json','INITIAL_GRADIENT_GATE.json']:
            shutil.copyfile(delivery/('train_'+p)/n,O/(p+'_'+n))
    completion={'phase':120,'status':'completed_exploratory','calls':all_calls,'elapsed_wall_seconds_including_engineering_pause':controller['elapsed_seconds'],
        'engineering_incidents':[{'reason':'JSON tuple/list graph equality','extra_construction_calls':18,'extra_training_steps':0},{'reason':'inference helper import missing','extra_calls':0,'extra_training_steps':0}],
        'guarded_CPU_replay_seconds':cpu['elapsed_seconds'],'first_CPU_pass_resource_peak_instrumented':False,
        'training_steps':{'C':1000,'D':1000},'failed_attempt_steps':0,'failed_construction_calls':18,'new_downloads':0,
        'CPU_peak_bytes':max(r['rss_peak_bytes'] for r in records),'GPU_reserved_peak_bytes':max(r['gpu_reserved_peak_bytes'] for r in records),
        'minimum_host_free_bytes':min(r['host_min_free_bytes'] for r in records if r['host_min_free_bytes'] is not None),
        'peak_intermediate_bytes':max(r['intermediate_peak_bytes'] or 0 for r in records),
        'decoder_files':{p:{'local_path':str(U/(p+'_decoder_final.safetensors')),'sha256':sha(U/(p+'_decoder_final.safetensors')),
            'bytes':(U/(p+'_decoder_final.safetensors')).stat().st_size} for p in ('C','D')},
        'work_dirs_transfers':0,'original_split_preserved':True,'new_confirmation_claim':False}
    save(O/'COMPLETION.json',completion)
    for p in ('C','D'):
        white=load(O/(p+'_PARAMETER_WHITELIST.json'))
        assert all(b['name'] in white['encoder_names'] for b in white['buffers'] if b['frozen'])
    friendly={};supports=[]
    for pool in ('Catalogue','PixBox'):
        per=rows(O/(pool+'_per_group.csv'));friendly[pool]=[]
        for r in per:
            if r['E_AP']=='':continue
            tile=','.join(sorted({v.split('_T')[1][:5] for v in r['products'].split(',')}))
            friendly[pool].append({'tile':tile,'group_id':r['group_id'],'positive':r['positive'],'negative':r['negative'],**{p+'_AP_pp':float(r[p+'_AP'])*100 for p in ('E','R','N','C','D')},**{k:float(r[k]) for k in ('D_minus_R_AP_pp','D_minus_C_AP_pp','C_minus_R_AP_pp')}})
        table(O/(pool+'_AP_groups_readable.csv'),friendly[pool])
        summary=rows(O/(pool+'_summary.csv'))[0]
        supports.append({'pool':pool,'all_groups':int(summary['all_groups']),'AP_groups':int(summary['AP_groups']),'positive':int(summary['pooled_positive']),'negative':int(summary['pooled_negative']),'effective_binary_pixels_or_points':int(summary['pooled_positive'])+int(summary['pooled_negative'])})
    save(O/'SUPPORT_SUMMARY.json',{'pools':supports,'Catalogue_support_loss':load(O/'Catalogue_CPU_ACCEPTANCE.json')['support_loss'],'PixBox_support_loss':load(O/'PixBox_CPU_ACCEPTANCE.json')['support_loss'],'PixBox_coordinate_support':9500,'PixBox_original_ignore_points':931})
    modeldir=O/'models';modeldir.mkdir(exist_ok=True)
    for p in ('C','D'):
        source=U/(p+'_decoder_final.safetensors');dest=modeldir/source.name
        shutil.copyfile(source,dest);assert sha(dest)==sha(source)
    shutil.copyfile(delivery/'Catalogue_CPU_REPLAY_RESOURCE.json',O/'Catalogue_CPU_REPLAY_RESOURCE.json')
    completion['combined_CPU_replay_and_live_inference_RSS_sampled_peak_bytes']=cpu['combined_CPU_replay_and_live_inference_RSS_sampled_peak_bytes']
    completion['local_necessary_models_pending_external_approval']={p:{'path':str((modeldir/(p+'_decoder_final.safetensors')).relative_to(R)).replace('\\','/'),'sha256':sha(modeldir/(p+'_decoder_final.safetensors'))} for p in ('C','D')}
    completion['Git_model_weights_uploaded']=False
    completion['model_external_delivery']='blocked_by_auto_review_pending_explicit_user_approval'
    save(O/'COMPLETION.json',completion)
    # Tables and raw comparisons are the data; editorial interpretation is supplied separately.
    text=['# Phase120 数据结果','',f'C／D各1000步，seed12001，原153 fit代表；固定末步后依次评价32 Catalogue development代表与16 PixBox产品／10相关组。仅训练原RegNet decoder，未新增输入或模块。',
          'E为原双模型教师；R为原RegNet；N为原ConvNeXt；C为纯监督；D为监督＋固定蒸馏。',
          '','## 1．数据结果','', '| 数据池 | 模型 | 宏AP % | 池化AP % | 组均召回 % | 组均负类误报 % | 最差组误报 % |', '|---|---|---:|---:|---:|---:|---:|']
    for pool in ('Catalogue','PixBox'):
        for r in rows(O/(pool+'_summary.csv')):
            text.append('| '+pool+' | '+r['setting']+' | '+' | '.join(f"{float(r[k]):.4f}" for k in ['macro_AP_pp','pooled_AP_pp','macro_recall_pp','macro_negative_FP_pp','worst_group_negative_FP_pp'])+' |')
    text += ['', '固定confidence≥0.5，正常推理温度1；负类包括Surface和Cloud。Catalogue为发布参考密集像元误报；PixBox仅为被选负参考点误报比例，不能解读为全景FPR。',
             '', '| 数据池 | 比较 | 平均ΔAP pp | 95%描述区间 pp | 改善/稳定/下降 |', '|---|---|---:|---|---|']
    damage=[]
    for pool in ('Catalogue','PixBox'):
        comparisons=load(O/(pool+'_comparisons.json'))['rows'];per=rows(O/(pool+'_per_group.csv'))
        for c in comparisons:
            ci=c['CI95_pp'];text.append(f"| {pool} | {c['comparison']} | {c['mean_delta_AP_pp']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {c['improved_gt1pp']}/{c['stable_within1pp']}/{c['declined_lt_minus1pp']} |")
        for a,b in [('D','R'),('D','C'),('C','R')]:
            apv=[(float(r[a+'_minus_'+b+'_AP_pp']),r['group_id']) for r in per if r[a+'_minus_'+b+'_AP_pp']!='']
            rec=[((float(r[a+'_recall'])-float(r[b+'_recall']))*100,r['group_id']) for r in per if r[a+'_recall']!='' and r[b+'_recall']!='']
            damage.append({'pool':pool,'comparison':a+'-'+b,'worst_AP_delta_pp':min(apv)[0],'worst_AP_group':min(apv)[1],
                'worst_recall_delta_pp':min(rec)[0],'worst_recall_group':min(rec)[1]})
    table(O/'worst_group_damage.csv',damage)
    text += ['', 'Catalogue为7个有参考云影相关组、全部32组参加误报评价；PixBox为8个有参考云影相关组、全部10组参加误报评价。无正类AP=NA；改善/稳定/下降沿用±1pp。10000次配对组bootstrap，seed120015，仅描述，不以区间跨零自动关闭方向。',
             '', '完整逐组差、逐组运行指标见 Catalogue_per_group.csv、PixBox_per_group.csv；最差AP与召回损伤见 worst_group_damage.csv。',
             '', '训练512裁剪／作者逐块归一化，评价1000/300完整滑窗，上下文差异保留。两个decoder各14,135,988可训练参数（总18,039,132的约78.4%），不称极少参数适配。可读逐组AP表另见 Catalogue_AP_groups_readable.csv 与 PixBox_AP_groups_readable.csv。',
             '', '## 成本与验收', '',f"实际模型调用{all_calls}，含D首次失败的18次构建；失败发生于JSON图签名比较，D当时0训练／0 gate前向，已保留记录，C完整复用。另一次评价入口导入失败产生0次模型调用，C/D均未重训。总墙钟（包括工程暂停）{controller['elapsed_seconds']:.1f}秒。训练损失、起末参数哈希、编码器／教师不变回执与参数白名单均已保存。",
             f"CPU进程峰值{completion['CPU_peak_bytes']/1e9:.3f}GB，GPU allocator预留峰值{completion['GPU_reserved_peak_bytes']/1e9:.3f}GB，宿主可用最小{completion['minimum_host_free_bytes']/1e9:.3f}GB，中间文件采样峰值{completion['peak_intermediate_bytes']/1e9:.3f}GB。",
             '首次Catalogue CPU指标通过未单独记录内存峰值；另对全部五模型及逐组/池化/区间做了受16GiB限制的完整缓存复算，结果复现，其峰值计入资源表，另记录并行推理＋CPU复算RSS采样峰值，合计也受16GiB约束；复算0模型调用。CPU-only复算辅助程序曾有一次入口语法错误（0模型调用），原日志保留；未改任何模型或指标定义。',
             '', '部署仅单RegNet，18,039,132参数，教师不加载。single_scene_deployment.csv保留同一冻结首景的R/C/D各两次完整推理计时／逐模型CUDA event；仅为结构验收，固定顺序与缓存状态未消除，不能替代Phase119完整速度基准。read-to-point计时扣除额外CPU核验，属于带仪器分段估计，非无仪器端到端保证。',
             f"所有完整推理的独立CPU confidence重建最大误差{max(checks):.9g}；新R基准景对原Phase119 R最大点分数误差{difference:.9g}，均≤1e-5。",
             '', '最终decoder权重留本地outputs和远端shared，两份各56.56MB的必要C/D decoder末步已在本地outputs和shared核SHA保存；向GitHub发送模型的动作被自动审批拦截，等待用户明确授权，Git暂不含模型权重，不含逐点参考、大像元缓存或work_dirs。复原时从PRE_EXECUTION_LOCK引用的原RegNet官方safetensors按原作者加载器构建，再仅load对应decoder文件（strict=False，缺项只能为冻结encoder）；PARAMETER_WHITELIST给出允许键。',
             '', '## 2．结论探讨', '', '见同目录 ANALYSIS.md；与上述原始数据分开。',
             '', '仍属已暴露池上的单seed探索；训练地点交集未知、参考语义局限与旧独立审查不足保留。不得改写旧主检验失败，也不构成新增独立确认。']
    (O/'README.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({'completion':completion,'acceptance':acceptance},indent=2))
if __name__=='__main__':main()

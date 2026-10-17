"""Documentation-only RegNet branch closeout; selected accepted sources, no re-audit."""
import csv,datetime,hashlib,json,re,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];A=R/'src/research_results/phase124_20261017'
O=R/'src/research_results/phase125_20261017';U=R/'outputs/phase125/phase125_20261017'
INPUTS={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    p=Path(p);assert p.suffix in ('.md','.json','.csv','.py'),p
    INPUTS[p.relative_to(R).as_posix()]=sha(p)
    return p.read_text(encoding='utf-8')
def load(p):return json.loads(read(p))
def rows(p):return list(csv.DictReader(read(p).splitlines()))
def write(name,text):
    with (O/name).open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')
def save(name,value):write(name,json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))
def link(p):return f'[{p.name}]({p.as_posix()})'
def mdtable(columns,data):
    return '\n'.join(['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+['| '+' | '.join(str(row.get(c,'')) for c in columns)+' |' for row in data])
SUMMARY='''# RegNet分支研究总结（Phase119—125）

## 一、数据与观察

本分支研究一个实际问题：已有双模型云影分割系统的计算开销能否降低，同时保留有用的识别能力。原系统E由RegNet与ConvNeXt两个成员组成，R是其中固定的原始单RegNet。研究先比较单成员与双模型，再在同一RegNet架构内进行监督适配与蒸馏，最后为监督适配模型留出固定校准组。目标是描述可复现的性能和成本取舍，而非预先承诺全面超过原系统。

Phase119固定评价原Catalogue development与PixBox。Phase120—121在153个fit代表上，以两个预定seed各训练监督模型C153及蒸馏模型D153，均封存1000步末步，不从评价结果挑选checkpoint。Phase122—123改用固定100训练组、53校准组，分别以seed12201和12301训练新C100。两次均从原始RegNet重新初始化，冻结encoder及其buffer，只训练原decoder；模型封存后才在53组全量有效支持上确定各自全局阈值，再冻结评价。旧C153与新C100不能混为同一模型，也不能把它们的差异单独归因于训练样本数。

校准采用逐组等权的负类误报约束，目标为校准组平均值不超过1%，使用完整有效支持、精确并列边界与float64比较。阈值没有在评价池重新选择，也未搜索其他预算。这里的留出只表示本轮训练未使用53组；这些成员和评价池的历史暴露仍在，不能因此获得全流程独立性。部署输入不包含人工标签比例或真值统计，参考只用于既定评价。

评价沿用已暴露的两个池，不增加确认集。Catalogue有32个相关组，其中7组可计算AP及召回；PixBox有16个产品、10个相关组，其中8组有正参考。AP和召回以有正参考的相关组为单位等权汇总，误报使用所有有有效负参考的组。无正类组不把AP填零，也不从误报分析中删除；两个seed不能使独立组数翻倍。

排序结果呈现明显的汇总尺度差异。C100两个seed相对R的Catalogue宏AP增量分别为−0.6860和−0.6520个百分点，先按组平均后为−0.6690；PixBox分别为−3.8983和−0.5556，平均−2.2270。两池宏AP均未整体超过R。与此同时，Catalogue的池化AP由R的73.5118%对应到两个C100的80.7688%和81.6099%；PixBox则由83.6151%对应到84.3069%和85.9706%。池化与等权逐组统计回答不同问题，必须同时保留，不能择优代替。

旧153组设置下，Phase121已有两seed汇总给出C153相对R的宏AP增量：Catalogue约+0.98、PixBox约+3.41个百分点。蒸馏D153相对监督C153的额外增量仅约+0.03和−1.32个百分点，描述区间都跨零。现有结果没有清楚分离出蒸馏的额外AP价值；这只限定当前配方与数据，不等于蒸馏在一般情形无效，也不能据此重新寻找胜出seed。

在冻结校准工作点，Catalogue两个C100相对固定R的宏召回差分别为+1.4719和+5.2600个百分点。先在组内平均后为+3.3659，已有95%描述区间为[−8.4370,+14.1707]；实际组均误报由R的1.2773%对应到两seed指标平均0.4367%。这记录了同向的工作点取舍，但召回不确定性仍大，实际误报也不同，不能称为相同FPR下的优越性。

PixBox相对R的召回差在两seed为−3.8360和+6.0717个百分点，方向发生变化；组内平均为+1.1178，区间[−9.5892,+12.5367]。实际所选负类点误报由0.5379%对应到两seed平均0.8474%，其误报代价在两seed中同向增加。1%是校准组内经验目标，不是评价池保证，更不是每景保证。这里平均的是各组指标，没有平均分数生成新的集成部署模型。

局部损伤同样构成主体证据。按严格大于正负1 AP百分点的描述边界，Catalogue的32TML、40XDR在两seed都下降，PixBox的19FDV、54TWN也重复下降；完整全组表同时保留改善与稳定组。35RPM排序改善却降低冻结工作点召回，44SKJ也表现为排序改善不保证召回改善。54TWN只有7个正参考，其幅度存在稀疏参考局限，不能剔除后报告均值，也不能由组名推定暗地表或某种物理机制。本分支没有新增机制验证。

损伤组没有被移出评价，缺失或无分母指标继续保持NA；文档归档也不会给尚未裁决的参考增加新的标签可靠性背书。

成本方面，Phase119按预先固定三景、各两次完整前向比较E和R，未只取最快一次。六次GPU模型段耗时合计E为41.5694秒、R为15.1217秒，R约为E的36.4%；读入至取点的整景分段估计分别为358.5138和334.1061秒，比例约93.2%。后者仍受插桩扣除、预取重叠、顺序、缓存及共享硬件影响，不能写成整景快一倍，也不能把剩余时间全部归因于磁盘。C100没有同口径配对的部署延迟与峰值测量，不能直接借用R的时间。

## 二、解释与适用边界

可支持的贡献是固定设置下单RegNet的计算成本证据，以及监督适配、全局校准所形成的性能取舍和跨池局限。C100部署仍是一个RegNet，但训练参数14,135,988个，占18,039,132总参数约78.4%，不属于微型适配器。已有56.56MB文件是decoder覆盖权重，重建还需基础模型，不能当作独立完整模型体积。实验调用总数及控制器墙钟包含训练、校准和验收，不能替代部署速度。

校准能改变召回与误报工作点，却不改变同一模型的AP，因而不能消除排序损伤。Catalogue较低误报的取舍值得保留，PixBox的seed波动和重复损伤也必须并列呈现。所有区间直接沿用既有相关组bootstrap，仅条件于两个固定seed，不是在估计跨seed总体稳定性。历史暴露、未知预训练地点交集、参考语义及稀疏样本限制保持；PixBox误报只针对选定负参考点，不能外推全景误报或三类mIoU。本次收口不改写原主检验失败，不追加独立确认标签，不承诺新机制、全面优势或投稿结果；它结束这轮既定分支的试验与整理，并不意味着相关研究问题已全部解决。
'''
def main():
    started=time.time();O.mkdir(parents=True,exist_ok=True);U.mkdir(parents=True,exist_ok=True)
    assert not (O/'CLOSEOUT.json').exists()
    pre=load(O/'PRE_EXECUTION_LOCK.json')
    for n,h in pre['source_sha256'].items():assert sha(R/n)==h,n
    selected=['README.md','CLAIMS_FOR_PHASE125.md','Catalogue_group_evidence.csv','PixBox_group_evidence.csv','Catalogue_performance.csv','PixBox_performance.csv',
        'DEPLOYMENT_COST_AND_SOURCES.csv','MISSING_AND_LIMITATIONS.csv','HISTORICAL_BACKGROUND_ONLY.csv','MODEL_MEMBER_IDENTITY.csv',
        'COST_METADATA_ACCEPTANCE.json','SUPPORT_AND_CHAIN_AUDIT.json','Catalogue_INTERVALS_REUSED.json','PixBox_INTERVALS_REUSED.json','PRE_EXECUTION_LOCK.json']
    manifest=load(A/'SHA256SUMS.json')['files'];material=[]
    for name in selected:
        p=A/name;assert sha(p)==manifest[name],name
        read(p);material.append(dict(role='accepted124 material',path=p.relative_to(R).as_posix(),sha256=sha(p)))
    oldindex=load(A/'INPUT_SHA_LOCK.json')['files']
    for pool in ('Catalogue','PixBox'):
        p=R/'src/research_results/phase121_20261015'/(pool+'_comparisons.json')
        assert sha(p)==oldindex[p.relative_to(R).as_posix()]['sha256']
        historical=load(p)['rows'];distill=next(v for v in historical if v['comparison']=='two_seed_D-C')
        assert round(distill['mean_delta_AP_pp'],2)==(.03 if pool=='Catalogue' else -1.32)
    write('BRANCH_SUMMARY.md',SUMMARY)
    tablelines=['# 结果表（仅复用已有精度与区间）','', 'R原始RegNet；C122=seed12201，C123=seed12301。百分点列保持原始输出精度，区间不重算。']
    for pool in ('Catalogue','PixBox'):
        rr=rows(A/(pool+'_performance.csv'))
        columns=['setting','threshold','macro_AP_pp','pooled_AP_pp','macro_recall_pp','macro_FPR_pp','FPR_U95_pp','max_group_FPR_pp','zero_recall_positive_groups','over_1pct_FP_groups']
        tablelines+=['','## '+pool+'：三个模型、六个工作点','',mdtable(columns,rr),'',
            'AP/召回有定义组数='+('7；FPR组数=32。' if pool=='Catalogue' else '8；FPR组数=10。PixBox为所选负参考点误报。'),
            '全部逐组证据：'+link(A/(pool+'_group_evidence.csv'))]
        reused=load(A/(pool+'_INTERVALS_REUSED.json'))
        assert reused['bootstrap_seed']==123015;values=reused['values'];principal=values['primary'];mean=values['two_seed_metric_mean_summary']
        tablelines+=['','两seed先在组内平均的指标汇总，不是分数集成；相对固定R校准工作点：','',
            mdtable(['metric','mean_delta_pp','CI95_pp','defined_groups'],[dict(metric=k,mean_delta_pp=principal[k]['mean'],CI95_pp=json.dumps(principal[k]['CI95']),defined_groups=principal[k]['n']) for k in ('AP','recall','FPR')]),
            '', '该指标汇总的宏AP='+str(mean['AP']['mean'])+'%；两seed池化AP的算术平均='+str(mean['pooled_AP_metric_mean_pp'])+'%（不是集成分数的池化AP）。']
    background=rows(A/'HISTORICAL_BACKGROUND_ONLY.csv')
    tablelines+=['','## 历史背景：153组适配与单成员结果','', '以下均为0.5工作点，不是与Tcal同误报预算的比较。旧C153/D153与新C100明确分开；E/R/N训练组数0仅指本分支未适配，原始预训练规模未知。','',
        mdtable(['pool','phase','setting','adapt_training_groups','operating_threshold','macro_AP_pp','pooled_AP_pp','macro_recall_pp','macro_FPR_pp'],background)]
    cost=rows(A/'DEPLOYMENT_COST_AND_SOURCES.csv')
    tablelines+=['','## 成本及口径','',mdtable(['policy','deployed_models','total_parameters','adapt_trainable_parameters','official_weights_bytes','decoder_overlay_bytes','required_reconstruction_assets_bytes','matched_GPU_seconds_six_runs','matched_read_to_point_seconds_six_runs','GPU_reserved_worker_peak_bytes'],cost),
        '', 'E/R仅引用119固定三景×两次；原始来源和条件见'+link(A/'DEPLOYMENT_COST_AND_SOURCES.csv')+'。C100未测栏原样保留，不移填R。基础权重+overlay是重建材料，不是合并后独立部署文件。',
        '122/123实验总调用2612/2392含训练、校准、验收及推理，不代表部署速度。逐阶段一次性成本见'+link(A/'ADAPTATION_CALIBRATION_COST.csv')+'。']
    write('RESULT_TABLES.md','\n'.join(tablelines))
    evidence=[dict(阶段=119,比较='E/R/N单成员与双模型',适配组数=0,seed='none',工作点='0.5',数据池='Catalogue/PixBox',来源='phase119 summary/benchmark/resource receipts'),
        dict(阶段=120,比较='C153监督/D153蒸馏',适配组数=153,seed=12001,工作点='0.5',数据池='Catalogue/PixBox',来源='phase120 endpoint receipts;124 historical background'),
        dict(阶段=121,比较='C153/D153第二seed与两seed汇总',适配组数=153,seed='12001/12002',工作点='0.5; historical oracle diagnostic not reused',数据池='Catalogue/PixBox',来源='phase121 comparisons;124 historical background'),
        dict(阶段=122,比较='C100监督及固定R校准',适配组数=100,seed=12201,工作点='0.5/own Tcal;53 calibration',数据池='Catalogue/PixBox',来源='phase122 locks/receipts;124 accepted tables'),
        dict(阶段=123,比较='C100第二seed/两seed指标均值',适配组数=100,seed='12201/12301',工作点='0.5/own frozen Tcal',数据池='Catalogue/PixBox',来源='phase123 receipts;124 intervals and evidence'),
        dict(阶段=124,比较='性能/工作点/成本整理与复核',适配组数='no new fit',seed='no new seed',工作点='unchanged',数据池='separate pools',来源='phase124 accepted documentation; no new experiment')]
    ilines=['# 证据索引','',mdtable(['阶段','比较','适配组数','seed','工作点','数据池','来源'],evidence),
        '', '以下只核验本轮实际引用的小材料，不重复124全量274条交付审计，也不重新读取模型、影像、参考或分数数组。已有模型SHA与位置沿用124验收记录。','', '## 本轮材料与SHA','',
        mdtable(['role','path','sha256'],material),'','## 权重身份（只引用，不复制）','']
    models=rows(A/'MODEL_MEMBER_IDENTITY.csv')
    for v in models:
        phase=v['phase'];arm=v['arm'];folder=next((R/'src/research_results').glob('phase'+phase+'_*')).name
        v['local_location']='outputs/phase'+phase+'/'+folder+'/'+arm+'_decoder_final.safetensors'
        v['controlled_shared_location']='/home/scv/shared/'+folder+'/'+arm+'_decoder_final.safetensors'
    ilines+=[mdtable(['phase','arm','adapt_training_groups','calibration_groups','model_sha256','local_location','controlled_shared_location'],models),
        '', '原R的已声明基础权重SHA：'+load(A/'COST_METADATA_ACCEPTANCE.json')['original_RegNet_declared_SHA']+'；本轮不把旧身份记录称为新的字节核验。E/R/N原始二进制复核范围沿用124缺项表。',
        '', '## 阈值、输入、代码与完整结果','']
    references=[('threshold and support chain',A/'SUPPORT_AND_CHAIN_AUDIT.json'),('training member identity',A/'MODEL_MEMBER_IDENTITY.csv'),('input and source SHA index',A/'INPUT_SHA_LOCK.json'),('124 code SHA',A/'PRE_EXECUTION_LOCK.json'),
        ('Catalogue original123 intervals',A/'Catalogue_INTERVALS_REUSED.json'),('PixBox original123 intervals',A/'PixBox_INTERVALS_REUSED.json'),('cost sources',A/'DEPLOYMENT_COST_AND_SOURCES.csv'),('unmeasured and unknown',A/'MISSING_AND_LIMITATIONS.csv')]
    for role,p in references:ilines.append('- '+role+': '+link(p)+'；SHA256 `'+sha(p)+'`。')
    ilines+=['', '123完整阈值/输入/代码/分数封存身份继续按124的INPUT_SHA_LOCK索引引用；无新阈值、模型或后处理版本。C100的精确阈值为R=0.7693686485290528、C122=0.36361625790596014、C123=0.2778868973255158，按float64比较。']
    write('EVIDENCE_INDEX.md','\n'.join(ilines))
    inherited=read(A/'CLAIMS_FOR_PHASE125.md');missing=rows(A/'MISSING_AND_LIMITATIONS.csv')
    write('CLAIMS_AND_LIMITATIONS.md',inherited+'\n\n## 未测/未知项保持原状态\n\n'+mdtable(['item','status','detail','impact'],missing)+'\n\n本分支收口不等于所有研究问题已解决。未进行新的独立确认；后续新方向从Phase126起，须按对应授权协议推进。')
    write('README.md','''# Phase125：RegNet分支已收口

Phase119—125本轮研究总结与小型文档归档完成。没有新增训练、拟合、前向、下载、像元读取、阈值搜索或区间重算，也未补seed或选胜出模型。

数据结果与解释在BRANCH_SUMMARY.md分开：C100两seed平均宏AP仍低于R；Catalogue保留较低误报的工作点取舍，PixBox召回方向有seed波动。R/E的已测GPU模型耗时比36.4%，整景分段比93.2%，不能互相替代；C100匹配部署计时未测。

- BRANCH_SUMMARY.md：可供论文/学位论文改写的研究总结。
- RESULT_TABLES.md：完整逐seed、两工作点、历史背景及成本表，保留原精度。
- EVIDENCE_INDEX.md：输入、代码、模型、阈值与结果的SHA和位置；权重不复制上传。
- CLAIMS_AND_LIMITATIONS.md：可支持表述、不能支持的结论和未测项。
- CLOSEOUT.json：完成时间、范围、文件清单及零新增实验声明。

既有逐组主体表仍保留在Phase124，不删除、不重新打包历史大文件。不自动追加实验；新方向从126起，本分支收口不干预其他并行方向。
''')
    # Document-consistency checks only, never inference or statistical recomputation.
    chars=len(re.findall(r'[\u4e00-\u9fff]',SUMMARY));assert 1500<=chars<=2500,chars
    expected=[('Catalogue',-.6690275232464111,3.3659292685626347),('PixBox',-2.226954755555976,1.1178279470602308)]
    for pool,ap,rec in expected:
        m=load(A/(pool+'_INTERVALS_REUSED.json'))['values']['primary']
        assert abs(m['AP']['mean']-ap)<1e-9 and abs(m['recall']['mean']-rec)<1e-9
    for p in O.glob('*.md'):
        for path in re.findall(r'\]\((D:/[^)]+)\)',p.read_text(encoding='utf-8')):assert Path(path).exists(),path
    checks=dict(status='passed',summary_Chinese_character_count=chars,range=[1500,2500],performance_rows=12,complete_own_threshold_and05=True,
        exact_existing_intervals_reused=True,links_exist=True,workpoint_and_cost_scopes_explicit=True,model_files_opened=0,bootstrap_recomputed=0)
    save('DOCUMENT_CONSISTENCY.json',checks);save('INPUT_SHA_LOCK.json',dict(selected_materials_only=True,files=INPUTS,no_full_delivery_reaudit=True))
    required=['README.md','BRANCH_SUMMARY.md','RESULT_TABLES.md','EVIDENCE_INDEX.md','CLAIMS_AND_LIMITATIONS.md','CLOSEOUT.json']
    from phase122_local_resources import peak_rss
    peak=peak_rss();assert peak<8*1024**3 and time.time()-started<3600
    save('CLOSEOUT.json',dict(phase=125,status='RegNet_branch_closed',covered_phases=list(range(119,126)),completed_time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        output_files=required,additional_receipts=['PRE_EXECUTION_LOCK.json','INPUT_SHA_LOCK.json','DOCUMENT_CONSISTENCY.json'],model_calls=0,training=0,fitting=0,
        downloads=0,pixels_read=0,model_binary_reads=0,threshold_searches=0,new_bootstrap=0,new_seed=0,full_historical_reaudit=False,unmeasured_items_retained=missing,
        elapsed_seconds=time.time()-started,CPU_peak_bytes=peak,subsequent_phase_started=False,new_direction_numbering_starts_at=126))
    for p in O.iterdir():
        if p.is_file():
            target=U/p.name
            assert not target.exists(),target
            target.write_bytes(p.read_bytes())
    print(json.dumps(dict(status='passed',Chinese_characters=chars,model_calls=0,seconds=time.time()-started)))
if __name__=='__main__':main()

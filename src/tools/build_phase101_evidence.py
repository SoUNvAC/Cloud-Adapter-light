"""Phase101: deterministic report extraction only; no metric estimation or model access."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'src/research_results/phase101_20261013'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def get(d, pointer):
    parts = pointer.split('/')
    while parts:
        if isinstance(d, list):
            d = d[int(parts.pop(0))]
        else:
            for width in range(len(parts), 0, -1):
                key = '/'.join(parts[:width])
                if key in d:
                    d = d[key]
                    parts = parts[width:]
                    break
            else:
                raise KeyError(pointer)
    return d


def compact(v):
    """Retain reported aggregates, intervals and decisions, omit large per-group arrays."""
    if isinstance(v, dict):
        return {k: compact(x) for k, x in v.items()
                if k not in ('pairs', 'rows', 'confusion', 'no_shadow_groups',
                             'zero_recall_group_ids', 'zero_recall_products',
                             'zero_recall_groups', 'tightest_scenes', 'macro_iou_percent',
                             'macro_iou', 'iou_percent')}
    if isinstance(v, list) and len(v) > 4:
        return {'omitted_list_length': len(v)}
    return v


def interval_fields(v, pointer=''):
    result = {}
    if isinstance(v, dict):
        for k, x in v.items():
            if k in ('ci95', 'descriptive_95_percentile', 'upper95_one_sided',
                     'lower95_one_sided', 'seed', 'resamples', 'n_groups', 'unit'):
                result[pointer + k] = x
            elif isinstance(x, dict):
                result.update(interval_fields(x, pointer + k + '/'))
    return result


def summary_fields(values, phase):
    result=[]
    def walk(v,path):
        if isinstance(v,dict):
            if 'mean' in v and isinstance(v['mean'],(int,float)):
                result.append(path+': '+format(v['mean'],'.8g')+
                              ('; 95%='+str(v.get('ci95',v.get('descriptive_95_percentile')))
                               if 'ci95' in v or 'descriptive_95_percentile' in v else ''))
            else:
                for k,x in v.items():
                    if k not in ('worst_AP','worst_ap','worst_recall','worst_damage','max_recall_decline'):
                        walk(x,path+'/'+k)
        elif isinstance(v,(int,float)) and not isinstance(v,bool):
            result.append(path+': '+format(v,'.8g'))
    for p,v in values.items():walk(v,p)
    return ' | '.join(result) or '布尔状态/历史判定，见原始摘录'


def make_specs():
    specs = []
    def add(n, comp, pointers, model, fit, cal, pool, groups, shadow, ref, decision,
            nature='探索性描述；区间不是新独立确认', limit='', unit='差值_pp；概率均值为0—1，mIoU_percent为百分数'):
        specs.append(dict(phase=str(n), comparison=comp, pointers=pointers, model=model,
                          fit=fit, calibration=cal, pool=pool, groups=groups,
                          shadow=shadow, reference=ref, decision=decision, nature=nature,
                          limitations=limit, units=unit))
    cat='Catalogue原参考；原20m有效像元支持；非人工逐像元裁决'
    dev='Catalogue development（已反复暴露）'
    in153='153 fit组参考非云影上冻结1%宏FPR阈值；非独立校准'
    cal53='固定100 fit/53 calibration；仅53组选择冻结阈值'
    alimit='ALCD非认证真值；主审身份/经验未核实，B未执行；54块恢复缓存无故障前SHA；9/6组不能合并为15组'
    add('65D','Source vs MsRE2000：宏AP与池化mIoU', ['points/0/macro_ap','points/1/macro_ap','points/0/miou_percent','points/1/miou_percent','points/0/ranking/ap','points/1/ranking/ap'],
        'Source RGB；MsRE seed65/step2000按dev mIoU选择',153,'argmax；非新增校准',dev,32,7,cat,
        '平均分割改善与逐组排序损伤并存',limit='选择集内描述；池化AP与宏AP方向不同；不能定位因果机制',unit='macro_ap/ranking/ap为0—1；miou_percent为百分数')
    add(66,'L3−L2（主要终点）',['comparisons/L3_minus_L2','policies/L3/fpr_constraint','primary_decision'],
        '冻结L3=Source+MsRE+B08/B11/B12；L2=Source+MsRE',153,in153,'Phase66原固定确认池',64,15,cat,
        'positive_increment_independently_supported；原H1失败不变',nature='原预设终点独立确认，限当时冻结候选/支持',limit='只有15个有云影组；不转移给100组模型或后续新候选')
    for control in ['L2_RGB','Source_spectral','MsRE_spectral']:
        add(67,'L3−'+control,['comparisons/L3_minus_'+control], '原153组冻结线性模型；'+control,153,in153,dev,32,7,cat,
            '描述性对照；跨零不等效',limit='相同dev重复探索；Source_spectral三类决策还使用Source非云影类别赢家',unit='ap均值/区间、pairs delta_ap为0—1；不是pp')
    for control in ['L3','RGB_local']:
        add(72,'局部光谱B−'+control,['comparisons/'+control,'decision','policies/B/fpr_strata/all'],
            '冻结局部B、L3、同维RGB局部对照',153,in153,'原65c固定确认池（Phase72已启封）',64,11,cat,
            'B−L3联合确认失败；光谱特异性前置门失败',nature='独立确认失败；RGB对比仅描述',limit='不得用RGB辅助区间替代失败主要终点；最大AP损伤−38.5628pp')
    add(85,'LoRA新seed66/67对MM的组内平均增量', ['new_seed_mean_AP_gain_pp','gate','retain_branch'],
        'LL/LM seed66/67；历史seed65不计晋级均值',153,in153,dev,32,7,cat,
        'retain_branch=false',limit='两个seed不把7组扩成14组；召回和池化mIoU条件失败；LM非单网络成本')
    for domain,total in [('original_main',9),('original_common',6),('revised_common',6)]:
        ref='ALCD原版9组；原生60m标签，20m概率3×3聚合' if total==9 else 'ALCD'+('原版' if domain=='original_common' else '修订版')+'6组共同有效支持；原生60m标签'
        add(89,'L3−L2',['versions/'+domain+'/primary_L3_minus_L2'], '继承153组冻结L2/L3',153,in153,'ALCD/'+domain,total,total,ref,
            '外部参考一致性；不是正式确认',limit=alimit)
        for control in ['L2_RGB','Source_spectral','MsRE_spectral']:
            add(90,'L3−'+control,['versions/'+domain+'/comparisons/L3_minus_'+control],
                '继承Phase67原153组模型',153,in153,'ALCD/'+domain,total,total,ref,
                '不选部署赢家；不声明MsRE普遍必要',limit=alimit)
    add(91,'L3−Source_spectral：1%平均FPR oracle召回',
        ['versions/original_main/comparisons_A_1percent/L3_minus_Source_spectral','oracle_only','no_new_deployment_threshold'],
        '原153组L3/Source_spectral',153,'本评价池标签扫描阈值：oracle，非部署','ALCD/original_main',9,9,
        'ALCD原版原生60m参考', '事后可达性诊断，不是校准收益',limit=alimit+'；无新增区间，A全池预算与B逐组预算不同')
    for n,controls in [(92,['Source_spectral_minus_S3','Source_spectral_minus_R6']),
                        (93,['G7_minus_G6'])]:
        for domain,total,sh in [('development',32,7),('original_main',9,9),('original_common',6,6),('revised_common',6,6)]:
            for control in controls:
                add(n,control,['domains/'+domain+'/comparisons/'+control],
                    '153组：Source_spectral/S3/R6固定线性' if n==92 else '153组：G7=RGB/NIR/SWIR+Source；G6=同六光谱HGB200',
                    153,in153,dev if domain=='development' else 'ALCD/'+domain,total,sh,
                    cat if domain=='development' else ('ALCD原版' if domain!='revised_common' else 'ALCD修订版')+'；原生60m参考，20m概率均值；'+('9组主支持' if total==9 else '6组共同支持'),
                    '条件性互补证据；无部署晋级',limit='固定拟合器下的输入增量，不识别上下文/物理/预训练因果机制；'+('dev重复探索' if domain=='development' else alimit))
    for model in ['G7_100','SourceS_100']:
        add(94,model+'：独立校准阈值−拟合内阈值',
            ['domains/development/comparisons/'+model+'_T_cal_minus_'+model+'_T_in',
             'domains/development/policies/'+model+'_T_cal'],
            '100 fit重新拟合，非原153组版本',100,cal53,dev,32,7,cat,
            '同一分数AP不变；冻结校准未保证dev预算',limit='不得将153→100的模型变化归因于校准；SourceS故障补拟合一次已授权、最终双模型')
    add(95,'G7_100：同支持合成60m阈值−20m阈值',
        ['domains/development_same60/comparisons/G7_100_T60_same_minus_T20_same',
         'domains/development_same60/policies/G7_100_T60_same/max_group_fpr'],
        '100组G7，复用同一分数',100,'53 calibration；同质3×3支持冻结T20_same/T60_same',
        'development_same60（已暴露）',32,7,'Catalogue合成60m同质格；不是ALCD原生60m',
        '支持/聚合改变工作点，不构成机制裁决',limit='不能把同质格筛选后的结果推广到全部原20m像元或ALCD')
    add(96,'T31TGF：G7_100尾部追踪链验收', ['no_manual_semantic_or_causal_claim','no_removal_based_pass','primary'],
        'G7_100；冻结T20_full',100,cal53,'单景事后诊断',1,0,cat,
        '不删场景判通过；不宣称已识别语义/因果机制',limit='该景无参考云影，AP/召回不适用；量化尾部数字见原逐组表，不扩成新统计证据',unit='布尔验收状态；本行不摘录新指标')
    add(97,'Source单调G7−G7_100',['paired_B_minus_A/intervals_pp','screening','decision'],
        '固定100 fit单调HGB vs G7_100',100,cal53,dev,32,7,cat,
        'close_this_fixed_monotonic_scheme',limit='平均FPR和最大组FPR均失败；非失效机制证明')
    add(98,'MIN−SourceS_100',['comparisons/MIN_minus_SourceS_100','policies/MIN/max_group_fpr','decision'],
        '100 fit双冻结模型，MIN分数融合',100,cal53,dev,32,7,cat,
        'close_both_fixed_fusion_branches',limit='排序和召回改善不能替代最坏组10%FPR门；AVG也关闭')
    add(99,'MIN−SourceS：O_tail可达召回',['primary','policies/development/MIN/O_tail','oracle_non_deployable','Phase98_branches_remain_closed'],
        '100 fit冻结分数',100,'用dev标签扫描O_tail；oracle非部署',dev,32,7,cat,
        '仅达到继续尾部校准研究的经验条件；未恢复Phase98分枝',limit='无bootstrap区间；报告修复仅验收行数，不重算分数或选择赢家')
    add(100,'MIN_T_tail−SourceS_T_tail',['comparisons/MIN_T_tail_minus_SourceS_T_tail','policies/MIN/T_tail','screening/MIN','decision'],
        '100 fit冻结G7/SourceS及MIN',100,'53 cal冻结平均≤1%且最大≤10%尾部阈值',dev,32,7,cat,
        'close_fixed_tail_calibration_candidate；end_global_threshold_repair_route',limit='MIN最大FPR12.2152%>10%，U95平均1.2931%；dev非独立确认')
    add(100,'G7_T_tail−SourceS_T_tail',['comparisons/G7_T_tail_minus_SourceS_T_tail','policies/G7_100/T_tail','screening/G7_100'],
        '100 fit冻结G7/SourceS',100,'53 cal冻结T_tail',dev,32,7,cat,
        '关闭：平均/尾部FPR均失败',limit='均值1.0846%>1%，最大29.0161%>10%；不能以召回区间替代约束')
    return specs


def collect():
    DEST.mkdir(parents=True, exist_ok=True)
    specs = make_specs()
    sources = {}
    for phase in dict.fromkeys(x['phase'] for x in specs):
        if phase=='65D':
            base=ROOT/'outputs/phase65/phase65d_followup_20261008'
            report=base/'delivery/development_report.json'
        else:
            candidates=list((ROOT/f'outputs/phase{phase}').rglob('tail_report.json' if phase=='96' else f'phase{phase}_report.json'))
            if phase=='89': candidates=[p for p in candidates if 'recovered_results' in str(p)]
            assert len(candidates)==1,(phase,candidates)
            report=candidates[0]
            base=report.parent.parent if report.parent.name=='delivery' else report.parent
        # A bounded allowlist of reports, existing receipts, locks, and per-group CSVs.
        # Never walk or copy work_dirs, imagery, probabilities, or model binaries.
        chosen={report}
        for directory in [base,base/'delivery']:
            if directory.is_dir():
                for p in directory.iterdir():
                    if p.is_file() and p.stat().st_size<3_000_000 and (
                        p.suffix=='.csv' or (p.suffix=='.json' and any(s in p.name.lower() for s in ['verif','accept','lock','sha256']))):
                        chosen.add(p)
        files=[]
        for p in sorted(chosen):
            rel=p.relative_to(base)
            copy=DEST/'sources'/phase/rel
            copy.parent.mkdir(parents=True,exist_ok=True)
            copy.write_bytes(p.read_bytes())
            assert sha(copy)==sha(p)
            files.append({'original_path':p.relative_to(ROOT).as_posix(),
                          'copy':copy.relative_to(DEST).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
        sources[phase]={'report':(DEST/'sources'/phase/report.relative_to(base)).relative_to(DEST).as_posix(),
                        'original_report':report.relative_to(ROOT).as_posix(),'files':files}
    (DEST/'source_files_sha256.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (DEST/'extraction_specs.json').write_text(json.dumps(specs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def render():
    sources=read(DEST/'source_files_sha256.json')
    specs=read(DEST/'extraction_specs.json')
    rows=[]; extracts=[]; source_checks=[]; historical_manifest_entries=0
    for phase,s in sources.items():
        for f in s['files']:
            assert sha(DEST/f['copy'])==f['sha256'],f['copy']
        available={Path(f['copy']).name:f['sha256'] for f in s['files']}
        def verify_existing_manifest(x):
            nonlocal historical_manifest_entries
            if isinstance(x,dict):
                for key,value in x.items():
                    name=Path(key).name
                    if isinstance(value,str) and len(value)==64 and name in available:
                        assert value==available[name],(phase,key)
                        historical_manifest_entries+=1
                    elif isinstance(value,dict):
                        verify_existing_manifest(value)
        for f in s['files']:
            if 'SHA256' in Path(f['copy']).name:
                verify_existing_manifest(read(DEST/f['copy']))
        report_hash=sha(DEST/s['report'])
        # Existing historical receipt, not a newly rerun CPU/scoring acceptance.
        matches=[]; receipts=[]
        for f in s['files']:
            name=Path(f['copy']).name.lower()
            if any(z in name for z in ['verif','accept','sha256']):
                text=(DEST/f['copy']).read_text(encoding='utf-8-sig')
                receipts.append(f['copy'])
                if report_hash in text:matches.append(f['copy'])
        source_checks.append({'phase':phase,'all_source_copies_hash_match':True,
            'report_sha256':report_hash,'historical_receipts':receipts,
            'historical_receipts_containing_current_report_sha256':matches,
            'cpu_acceptance_receipt_present':any('cpu_verif' in x.lower() for x in receipts),
            'warning':'当前报告与历史回执绑定' if matches else '历史回执存在，但报告SHA直接绑定未核实；不将本轮摘录校验称新CPU验收'})
    for i,s in enumerate(specs,1):
        source=sources[s['phase']]; d=read(DEST/source['report'])
        values={p:compact(get(d,p)) for p in s['pointers']}
        intervals={p:interval_fields(v) for p,v in values.items() if isinstance(v,dict)}
        intervals={p:v for p,v in intervals.items() if v}
        check=next(x for x in source_checks if x['phase']==s['phase'])
        row={'行号':i,'claim_id':f'E{i:02d}','阶段':'Phase'+s['phase'],
             '比较':s['comparison'],'模型版本':s['model'],'拟合组数':s['fit'],
             '校准来源':s['calibration'],'评价池':s['pool'],'总相关组数':s['groups'],
             '有云影组数':s['shadow'],'参考标签版本／支持':s['reference'],
             '主要指标及差值':summary_fields(values,s['phase']),
             '精确原始摘录':json.dumps(values,ensure_ascii=False,separators=(',',':')),
             '已有区间':json.dumps(intervals,ensure_ascii=False,separators=(',',':')) if intervals else '原报告未提供区间；本轮不重算',
             '原判定':s['decision'],'证据性质':s['nature'],
             '最终报告路径':source['original_report'],'关键限制':s['limitations'],
             '单位说明':s['units'],'来源JSON路径':' | '.join(s['pointers']),
             '验收边界':check['warning']+'；CPU历史回执'+('已收录' if check['cpu_acceptance_receipt_present'] else '未核实/仅交付回执')}
        rows.append(row)
        extracts.append({'claim_id':row['claim_id'],'source':source['report'],
                         'source_sha256':sha(DEST/source['report']),'exact_extracted_values':values})
    with (DEST/'evidence_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (DEST/'extracted_values.json').write_text(json.dumps(extracts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Reopen rendered CSV and compare exact extraction objects: no metric recomputation.
    with (DEST/'evidence_matrix.csv').open(encoding='utf-8-sig',newline='') as f: rendered=list(csv.DictReader(f))
    assert len(rendered)==len(extracts)
    for r,e in zip(rendered,extracts):
        assert json.loads(r['精确原始摘录'])==e['exact_extracted_values']
        assert int(r['有云影组数'])<=int(r['总相关组数'])
    # Verify the literal headline numbers; this is formatting existing values only.
    headline_specs=[('66','comparisons/L3_minus_L2/ap/mean','10.8610'),
                    ('66','comparisons/L3_minus_L2/ap/ci95/0','5.1980'),
                    ('66','comparisons/L3_minus_L2/ap/ci95/1','17.6094'),
                    ('93','domains/development/comparisons/G7_minus_G6/delta_ap_pp/mean','19.1678'),
                    ('93','domains/original_main/comparisons/G7_minus_G6/delta_ap_pp/mean','19.8269'),
                    ('100','comparisons/MIN_T_tail_minus_SourceS_T_tail/intervals_pp/recall/mean','20.2349')]
    claims=(DEST/'CLAIMS_AND_GAPS.md').read_text(encoding='utf-8')
    for phase,pointer,literal in headline_specs:
        assert format(get(read(DEST/sources[phase]['report']),pointer),'.4f')==literal
        assert literal in claims
    verification={'status':'passed','rows':len(rows),'source_phases':len(sources),
        'all_extracted_values_equal_final_reports':True,'all_source_copy_SHA256_match':True,
        'historical_manifest_entries_verified':historical_manifest_entries,
        'headline_numbers_verified':len(headline_specs),
        'source_checks':source_checks,'training':0,'fits':0,'calibrations':0,
        'network_forwards':0,'new_data_evaluations':0,'interval_recomputations':0,
        'literature_searches':0,'verification_scope':'file SHA and literal JSON/CSV extraction; no pixel/model/metric rerun',
        'unchanged_boundaries':['original H1 failure','closed local/LoRA/fusion/threshold branches','no work_dirs synchronization'],
        'unverified':['expert_A identity/experience','human_B not performed',
                      '54 recovered precrash probability blocks lack precrash SHA',
                      'older phases without standalone CPU receipt: no invented receipt']}
    (DEST/'VERIFICATION.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','rows':len(rows),'sources':len(sources),'source_bytes':sum(f['bytes'] for s in sources.values() for f in s['files'])}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--collect',action='store_true');args=parser.parse_args()
    if args.collect:collect()
    render()

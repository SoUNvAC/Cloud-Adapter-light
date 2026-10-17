"""CPU-only Phase119--123 evidence/cost audit; no models, pixels or bootstrap."""
import csv,hashlib,json,math,re,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[2]
O=R/'src/research_results/phase124_20261017'
U=R/'outputs/phase124/phase124_20261017'
PH={n:next((R/'src/research_results').glob(f'phase{n}_*')) for n in range(119,124)}
READ={}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def note(p):
    p=Path(p);READ[p.relative_to(R).as_posix()]=dict(sha256=sha(p),bytes=p.stat().st_size)
    return p
def load(p):return json.loads(note(p).read_text(encoding='utf-8'))
def rows(p):
    with note(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def save(name,data):
    with (O/name).open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def table(name,data):
    assert data,name
    fields=list(dict.fromkeys(k for row in data for k in row))
    with (O/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
def optional(v):return float(v) if v not in ('',None) else None
def difference(a,b):return (a-b)*100 if a is not None and b is not None else None
def direction(v):return 'NA' if v is None else 'increase' if v>0 else 'decrease' if v<0 else 'unchanged'
def ranking(v):return 'NA' if v is None else 'improved' if v>1 else 'declined' if v< -1 else 'stable'
def evidence(data,pool):
    states={(v['group_id'],v['setting']):v for v in data};result=[]
    for gid in sorted({v['group_id'] for v in data}):
        base=states[gid,'R_cal'];items=[]
        for arm in ('C122','C123'):
            cal,default=states[gid,arm+'_cal'],states[gid,arm+'_05']
            for key in ('positive','negative','products'):
                assert cal[key]==default[key]==base[key],(pool,gid,key)
            ap=difference(optional(cal['AP']),optional(base['AP']))
            recall=difference(optional(cal['recall']),optional(base['recall']))
            fp=difference(optional(cal['FPR']),optional(base['FPR']))
            q=dict(pool=pool,group_id=gid,products=base['products'],tiles='/'.join(sorted(set(re.findall(r'_T(\d{2}[A-Z]{3})',base['products'])))),
                seed=12201 if arm=='C122' else 12301,arm=arm,positive=int(cal['positive']),negative=int(cal['negative']),
                AP_delta_vs_R_pp=ap,AP_category=ranking(ap),cal_recall_delta_vs_R_pp=recall,cal_FPR_delta_vs_R_pp=fp,
                recall_direction_vs_R=direction(recall),FPR_direction_vs_R=direction(fp),
                own_cal_minus05_recall_pp=difference(optional(cal['recall']),optional(default['recall'])),
                own_cal_minus05_FPR_pp=difference(optional(cal['FPR']),optional(default['FPR'])),
                cal_zero_recall=(optional(cal['recall'])==0) if int(cal['positive']) else 'NA',
                default_zero_recall=(optional(default['recall'])==0) if int(cal['positive']) else 'NA',
                cal_recall=optional(cal['recall']),cal_FPR=optional(cal['FPR']),R_cal_recall=optional(base['recall']),R_cal_FPR=optional(base['FPR']),
                calibration_threshold=cal['threshold'],R_calibration_threshold=base['threshold'],
                ordering_gain_recall_loss=ap is not None and ap>0 and recall is not None and recall<0,
                ordering_loss_lower_FPR=ap is not None and ap<0 and fp is not None and fp<0,
                source='phase123/'+pool+'_per_group.csv')
            items.append(q)
        categories=[v['AP_category'] for v in items]
        repeat='both_improved' if categories==['improved']*2 else 'both_declined' if categories==['declined']*2 else 'NA_no_positive' if categories==['NA']*2 else 'remaining'
        for q in items:
            q['two_seed_AP_repeat_class']=repeat
            q['two_seed_recall_directions_agree']=items[0]['recall_direction_vs_R']==items[1]['recall_direction_vs_R']
            q['two_seed_FPR_directions_agree']=items[0]['FPR_direction_vs_R']==items[1]['FPR_direction_vs_R']
        result.extend(items)
    return result
def manifests():
    report=[];gaps=[]
    for n,folder in PH.items():
        candidates=([folder/'REPOSITORY_DELIVERY_SHA256.json'] if n==119 else [folder/'GIT_CONTENT_SHA256SUMS.json'] if n>=122 else [folder/'SHA256SUMS.json'])
        for path in candidates:
            manifest=load(path);entries=manifest.get('files',manifest)
            for key,value in entries.items():
                h=value.get('sha256') if isinstance(value,dict) else value
                if not isinstance(h,str) or len(h)!=64:continue
                p=R/key if key.startswith('src/') else folder/key
                status='missing';domain='unknown';actual=None
                if p.exists():
                    actual=sha(p);note(p)
                    if actual==h:status='verified';domain='raw_file'
                    else:
                        blob=subprocess.run(['git','show','HEAD:'+p.relative_to(R).as_posix()],cwd=R,capture_output=True)
                        if blob.returncode==0 and hashlib.sha256(blob.stdout).hexdigest()==h:status='verified';domain='Git_blob'
                report.append(dict(phase=n,manifest=path.relative_to(R).as_posix(),file=p.relative_to(R).as_posix(),expected_sha256=h,actual_raw_sha256=actual,status=status,verified_domain=domain))
                if status!='verified':gaps.append(dict(item='historical_manifest_'+str(n),status='unknown',detail=p.relative_to(R).as_posix()+': '+status,impact='do not rerun or silently repair historical artifacts'))
    return report,gaps
def main():
    start=time.time();assert not (O/'COMPLETION.json').exists();O.mkdir(parents=True,exist_ok=True);U.mkdir(parents=True,exist_ok=True)
    pre=load(O/'PRE_EXECUTION_LOCK.json')
    for p,h in pre['source_sha256'].items():assert sha(R/p)==h,p
    checks,gaps=manifests();table('HISTORICAL_SHA_AUDIT.csv',checks)
    fit={n:load(PH[n]/'FIT_INPUT_LOCK.json')['bindings'] for n in (120,121,122,123)}
    member=load(PH[122]/'MEMBER_INPUT_LOCK.json');cal=member['calibration']
    assert fit[120]==fit[121] and fit[122]==fit[123]==member['fit']
    assert len(fit[120])==153 and len(fit[122])==100 and len(cal)==53
    assert {v['group_id'] for v in fit[122]}.isdisjoint(v['group_id'] for v in cal)
    assert {v['group_id'] for v in fit[122]+cal}=={v['group_id'] for v in fit[120]}
    identities=[]
    for phase in (120,121,122,123):
        seal=load(PH[phase]/'TRAINING_SEAL.json')
        for arm,h in seal['models'].items():
            p=R/f'outputs/phase{phase}'/PH[phase].name/(arm+'_decoder_final.safetensors')
            state='unknown_missing_local_model'
            if p.exists():assert sha(p)==h;note(p);state='verified_local_bytes'
            else:gaps.append(dict(item='model_'+str(phase)+arm,status='unknown',detail=str(p.relative_to(R)),impact='model identity retained from prior seal, no reconstruction or re-download'))
            identities.append(dict(phase=phase,arm=arm,adapt_training_groups=len(fit[phase]),calibration_groups=53 if phase>=122 else 0,
                model_sha256=h,local_state=state,local_bytes=p.stat().st_size if p.exists() else 'unknown',original_pretraining_overlap='unknown'))
    table('MODEL_MEMBER_IDENTITY.csv',identities)
    threshold=load(PH[123]/'THRESHOLD_LOCK.json');oldthreshold=load(PH[122]/'THRESHOLD_LOCK.json')
    assert threshold['thresholds']['R']==oldthreshold['thresholds']['R'] and threshold['thresholds']['C122']==oldthreshold['thresholds']['C']
    arithmetic=[];repeat=[];support=[];history=[]
    for pool in ('Catalogue','PixBox'):
        summaries=rows(PH[123]/(pool+'_six_workpoints.csv'));per=rows(PH[123]/(pool+'_per_group.csv'));cmp=load(PH[123]/(pool+'_comparisons.json'))
        table(pool+'_performance.csv',[dict(**row,source_phase=123,source_file=pool+'_six_workpoints.csv',interval_bootstrap_seed=123015,
            interval_use='reused exactly; not recomputed',adapt_training_groups=0 if row['setting'].startswith('R_') else 100) for row in summaries])
        ev=evidence(per,pool);table(pool+'_group_evidence.csv',ev)
        first={v['group_id']:v for v in ev if v['arm']=='C122'}
        repeat.append(dict(pool=pool,all_groups=len(first),positive_groups=sum(v['positive']>0 for v in first.values()),
            both_improved_groups=[v['tiles'] for v in first.values() if v['two_seed_AP_repeat_class']=='both_improved'],
            both_declined_groups=[v['tiles'] for v in first.values() if v['two_seed_AP_repeat_class']=='both_declined'],
            remaining_positive_groups=[v['tiles'] for v in first.values() if v['two_seed_AP_repeat_class']=='remaining'],
            no_positive_groups=sum(v['positive']==0 for v in first.values()),boundary='strict +/-1 AP pp; descriptive'))
        for summary in summaries:
            rr=[v for v in per if v['setting']==summary['setting']]
            assert len(rr)==int(summary['all_groups'])
            for metric,column in [('AP','macro_AP_pp'),('recall','macro_recall_pp'),('FPR','macro_FPR_pp'),('Surface_FPR','macro_Surface_FPR_pp'),('Cloud_FPR','macro_Cloud_FPR_pp')]:
                values=[float(v[metric])*100 for v in rr if v[metric]!=''];computed=math.fsum(values)/len(values);expected=float(summary[column]);error=abs(computed-expected)
                assert error<1e-9,(pool,summary['setting'],metric,error)
                arithmetic.append(dict(pool=pool,setting=summary['setting'],metric=metric,groups=len(values),recomputed_mean_pp=computed,published_mean_pp=expected,absolute_error=error))
        states={(v['group_id'],v['setting']):v for v in per}
        for metric in ('AP','recall','precision','FPR','Surface_FPR','Cloud_FPR'):
            differences=[]
            for gid in first:
                vals=[optional(states[gid,k][metric]) for k in ('C122_cal','C123_cal','R_cal')]
                if all(x is not None for x in vals):a,b,c=vals;differences.append(((a+b)/2-c)*100)
            expected=cmp['primary'][metric];assert len(differences)==expected['n'];actual=math.fsum(differences)/len(differences)
            assert abs(actual-expected['mean'])<1e-9
            arithmetic.append(dict(pool=pool,setting='two_seed_metric_mean_minus_R',metric=metric,groups=len(differences),recomputed_mean_pp=actual,published_mean_pp=expected['mean'],absolute_error=abs(actual-expected['mean'])))
        # Copy original intervals verbatim. Never call a bootstrap/oracle implementation.
        save(pool+'_INTERVALS_REUSED.json',dict(source=(PH[123]/(pool+'_comparisons.json')).relative_to(R).as_posix(),source_sha256=sha(PH[123]/(pool+'_comparisons.json')),bootstrap_seed=123015,resamples=10000,values=cmp))
        for n in (119,120,121):
            for summary in rows(PH[n]/(pool+'_summary.csv')):
                setting=summary['setting'];include=(n==119 and setting in ('E','R','N')) or (n==120 and setting in ('C','D')) or (n==121 and setting in ('C12002','D12002'))
                if include:history.append(dict(pool=pool,phase=n,setting=setting,adapt_training_groups=153 if setting.startswith(('C','D')) else 0,
                    operating_threshold=.5,calibration_scheme='none; not budget matched to C100 Tcal',macro_AP_pp=summary['macro_AP_pp'],pooled_AP_pp=summary['pooled_AP_pp'],macro_recall_pp=summary['macro_recall_pp'],macro_FPR_pp=summary['macro_negative_FP_pp'],source_file=(PH[n]/(pool+'_summary.csv')).relative_to(R).as_posix()))
        for n in (119,120,121):
            old=rows(PH[n]/(pool+'_per_group.csv'))
            # 119--121 are wide (R_AP/R_recall columns); 123 is long by setting.
            baseline=[v for v in old if v.get('setting','R')=='R'];current=[v for v in per if v['setting']=='R_05']
            assert {v['group_id'] for v in baseline}=={v['group_id'] for v in current}
            b={v['group_id']:v for v in baseline}
            assert all(int(v['positive'])==int(b[v['group_id']]['positive']) and int(v['negative'])==int(b[v['group_id']]['negative']) for v in current)
        acceptance=load(PH[123]/(pool+'_CPU_ACCEPTANCE.json'));assert acceptance['support_loss']==0
        support.append(dict(pool=pool,**acceptance,historical119_121_R_group_denominators_match=True))
    table('HISTORICAL_BACKGROUND_ONLY.csv',history);table('INDEPENDENT_ARITHMETIC.csv',arithmetic);save('RANKING_REPEATABILITY.json',repeat);save('SUPPORT_AND_CHAIN_AUDIT.json',dict(rows=support,
        input_channels=['B04','B03','B8A'],normalization='accepted original author per-band nonzero channel_norm',patch=1000,overlap=300,temperature=1,
        inference_source_bindings=load(PH[123]/'PRE_EXECUTION_LOCK.json')['accepted_worker_transforms']['infer'],
        semantic_scope='identity/receipts only; no new pixel reads, postprocessing execution or model calls',thresholds=threshold['thresholds']))
    resources=rows(PH[119]/'model_resources.csv');benchmark=rows(PH[119]/'benchmark_all18.csv');ratios=load(PH[119]/'COST_RATIO_SUMMARY.json');cost=[]
    assert len(benchmark)==18
    for policy in ('E','R'):
        a=next(v for v in resources if v['policy']==policy);b=[v for v in benchmark if v['policy']==policy];assert len(b)==6 and len({v['product'] for v in b})==3
        gpu=math.fsum(float(v['GPU_model_seconds']) for v in b);whole=math.fsum(float(v['whole_scene_read_to_point_seconds']) for v in b)
        assert abs(gpu-ratios['matched_six_scene_runs_per_policy'][policy]['GPU_seconds'])<1e-9
        assert abs(whole-ratios['matched_six_scene_runs_per_policy'][policy]['whole_seconds'])<1e-9
        cost.append(dict(policy=policy,total_parameters=int(a['total_parameters']),adapt_trainable_parameters=0,deployed_models=int(a['models']),
            official_weights_bytes=int(a['weight_bytes']),decoder_overlay_bytes=0,required_reconstruction_assets_bytes=int(a['weight_bytes']),
            matched_GPU_seconds_six_runs=gpu,matched_read_to_point_seconds_six_runs=whole,GPU_reserved_worker_peak_bytes=int(a['GPU_reserved_peak_bytes']),
            source='Phase119 fixed3 scenes x2 repeats; entire isolated worker peak; measured sums, not fastest-run selection'))
    cost.append(dict(policy='C100',total_parameters=18039132,adapt_trainable_parameters=14135988,deployed_models=1,official_weights_bytes=72314968,
        decoder_overlay_bytes=56556112,required_reconstruction_assets_bytes=72314968+56556112,
        matched_GPU_seconds_six_runs='unmeasured',matched_read_to_point_seconds_six_runs='unmeasured',GPU_reserved_worker_peak_bytes='unmeasured_in_matched119_protocol',
        source='architecture/decoder whitelist from122/123; reconstruction storage is base+overlay, not merged standalone export; do not substitute R timing/peak'))
    table('DEPLOYMENT_COST_AND_SOURCES.csv',cost)
    onetime=[]
    for phase in (122,123):
        completion=load(PH[phase]/'COMPLETION.json')
        if phase==122:
            for stage in rows(PH[phase]/'actual_cost.csv'):onetime.append(dict(phase=phase,stage=stage['stage'],model_calls=int(stage['calls']),wall_seconds=stage['elapsed_seconds'],scope='stage includes load, checks, input and inference/training; not paired deployment timing'))
        else:
            for stage in load(PH[phase]/'ACTUAL_RESOURCE_COST.json')['stages']:onetime.append(dict(phase=phase,stage=stage['stage'],model_calls=stage['all_calls'],wall_seconds=stage['elapsed_seconds'],scope='stage includes load, checks, input and inference/training; not paired deployment timing'))
        assert sum(v['model_calls'] for v in onetime if v['phase']==phase)==completion['model_calls']
        calcheck=load(PH[phase]/'CALIBRATION_INDEPENDENT_CPU_ACCEPTANCE.json')
        onetime.append(dict(phase=phase,stage='independent_full53_boundary_CPU_check',model_calls=0,wall_seconds=calcheck['seconds'],scope='zero-forward CPU boundary proof, not neural calibration scoring cost'))
    table('ADAPTATION_CALIBRATION_COST.csv',onetime)
    save('COST_ARITHMETIC.json',dict(R_to_E_GPU_ratio=cost[1]['matched_GPU_seconds_six_runs']/cost[0]['matched_GPU_seconds_six_runs'],
        R_to_E_read_to_point_ratio=cost[1]['matched_read_to_point_seconds_six_runs']/cost[0]['matched_read_to_point_seconds_six_runs'],
        C100_trainable_fraction=14135988/18039132,matched_runs=6,scenes=3,original_intervals_reused=True,no_new_timing=True))
    gaps += [dict(item=k,status='unknown' if k in ('pretraining_geographic_overlap','original_Google_drive_tensor_equivalence') else 'unmeasured',detail=v,impact=w) for k,v,w in [
        ('C100_matched_latency_and_peak','no C100 x3scene x2repeat matched119 benchmark','cannot copy R latency/worker peak or claim C100 timing measured'),
        ('C100_merged_standalone_weight_size','existing56.56MB file is decoder overlay; base required','base+overlay reconstruction128.87MB is not merged deployment file size'),
        ('pretraining_geographic_overlap','unknown overlap with pretrained model data','no independent confirmation claim'),
        ('original_Google_drive_tensor_equivalence','historical model reconstruction metadata says not verified','HF identity/accepted receipts do not prove original Google Drive equivalence'),
        ('PixBox_full_scene_FPR_and_three_class_mIoU','only chosen negative reference point FPR','cannot extrapolate full-scene FPR or three-class mIoU'),
        ('seed_population_variation','only two fixed seeds; conditional related-group intervals reused','cannot claim cross-seed population stability'),
        ('training_group_count_causal_effect','C100 vs C153 changes data membership and seed','cannot identify data-size effect alone')]]
    table('MISSING_AND_LIMITATIONS.csv',gaps)
    save('INPUT_SHA_LOCK.json',dict(files=READ,model_pixels_read=False,reference_pixels_read=False))
    save('ARITHMETIC_ACCEPTANCE.json',dict(status='passed',checks=len(arithmetic),maximum_absolute_error=max(v['absolute_error'] for v in arithmetic),
        historical_manifest_entries=len(checks),historical_verified=sum(v['status']=='verified' for v in checks),historical_unknown=sum(v['status']!='verified' for v in checks),
        model_calls=0,new_bootstrap=0,threshold_search=0))
    from phase122_local_resources import peak_rss
    peak=peak_rss();assert peak<=8*1024**3 and time.time()-start<=3600
    save('COMPLETION.json',dict(phase=124,status='completed_CPU_evidence_cost_audit',model_calls=0,training=0,fitting=0,new_downloads=0,
        bootstrap_recomputations=0,threshold_searches=0,pixels_read=0,work_dirs_transfers=0,elapsed_seconds=time.time()-start,CPU_peak_bytes=peak,subsequent_phase_started=False))
    print(json.dumps(dict(status='passed',checks=len(arithmetic),historical_entries=len(checks),historical_unknown=sum(v['status']!='verified' for v in checks),model_calls=0,seconds=time.time()-start)))
if __name__=='__main__':main()

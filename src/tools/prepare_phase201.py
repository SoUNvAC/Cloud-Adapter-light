"""Freeze only existing historical tables/locks before Phase201 comparisons."""
import csv,datetime,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase201/phase201_20261017'
SOURCES={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(path,role):
    p=R/path;SOURCES[path]=dict(path=path,sha256=sha(p),bytes=p.stat().st_size,role=role,historical_exposure='all pools previously exposed; historical exploratory replay',identity_scope='table/metadata bytes; no new model or score-array verification')
    if p.suffix=='.csv':
        with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    if p.suffix=='.md':return p.read_text(encoding='utf-8-sig')
    return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def metric_rows(policy):
    out=[]
    for v in policy['rows']:
        cm=v['confusion'];pos=v['positive'];neg=v['negative']
        out.append(dict(group_id=v['group_id'],products=v.get('product'),positive=pos,negative=neg,AP=v['ranking']['ap'] if pos and neg else None,
            TP=v['true_positive'],FP=v['false_positive'],recall=v['recall'],FPR=v['fpr'],miou=v['miou_percent']/100,cm=cm,threshold=v.get('threshold')))
    pooled=policy.get('pooled',policy);rank=pooled.get('ranking',{})
    expected={}
    for k,target in [('macro_ap','macro_AP'),('macro_recall','macro_recall'),('macro_fpr','macro_FPR'),('macro_miou_percent','macro_miou')]:
        v=policy.get(k)
        if isinstance(v,dict):v=v.get('mean')
        if isinstance(v,(int,float)):expected[target]=v/(100 if k.endswith('percent') else 1)
    if 'macro' in policy:
        expected['macro_recall']=policy['macro']['recall']['mean'];expected['macro_FPR']=policy['macro']['fpr']['mean']
    if 'fpr_strata' in policy:expected['macro_FPR']=policy['fpr_strata']['all']['mean']
    return dict(rows=out,pooled_AP=rank.get('ap'),pooled_cm=pooled['confusion'],historical_expected=expected,
                miou_definition='nanmean over classes with reference+prediction union>0; absent in both excluded; pooled uses summed confusion',AP_definition='existing exact-ties noninterpolated AP; pooled AP reused from historical pooled ranking, never macro surrogate')
def tabular(model,summary,rows):
    q=[x for x in summary if x['setting']==model];assert len(q)==1
    q=q[0];rr=[]
    for x in rows:
        if x.get('setting')!=model:continue
        pos=int(x['positive']);neg=int(x['negative']);rr.append(dict(group_id=x['group_id'],products=x.get('products'),positive=pos,negative=neg,AP=float(x['AP']) if pos and neg else None,
            TP=int(x['TP']),FP=int(x['FP']),recall=float(x['recall']) if pos else None,FPR=float(x['FPR']) if neg else None,miou=None,cm=None,threshold=float(x['threshold'])))
    return dict(rows=rr,pooled_AP=float(q['pooled_AP_pp'])/100,pooled_cm=None,historical_expected={k:float(q[v])/100 for k,v in [('macro_AP','macro_AP_pp'),('macro_recall','macro_recall_pp'),('macro_FPR','macro_FPR_pp'),('max_FPR','max_group_FPR_pp')]},
        published_pooled_counts={k:int(q['pooled_'+v]) for k,v in [('positive','positive'),('negative','negative'),('TP','TP'),('FP','FP')]},miou_definition='not available: saved evaluation is binary Shadow only; no common three-class outputs',AP_definition='historical exact-ties AP; full pooled AP from accepted summary')
def main():
    O.mkdir(parents=True,exist_ok=True);assert not (O/'INPUT_CODE_LOCK.json').exists();blocks={};missing=[]
    read('src/research_plans/PHASE201_HISTORICAL_SELECTION_REPLAY_20261017.md','Phase201 fixed execution instructions')
    a=read('outputs/phase65/phase65d_recovery_20261008/delivery/recovery_report.json','A original32 development')
    ae=read('outputs/phase66/phase66_20261010/delivery/phase66_report.json','A original64 evaluation')
    al=read('outputs/phase65/phase65d_recovery_20261008/delivery/recovery_lock.json','A model parameters and frozen thresholds')
    read('outputs/phase66/phase66_20261010/phase66_lock.json','A identity/group/source/threshold chain')
    blocks['A']=dict(order=['L2','L3'],selection={k:metric_rows(a['policies'][v]) for k,v in [('L2','msre_source'),('L3','msre_source_spectral')]},evaluation={k:metric_rows(ae['policies'][k]) for k in ['L2','L3']},identities={k:al['models'][v] for k,v in [('L2','msre_source'),('L3','msre_source_spectral')]},expected_groups=[32,64],pool='Phase66 original64')
    b=read('outputs/phase71/phase71_20261010/delivery/phase71_report.json','B original32 development frozen baseline reproduction')
    be=read('outputs/phase72/phase72_20261010/delivery/phase72_report.json','B original65c64')
    bl=read('outputs/phase72/phase72_20261010/delivery/confirmation_lock.json','B exact frozen parameters/thresholds/support')
    read('outputs/phase72/phase72_20261010/delivery/development_reproduction.json','B historical development reproduction acceptance')
    blocks['B']=dict(order=['L3','B','RGB_local'],selection={k:metric_rows(b['policies'][k]) for k in ['L3','B','RGB_local']},evaluation={k:metric_rows(be['policies'][k]) for k in ['L3','B','RGB_local']},identities=bl['models'],expected_groups=[32,64],pool='Phase72 original65c64')
    c=dict(order=['R','C122','C123'],selection={},evaluation={},identities={},expected_groups=[32,10],pool='PixBox calibrated support')
    for phase,label,root in [(122,'C122','outputs/phase122/phase122_20261015'),(123,'C123','outputs/phase123/phase123_20261016/C123')]:
        lockpath=f'outputs/phase{phase}/phase{phase}_'+('20261015' if phase==122 else '20261016')+'/THRESHOLD_LOCK.json'
        c['identities'][label]=read(lockpath,'C frozen calibration thresholds and score/model bindings')
        for side,pool in [('selection','Catalogue'),('evaluation','PixBox')]:
            summ=read(root+'/'+pool+'_four_workpoints.csv','C accepted pooled AP and operating summary')
            rr=read(root+'/'+pool+'_per_group.csv','C per-group calibrated results')
            c[side][label]=tabular('C100_cal',summ,rr)
            r=tabular('R_cal',summ,rr)
            if 'R' in c[side]:assert c[side]['R']['rows']==r['rows'],'R changed across C seed reports'
            else:c[side]['R']=r
    for root in ['outputs/phase122/phase122_20261015','outputs/phase123/phase123_20261016']:
        for pool in ['Catalogue','PixBox']:
            p=root+'/'+pool+'_COMMON_SUPPORT_LOCK.json'
            if (R/p).exists():read(p,'C same-reference common support identity')
    read('outputs/phase125/phase125_20261017/CLOSEOUT.json','C model identity/exposure closeout')
    read('src/research_results/phase124_20261017/MODEL_MEMBER_IDENTITY.csv','C recorded decoder/model identity')
    c['identities']['R']='official V1 RegNet 044f2d6a3b4b59817f19d4ca080485f01c4bf9a6fddd053df184ee98144b46a2, threshold0.7693686485290528; source identity recorded, not reread weights'
    blocks['C']=c;missing.append(dict(block='C',side='selection/evaluation',item='pooled three-class mIoU selection rule and full-map three-class errors',reason='Existing accepted RegNet tables preserve only Shadow binary results; no common three-class confusion supplied. Rule not executed; no reconstruction/inference.'))
    d=dict(order=['E','R','N'],selection={},evaluation={},expected_groups=[32,10],pool='PixBox original119 support',identities={})
    for side,pool in [('selection','Catalogue'),('evaluation','PixBox')]:
        root='outputs/phase119/phase119_20261015';summ=read(root+'/'+pool+'_summary.csv','D fixed E/R/N pooled metrics');rr=read(root+'/'+pool+'_per_group.csv','D original119 per-group support')
        for name in d['order']:
            ss=next(x for x in summ if x['setting']==name);rows=[]
            for v in rr:
                pos=int(v['positive']);neg=int(v['negative']);rows.append(dict(group_id=v['group_id'],products=v['products'],positive=pos,negative=neg,AP=float(v[name+'_AP']) if pos and neg else None,
                    TP=int(v[name+'_TP']),FP=int(v[name+'_FP']),recall=float(v[name+'_recall']) if pos else None,FPR=float(v[name+'_negative_FP_fraction']) if neg else None,miou=None,cm=None,threshold=.5))
            d[side][name]=dict(rows=rows,pooled_AP=float(ss['pooled_AP_pp'])/100,pooled_cm=None,historical_expected={k:float(ss[v])/100 for k,v in [('macro_AP','macro_AP_pp'),('macro_recall','macro_recall_pp'),('macro_FPR','macro_negative_FP_pp'),('max_FPR','worst_group_negative_FP_pp')]},miou_definition='not applicable for point/binary member comparison',AP_definition='accepted original119 exact-ties AP')
    d['identities']['pre_execution']=read('outputs/phase119/phase119_20261015/PRE_EXECUTION_LOCK.json','D official members source/weight bindings')
    read('outputs/phase119/phase119_20261015/PixBox_COMMON_SUPPORT_LOCK.json','D original point support; kept separate from C')
    blocks['D']=d
    metroot='outputs/phasemet/phasemet_1_20261015/remote_backup/phasemet_1_20261015/'
    met=read(metroot+'GROUP_DETAILS.json','MET accepted group confusion-derived counts')
    rankings=read(metroot+'RANKINGS.json','MET historical accepted rankings')
    pooled={}
    for phase,names in [('81',['LL','LM','MM','ML']),('84',['LL','R4','F_real','F_shuffle'])]:
        for n in names:pooled[phase+'_'+n]=read(metroot+phase+'_'+n+'_pooled.json','MET accepted pooled confusion-derived counts')
    for n in ['INPUT_MANIFEST.json','CHECKS.json']:read(metroot+n,'MET original matrix hashes/dual implementation acceptance')
    supports={}
    for block,obj in blocks.items():
        supports[block]={}
        for i,side in enumerate(['selection','evaluation']):
            reference=None
            for name in obj['order']:
                rows=obj[side][name]['rows'];ids=[v['group_id'] for v in rows];assert len(ids)==len(set(ids))==obj['expected_groups'][i]
                sig=sorted([(v['group_id'],v['positive'],v['negative']) for v in rows])
                if reference is None:reference=sig
                else:assert sig==reference,(block,side,name)
            supports[block][side]=reference
    norm=dict(blocks=blocks,met=met,met_rankings=rankings,met_pooled=pooled,missing=missing,supports=supports)
    write(O/'NORMALIZED_INPUT.json',norm)
    code={str(p.relative_to(R)).replace('\\','/'):sha(p) for p in sorted((R/'src/tools').glob('*phase201*.py'))}
    write(O/'INPUT_CODE_LOCK.json',dict(phase=201,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),historical_not_blind=True,source_files=list(SOURCES.values()),code_sha256=code,normalized_sha256=sha(O/'NORMALIZED_INPUT.json'),group_support=supports,selection_rules=['pooled_miou','pooled_AP','macro_AP'],candidate_order={k:v['order'] for k,v in blocks.items()},dominance_tolerance=1e-10,mean_budget=.01,tail_budget=.1,bootstrap_resamples=10000,bootstrap_seed=201017,resource_cap=dict(seconds=7200,memory=16*1024**3,new_bytes=4*1024**3),no_push=True,no_model_calls=True))
    print('Frozen',len(SOURCES),'historical small inputs; all four block group supports match; comparisons not yet produced.')
if __name__=='__main__':main()

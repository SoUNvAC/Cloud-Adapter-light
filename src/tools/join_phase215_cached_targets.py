"""Phase215 stage2: join fixed source selections to existing target statistics."""
from pathlib import Path
import csv,hashlib,json,time,ctypes
from ctypes import wintypes
ROOT=Path(__file__).resolve().parents[2];P213=ROOT/'outputs/phase213/phase213_20261021';P214=ROOT/'outputs/phase214/phase214_20261023';OUT=ROOT/'outputs/phase215/phase215_20261023';START=time.monotonic()
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rd(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write(n,rs):
    with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def number(r,k):return float(r[k]) if r[k]!='' else None
def peak():
    class P(ctypes.Structure):_fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['peak','working','qpp','qp','qnp','qn','page','peakpage']]
    p=P();p.cb=ctypes.sizeof(p);k=ctypes.windll.kernel32;k.GetCurrentProcess.restype=wintypes.HANDLE;f=ctypes.windll.psapi.GetProcessMemoryInfo;f.argtypes=[wintypes.HANDLE,ctypes.POINTER(P),wintypes.DWORD];assert f(k.GetCurrentProcess(),ctypes.byref(p),p.cb);return p.peak
def main():
    lock=js(OUT/'SOURCE_SELECTION_LOCK.json');accept=js(OUT/'SOURCE_ONLY_ACCEPTANCE.json');assert accept['status']=='passed' and sha(OUT/'SOURCE_SELECTION_LOCK.json')==accept['source_lock_sha256']
    assert lock['target_metrics_loaded_by_selection_stage'] is False
    assert not (OUT/'RULE_COMPARISON.csv').exists()
    # No re-selection happens in this stage; all alpha values are from the durable source lock.
    target={(r['target'],float(r['alpha'])):r for r in rd(P213/'PARETO_TEST_TABLE.csv') if r['alpha']!=''}
    pools={(r['target'],r['pool'],float(r['alpha'])):r for r in rd(P213/'ALL_POOL_METRICS.csv')}
    source_ci={(r['target'],float(r['alpha'])):r for r in rd(P213/'PAIRED_DIFFERENCES_CI.csv') if r['pool']=='test' and float(r['reference_alpha'])==0}
    constraints=[];comparison=[];metrics=[];aux=[]
    for t,cs in lock['candidates'].items():
        for alpha,c in cs.items():
            a=float(alpha);r=target[t,a]
            constraints.append({'target':t,'alpha':a,'source_pool_loss_pp':c['pool_loss_pp'],'T11_pp':c['T11_pp'],'worst_ROI_loss_pp':c['worst_loss_pp'],'worst_ROI':c['worst_ROI'],'ROI_loss_gt_1pp':c['ROI_loss_gt_1pp'],'ROI_loss_gt_3pp':c['ROI_loss_gt_3pp'],'ROI_loss_gt_5pp':c['ROI_loss_gt_5pp'],'ROI_loss_gt_10pp':c['ROI_loss_gt_10pp'],'tail_feasible':c['T11_pp']<=5,'B1_feasible':c['feasible']['1'],'B3_feasible':c['feasible']['3'],'B5_feasible':c['feasible']['5'],'test_mIoU':number(r,'test_pooled_mIoU'),'test_gain_vs_Source_pp':number(r,'test_gain_vs_Source_pp')})
    strata={(r['support'],float(r['alpha'])):r for r in rd(P214/'L8_STRATIFIED_RESULTS.csv')}
    for s in lock['selections']:
        t,b,oa,na=s['target'],s['budget_pp'],s['old_alpha'],s['new_alpha'];oc=lock['candidates'][t][str(oa)];nc=lock['candidates'][t][str(na)];ot=target[t,oa];nt=target[t,na];ci=source_ci[t,na]
        comparison.append({'target':t,'budget_pp':b,'old_alpha':oa,'new_alpha':na,'fallback_Source':na==0,'same_working_point':oa==na,'old_source_pool_loss_pp':oc['pool_loss_pp'],'new_source_pool_loss_pp':nc['pool_loss_pp'],'old_T11_pp':oc['T11_pp'],'new_T11_pp':nc['T11_pp'],'old_worst_ROI_loss_pp':oc['worst_loss_pp'],'new_worst_ROI_loss_pp':nc['worst_loss_pp'],'old_ROI_loss_gt5_count':oc['ROI_loss_gt_5pp'],'new_ROI_loss_gt5_count':nc['ROI_loss_gt_5pp'],'old_test_mIoU':number(ot,'test_pooled_mIoU'),'new_test_mIoU':number(nt,'test_pooled_mIoU'),'new_gain_vs_Source_pp':number(nt,'test_gain_vs_Source_pp'),'forgone_test_gain_pp':number(ot,'test_pooled_mIoU')-number(nt,'test_pooled_mIoU'),'new_gain_CI95_low':number(ci,'CI95_low'),'new_gain_CI95_high':number(ci,'CI95_high'),'CI_source':'Phase213 same fixed alpha, conditional on fixed model/alpha; not selection-rule CI','new_minus_old_CI':'not calculated; point comparison only'})
        m={'target':t,'budget_pp':b,'old_alpha':oa,'new_alpha':na}
        for label,a in [('old',oa),('new',na)]:
            sr=pools[t,'source',a];tr=pools[t,'test',a]
            for k in ['mIoU','Surface_IoU','Cloud_IoU','Shadow_IoU','Shadow_recall','nonShadow_to_Shadow_FPR','group_mean_mIoU','max_group_FPR']:
                m[label+'_source_'+k]=number(sr,k);m[label+'_test_'+k]=number(tr,k)
            m[label+'_test_groups']=int(tr['groups']);m[label+'_test_no_shadow_groups']=int(tr['no_shadow_groups']);m[label+'_test_zero_shadow_recall_groups']=int(tr['zero_shadow_recall_groups'])
        metrics.append(m)
        if t=='l8':
            for support in ['yes','no']:
                r={'target':t,'budget_pp':b,'support':support,'old_alpha':oa,'new_alpha':na,'interpretation':'Phase214 posthoc support auxiliary only; never used to select alpha; no does not mean true-negative'}
                for label,a in [('old',oa),('new',na)]:
                    st=strata[support,a]
                    for k in ['mIoU','delta_mIoU_pp','Surface_IoU','Cloud_IoU','Shadow_IoU','Shadow_recall','nominal_FPR']:r[label+'_'+k]=number(st,k)
                aux.append(r)
    write('ALL_ALPHA_CONSTRAINTS.csv',constraints);write('RULE_COMPARISON.csv',comparison);write('RULE_METRICS_SIDE_BY_SIDE.csv',metrics);write('L8_SUPPORT_AUXILIARY.csv',aux)
    save('EXISTING_EVIDENCE_REFERENCES.json',{'Phase214_reused_not_recomputed':{n:{'path':str(P214/n),'sha256':sha(P214/n)} for n in ['SOURCE_WORST10_ROIS.csv','SOURCE_ROI_NET_CHANGES.csv','SOURCE_ERROR_CONCENTRATION.csv','SOURCE_DENOMINATOR_AUDIT.csv']},'Phase213_test_CI':{'path':str(P213/'PAIRED_DIFFERENCES_CI.csv'),'sha256':sha(P213/'PAIRED_DIFFERENCES_CI.csv'),'new_bootstrap_draws':0},'source_selection_lock_sha256_before_join':accept['source_lock_sha256'],'join_code_sha256':sha(Path(__file__)),'no_new_minus_old_CI':True})
    elapsed=time.monotonic()-START;memory=peak();assert elapsed<=1800 and memory<=4*1024**3
    save('TARGET_JOIN_COST.json',{'elapsed_seconds':elapsed,'native_peak_working_set_bytes':memory,'new_model_calls':0,'new_fits':0,'new_bootstrap_draws':0,'remote_connections':0,'source_lock_unchanged':sha(OUT/'SOURCE_SELECTION_LOCK.json')==accept['source_lock_sha256']})
    print(json.dumps(comparison,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

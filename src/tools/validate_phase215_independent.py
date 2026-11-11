"""Independent scalar IoU + heap/fmean source selection and delete-one audit."""
from pathlib import Path
import collections,csv,hashlib,heapq,json,math,statistics,time,ctypes
from ctypes import wintypes
ROOT=Path(__file__).resolve().parents[2];P213=ROOT/'outputs/phase213/phase213_20261021';P214=ROOT/'outputs/phase214/phase214_20261023';OUT=ROOT/'outputs/phase215/phase215_20261023';AS=[0,.25,.5,.75,1];START=time.monotonic();CHECKS=collections.Counter()
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rd(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert abs(float(a)-float(b))<1e-9,(a,b)
def scalar(c):
    ious=[]
    for k in range(3):
        u=sum(c[3*k+j] for j in range(3))+sum(c[3*j+k] for j in range(3))-c[4*k]
        if u:ious.append(100*c[4*k]/u)
    assert ious;return statistics.fmean(ious)
def add(cs):
    cs=list(cs);return tuple(sum(c[k] for c in cs) for k in range(9))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def choose(cs,b):return max(a for a,(pool,tail) in cs.items() if pool<=b and tail<=5)
def peak():
    class P(ctypes.Structure):_fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['peak','working','qpp','qp','qnp','qn','page','peakpage']]
    p=P();p.cb=ctypes.sizeof(p);k=ctypes.windll.kernel32;k.GetCurrentProcess.restype=wintypes.HANDLE;f=ctypes.windll.psapi.GetProcessMemoryInfo;f.argtypes=[wintypes.HANDLE,ctypes.POINTER(P),wintypes.DWORD];assert f(k.GetCurrentProcess(),ctypes.byref(p),p.cb);return p.peak
def main():
    registry=js(OUT/'INPUT_REGISTRY.json');lock=js(OUT/'SOURCE_SELECTION_LOCK.json');source_accept=js(OUT/'SOURCE_ONLY_ACCEPTANCE.json');assert sha(OUT/'SOURCE_SELECTION_LOCK.json')==source_accept['source_lock_sha256']
    assert lock['target_metrics_loaded_by_selection_stage'] is False and lock['posthoc_exploratory_rule'] and lock['humans_already_exposed_to_prior_target_results']
    for label,base in [('213',P213),('214',P214)]:
        for key,value in registry['files'].items():
            prefix,name=key.split('/',1)
            if prefix==label:assert sha(base/name)==value
    cases=collections.defaultdict(dict)
    for r in rd(P213/'PER_GROUP.csv'):
        raw=json.loads(r['confusion']);assert len(raw)==3 and all(len(x)==3 for x in raw) and all(type(v)is int and v>=0 for x in raw for v in x)
        c=tuple(v for x in raw for v in x);t,p,a,g=r['target'],r['pool'],float(r['alpha']),r['group'];assert g not in cases[t,p,a];cases[t,p,a][g]=c;close(scalar(c),r['mIoU']);CHECKS['independent_group_IoU']+=1
    full={};loss={};computed={}
    for t in ['l8','sparcs']:
        ids=sorted(cases[t,'source',0]);assert len(ids)==107
        computed[t]={}
        for a in AS:
            gs=cases[t,'source',a];assert sorted(gs)==ids
            for g,c in gs.items():
                b=cases[t,'source',0][g];assert [sum(c[i:i+3]) for i in [0,3,6]]==[sum(b[i:i+3]) for i in [0,3,6]]
                assert all(sum(c[3*k+j] for j in range(3))+sum(c[3*j+k] for j in range(3))-c[4*k]>0 for k in range(3))
            full[t,a]=add(gs.values());loss[t,a]={g:scalar(cases[t,'source',0][g])-scalar(c) for g,c in gs.items()}
            lp=scalar(add(cases[t,'source',0].values()))-scalar(full[t,a]);tail=statistics.fmean(heapq.nlargest(11,loss[t,a].values()));computed[t][a]=(lp,tail)
            c=next(v for k,v in lock['candidates'][t].items() if float(k)==a);close(c['pool_loss_pp'],lp);close(c['T11_pp'],tail);close(c['worst_loss_pp'],max(loss[t,a].values()))
            assert c['tail11_ids']==sorted(loss[t,a],key=lambda g:(-loss[t,a][g],g))[:11]
            for budget in [1,3,5]:assert c['feasible'][str(budget)]==(lp<=budget and tail<=5)
            CHECKS['independent_T11_and_candidate_feasibility']+=1
    tail214={(r['target'],float(r['alpha'])):r for r in rd(P214/'SOURCE_TAIL_SUMMARY.csv')}
    constraints={(r['target'],float(r['alpha'])):r for r in rd(OUT/'ALL_ALPHA_CONSTRAINTS.csv')};assert len(constraints)==10
    for (t,a),r in constraints.items():
        lp,ta=computed[t][a];close(r['source_pool_loss_pp'],lp);close(r['T11_pp'],ta);close(tail214[t,a]['worst11_mean_loss_pp'],ta)
        for k in [1,3,5,10]:assert int(r[f'ROI_loss_gt_{k}pp'])==sum(v>k for v in loss[t,a].values())
        for budget in [1,3,5]:assert (r[f'B{budget}_feasible']=='True')==(lp<=budget and ta<=5)
        close(r['test_mIoU'],scalar(add(cases[t,'test',a].values())))
    original=js(P213/'ALPHA_SELECTION_LOCK.json');selected={}
    comparison=rd(OUT/'RULE_COMPARISON.csv');assert len(comparison)==6
    for r in lock['selections']:
        t,b=r['target'],r['budget_pp'];new=choose(computed[t],b);old=max(a for a,(lp,tail) in computed[t].items() if lp<=b)
        assert new==r['new_alpha'] and old==r['old_alpha'];assert old==next(x['chosen_alpha'] for x in original['rows'] if x['target']==t and x['budget_pp']==b);selected[t,b]=new
        c=next(x for x in comparison if x['target']==t and int(x['budget_pp'])==b);assert float(c['new_alpha'])==new and float(c['old_alpha'])==old
        baseline=scalar(add(cases[t,'test',0].values()));oldtest=scalar(add(cases[t,'test',old].values()));newtest=scalar(add(cases[t,'test',new].values()))
        for key,val in [('new_test_mIoU',newtest),('old_test_mIoU',oldtest),('forgone_test_gain_pp',oldtest-newtest),('new_gain_vs_Source_pp',newtest-baseline)]:close(c[key],val)
        for prefix,a in [('old',old),('new',new)]:
            close(c[prefix+'_source_pool_loss_pp'],computed[t][a][0]);close(c[prefix+'_T11_pp'],computed[t][a][1]);close(c[prefix+'_worst_ROI_loss_pp'],max(loss[t,a].values()))
        assert (c['fallback_Source']=='True')==(new==0);assert (c['same_working_point']=='True')==(old==new);CHECKS['old_new_selection_and_target_cost']+=1
    # Recompute leave-one via full matrix subtraction; tail via heap, not primary sort/np.mean.
    independently_changed=collections.defaultdict(list);details=rd(OUT/'SOURCE_LEAVE_ONE_SELECTION_DETAILS.csv');assert len(details)==642
    for t in ['l8','sparcs']:
        ids=sorted(cases[t,'source',0])
        for g in ids:
            src=scalar(sub(full[t,0],cases[t,'source',0][g]));cs={}
            for a in AS:
                lp=src-scalar(sub(full[t,a],cases[t,'source',a][g]));tail=statistics.fmean(heapq.nlargest(11,(v for k,v in loss[t,a].items() if k!=g)));cs[a]=(lp,tail)
            for b in [1,3,5]:
                chosen=choose(cs,b);r=next(r for r in details if r['target']==t and int(r['budget_pp'])==b and r['left_out_ROI']==g);assert float(r['selected_alpha'])==chosen and int(r['remaining_ROIs'])==106 and int(r['tail_size'])==11
                close(r['chosen_pool_loss_pp'],cs[chosen][0]);close(r['chosen_T11_pp'],cs[chosen][1]);assert (r['differs_from_full']=='True')==(chosen!=selected[t,b]);CHECKS['leave_one_independent_selections']+=1
                if chosen!=selected[t,b]:independently_changed[t,b].append(g)
    for r in rd(OUT/'SOURCE_LEAVE_ONE_SELECTION.csv'):
        t,b=r['target'],int(r['budget_pp']);ds=[x for x in details if x['target']==t and int(x['budget_pp'])==b];assert len(ds)==107
        assert sum(int(r[f'alpha_{a:g}_count']) for a in AS)==107
        for a in AS:assert int(r[f'alpha_{a:g}_count'])==sum(float(x['selected_alpha'])==a for x in ds)
        assert int(r['changed_count'])==len(independently_changed[t,b]) and r['changed_ROI_ids']==';'.join(independently_changed[t,b]);CHECKS['LOO_count_total107_and_change_IDs']+=1
    changed=rd(OUT/'SOURCE_SELECTION_CHANGED_ROIS.csv');assert len(changed)==sum(len(v) for v in independently_changed.values())
    for r in changed:assert r['left_out_ROI'] in independently_changed[r['target'],int(r['budget_pp'])]
    # Side-by-side metrics and auxiliary strata are exact cached references, no selective omissions.
    oldpools={(r['target'],r['pool'],float(r['alpha'])):r for r in rd(P213/'ALL_POOL_METRICS.csv')}
    for r in rd(OUT/'RULE_METRICS_SIDE_BY_SIDE.csv'):
        for prefix in ['old','new']:
            a=float(r[prefix+'_alpha'])
            for p in ['source','test']:
                q=oldpools[r['target'],p,a]
                for k in ['mIoU','Surface_IoU','Cloud_IoU','Shadow_IoU','Shadow_recall','nonShadow_to_Shadow_FPR','group_mean_mIoU','max_group_FPR']:assert r[prefix+'_'+p+'_'+k]==q[k]
    strata={(r['support'],float(r['alpha'])):r for r in rd(P214/'L8_STRATIFIED_RESULTS.csv')}
    for r in rd(OUT/'L8_SUPPORT_AUXILIARY.csv'):
        for prefix in ['old','new']:
            q=strata[r['support'],float(r[prefix+'_alpha'])]
            for k in ['mIoU','delta_mIoU_pp','Surface_IoU','Cloud_IoU','Shadow_IoU','Shadow_recall','nominal_FPR']:assert r[prefix+'_'+k]==q[k]
    cis={(r['target'],float(r['alpha'])):r for r in rd(P213/'PAIRED_DIFFERENCES_CI.csv') if r['pool']=='test' and float(r['reference_alpha'])==0}
    for r in comparison:
        q=cis[r['target'],float(r['new_alpha'])];assert r['new_gain_CI95_low']==q['CI95_low'] and r['new_gain_CI95_high']==q['CI95_high']
    # Fixed-boundary and fallback checks: no epsilon, no clipping signed tail, no forced positive forgone gain.
    assert choose({0:(0,0),.25:(.5,5),.5:(1,math.nextafter(5,math.inf)),.75:(math.nextafter(1,math.inf),4)},1)==.25
    assert choose({0:(0,0),.25:(0,math.nextafter(5,math.inf))},5)==0
    assert choose({0:(0,0),.25:(-.1,-.2)},1)==.25
    assert statistics.fmean(heapq.nlargest(11,[-2.]*107))==-2.
    assert .5-1.0<0  # forgone gain is signed; never max(0, old-new)
    assert selected['l8',1]==selected['l8',3]==selected['l8',5]==0
    assert selected['sparcs',1]==selected['sparcs',3]==selected['sparcs',5]==.25
    memory=peak();elapsed=time.monotonic()-START;assert memory<4*1024**3 and elapsed<1800
    result={'status':'passed','independent_formula':'per-class integer row+column union; statistics.fmean; heapq.nlargest signed11 losses; LOO full-minus-one confusion','main_functions_imported':False,'checks':dict(CHECKS),'unrounded_boundary_no_epsilon_verified':True,'signed_tail_not_positive_clipped':True,'fallback_alpha0_and_repeated_working_points_verified':True,'signed_forgone_gain_definition_verified':True,'all_target_metrics_and_CI_exact_cached_references':True,'source_selection_lock_unchanged':sha(OUT/'SOURCE_SELECTION_LOCK.json')==source_accept['source_lock_sha256'],'elapsed_seconds':elapsed,'native_peak_working_set_bytes':memory,'new_model_calls':0,'remote_connections':0,'new_bootstrap_draws':0}
    (OUT/'INDEPENDENT_CPU_ACCEPTANCE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()

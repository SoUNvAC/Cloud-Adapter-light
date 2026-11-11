"""Phase215 stage1: source-only deterministic selection, before target table join."""
from pathlib import Path
import collections,csv,hashlib,json,math,time,ctypes
from ctypes import wintypes
import numpy as np
ROOT=Path(__file__).resolve().parents[2];P213=ROOT/'outputs/phase213/phase213_20261021';P214=ROOT/'outputs/phase214/phase214_20261023';OUT=ROOT/'outputs/phase215/phase215_20261023';AS=[0,.25,.5,.75,1];START=time.monotonic()
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rd(p):return csv.DictReader(p.open(encoding='utf-8-sig',newline=''))
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write(n,rs):
    with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def miou(c):
    c=np.asarray(c,dtype=np.int64);tp=np.diag(c);u=c.sum(0)+c.sum(1)-tp
    return float(100*np.nanmean(np.divide(tp,u,out=np.full(3,np.nan),where=u>0)))
def peak():
    class P(ctypes.Structure):_fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['peak','working','qpp','qp','qnp','qn','page','peakpage']]
    p=P();p.cb=ctypes.sizeof(p);k=ctypes.windll.kernel32;k.GetCurrentProcess.restype=wintypes.HANDLE;f=ctypes.windll.psapi.GetProcessMemoryInfo;f.argtypes=[wintypes.HANDLE,ctypes.POINTER(P),wintypes.DWORD];assert f(k.GetCurrentProcess(),ctypes.byref(p),p.cb);return p.peak
def main():
    OUT.mkdir(parents=True,exist_ok=False);sources={};source_inputs={}
    # SHA verification reads bytes, not target metric values or pixel arrays.
    for label,base in [('213',P213),('214',P214)]:
        done=js(base/'DELIVERY_COMPLETE.json');manifest=js(base/'DELIVERY_MANIFEST.json');assert sha(base/'DELIVERY_MANIFEST.json')==done['manifest_sha256']
        assert all((base/n).stat().st_size==r['bytes'] and sha(base/n)==r['sha256'] for n,r in manifest.items())
        sources[label]={'root':str(base),'manifest_sha256':done['manifest_sha256'],'original_artifacts_verified':len(manifest)}
    needed213=['PER_GROUP.csv','PER_GROUP_CHANGES.csv','ALPHA_SELECTION_TABLE.csv','ALPHA_SELECTION_LOCK.json','PARETO_TEST_TABLE.csv','PAIRED_DIFFERENCES_CI.csv']
    needed214=['SOURCE_TAIL_SUMMARY.csv','SOURCE_DENOMINATOR_AUDIT.csv','L8_STRATIFIED_RESULTS.csv','INPUT_HASHES.json','SOURCE_WORST10_ROIS.csv','SOURCE_ERROR_CONCENTRATION.csv','SOURCE_ROI_NET_CHANGES.csv']
    for label,base,names in [('213',P213,needed213),('214',P214,needed214)]:
        for n in names:assert (base/n).is_file();source_inputs[label+'/'+n]=sha(base/n)
    save('INPUT_REGISTRY.json',{'sources':sources,'files':source_inputs,'actual_filenames_registered':True,'protocol_sha256':sha(ROOT/'src/research_plans/PHASE215_SOURCE_MEAN_TAIL_SELECTION_20261023.md')})
    cases=collections.defaultdict(dict)
    for r in rd(P213/'PER_GROUP.csv'):
        if r['pool']!='source':continue  # never parse target confusion or metric values in this stage
        t,a,g=r['target'],float(r['alpha']),r['group'];raw=json.loads(r['confusion']);c=np.asarray(raw)
        assert c.shape==(3,3) and np.issubdtype(c.dtype,np.integer) and (c>=0).all() and g not in cases[t,a]
        cases[t,a][g]=c.astype(np.int64)
    assert len(cases)==10
    ids=sorted(cases['l8',0]);assert len(ids)==107
    for (t,a),gs in cases.items():
        assert sorted(gs)==ids
        for g,c in gs.items():
            assert np.array_equal(c.sum(1),cases[t,0][g].sum(1))
            u=c.sum(0)+c.sum(1)-np.diag(c);assert (u>0).sum()==3
    assert all(np.array_equal(cases['l8',0][g],cases['sparcs',0][g]) for g in ids)
    tail214={(r['target'],float(r['alpha'])):r for r in rd(P214/'SOURCE_TAIL_SUMMARY.csv')}
    denom214={(r['target'],float(r['alpha'])):r for r in rd(P214/'SOURCE_DENOMINATOR_AUDIT.csv')}
    def calculate(t,a,kept):
        base=cases[t,0];gs=cases[t,a];b=sum((base[g] for g in kept),np.zeros((3,3),np.int64));c=sum((gs[g] for g in kept),np.zeros((3,3),np.int64))
        losses={g:miou(base[g])-miou(gs[g]) for g in kept};ordered=sorted(losses,key=lambda g:(-losses[g],g));k=math.ceil(len(kept)*.1);assert k==11
        return {'source_mIoU':miou(c),'pool_loss_pp':miou(b)-miou(c),'T11_pp':float(np.mean([losses[g] for g in ordered[:k]])),'worst_ROI':ordered[0],'worst_loss_pp':losses[ordered[0]],'tail11_ids':ordered[:k],**{f'ROI_loss_gt_{b}pp':sum(x>b for x in losses.values()) for b in [1,3,5,10]}}
    candidates={t:{} for t in ['l8','sparcs']}
    for t in candidates:
        for a in AS:
            c=calculate(t,a,ids);old=tail214[t,a]
            for k,col in [('pool_loss_pp','source_pooled_loss_pp'),('T11_pp','worst11_mean_loss_pp'),('worst_loss_pp','worst_ROI_loss_pp')]:assert abs(c[k]-float(old[col]))<1e-10
            assert c['tail11_ids']==old['worst11_ROI_ids'].split(';') and int(denom214[t,a]['effective_class_count_changed'])==0
            c['feasible']={str(b):c['pool_loss_pp']<=b and c['T11_pp']<=5 for b in [1,3,5]};candidates[t][str(a)]=c
    original=js(P213/'ALPHA_SELECTION_LOCK.json');selections=[]
    for t in candidates:
        for b in [1,3,5]:
            mean_only=[a for a in AS if candidates[t][str(a)]['pool_loss_pp']<=b];joint=[a for a in AS if candidates[t][str(a)]['feasible'][str(b)]]
            old=next(r['chosen_alpha'] for r in original['rows'] if r['target']==t and r['budget_pp']==b);assert max(mean_only)==old and 0 in joint
            selections.append({'target':t,'budget_pp':b,'old_alpha':old,'new_alpha':max(joint),'mean_only_feasible':mean_only,'joint_feasible':joint,'fallback_to_Source':max(joint)==0})
    # Durable source-only lock is written before any target metrics are loaded.
    lock={'phase':'215','epoch':time.time(),'rule':'largest alpha with unrounded L_pool<=B AND signed T11<=5pp; no epsilon','budgets_pp':[1,3,5],'fixed_tail_budget_pp':5,'tail_size':11,'source_ROIs':107,'alphas':AS,'Source_source_mIoU':miou(sum(cases['l8',0].values(),np.zeros((3,3),np.int64))),'candidates':candidates,'selections':selections,'source_input_SHA':source_inputs,'original_selection_SHA':sha(P213/'ALPHA_SELECTION_LOCK.json'),'selection_code_sha256':sha(Path(__file__)),'target_metrics_loaded_by_selection_stage':False,'CSV_container_handling':'PER_GROUP includes source/test rows; source stage only parses confusion after pool==source; target rows skipped; global byte SHA is integrity verification only','posthoc_exploratory_rule':True,'humans_already_exposed_to_prior_target_results':True,'no_new_alpha_or_threshold_scan':True}
    save('SOURCE_SELECTION_LOCK.json',lock)
    loo=[];counts=[];changed=[]
    for t in candidates:
        for g in ids:
            kept=[x for x in ids if x!=g];cs={a:calculate(t,a,kept) for a in AS}
            for b in [1,3,5]:
                feasible=[a for a in AS if cs[a]['pool_loss_pp']<=b and cs[a]['T11_pp']<=5];assert feasible and 0 in feasible;chosen=max(feasible);full=next(r['new_alpha'] for r in selections if r['target']==t and r['budget_pp']==b)
                row={'target':t,'budget_pp':b,'left_out_ROI':g,'remaining_ROIs':106,'tail_size':11,'full_selection':full,'selected_alpha':chosen,'differs_from_full':chosen!=full,'feasible_alphas':','.join(str(a) for a in feasible),'chosen_pool_loss_pp':cs[chosen]['pool_loss_pp'],'chosen_T11_pp':cs[chosen]['T11_pp']};loo.append(row)
                if chosen!=full:changed.append(row)
        for b in [1,3,5]:
            rs=[r for r in loo if r['target']==t and r['budget_pp']==b];assert len(rs)==107
            counts.append({'target':t,'budget_pp':b,'full_selection':next(r['new_alpha'] for r in selections if r['target']==t and r['budget_pp']==b),**{f'alpha_{a:g}_count':sum(r['selected_alpha']==a for r in rs) for a in AS},'changed_count':sum(r['differs_from_full'] for r in rs),'changed_ROI_ids':';'.join(r['left_out_ROI'] for r in rs if r['differs_from_full'])})
    write('SOURCE_LEAVE_ONE_SELECTION.csv',counts);write('SOURCE_LEAVE_ONE_SELECTION_DETAILS.csv',loo)
    if changed:write('SOURCE_SELECTION_CHANGED_ROIS.csv',changed)
    else:
        (OUT/'SOURCE_SELECTION_CHANGED_ROIS.csv').write_text('target,budget_pp,left_out_ROI,selected_alpha\n',encoding='utf-8-sig')
    save('SOURCE_ONLY_ACCEPTANCE.json',{'status':'passed','source_candidates_reproduce_214':True,'old_mean_selections_reproduce_213':True,'all_ROI_reference_rows_exact_and_effective_classes3':True,'target_metrics_used_for_selection':False,'source_lock_sha256':sha(OUT/'SOURCE_SELECTION_LOCK.json'),'new_fits':0,'new_model_calls':0,'remote_connections':0})
    elapsed=time.monotonic()-START;memory=peak();assert elapsed<1800 and memory<4*1024**3
    save('SOURCE_STAGE_COST.json',{'elapsed_seconds':elapsed,'native_peak_working_set_bytes':memory,'model_calls':0,'remote_connections':0,'target_metric_join_done':False})
    print(json.dumps({'selections':selections,'leave_one_counts':counts,'elapsed_seconds':elapsed},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

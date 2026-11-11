"""Independent scalar, additive-count and resampling validation for Phase214."""
from pathlib import Path
import collections,csv,hashlib,json,math,time,ctypes
from ctypes import wintypes
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'outputs/phase213/phase213_20261021';OUT=ROOT/'outputs/phase214/phase214_20261023';START=time.monotonic()
NAMES=['Surface','Cloud','Shadow'];AS=[0,.25,.5,.75,1];checks=collections.Counter()
def rd(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def close(a,b):assert abs(float(a)-float(b))<1e-9,(a,b)
def matrix(r):
    c=json.loads(r['confusion']);assert len(c)==3 and all(len(v)==3 for v in c) and all(type(v) is int and v>=0 for row in c for v in row)
    return tuple(v for row in c for v in row)
def add(cs):
    cs=list(cs)
    return tuple(sum(c[k] for c in cs) for k in range(9))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def scalar(c):
    vals=[]
    for k in range(3):
        union=sum(c[k*3+j] for j in range(3))+sum(c[j*3+k] for j in range(3))-c[k*3+k]
        if union:vals.append(c[k*3+k]/union*100)
    assert vals
    return math.fsum(vals)/len(vals)
def error(c):return sum(c[k] for k in [1,2,3,5,6,7])
def quantile(v,q):
    s=sorted(float(x) for x in v);r=(len(s)-1)*q;i=math.floor(r);j=math.ceil(r)
    return s[i]+(r-i)*(s[j]-s[i])
def batch_scalar(flat):
    total=np.zeros(len(flat));nc=np.zeros(len(flat),dtype=np.int64)
    for k in range(3):
        den=sum(flat[:,k*3+j] for j in range(3))+sum(flat[:,j*3+k] for j in range(3))-flat[:,4*k]
        valid=den>0;total[valid]+=flat[valid,4*k]/den[valid]*100;nc+=valid
    assert (nc>0).all();return total/nc
def native_peak():
    class P(ctypes.Structure):_fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['peak','working','qpp','qp','qnp','qn','page','peakpage']]
    x=P();x.cb=ctypes.sizeof(x);f=ctypes.windll.psapi.GetProcessMemoryInfo;f.argtypes=[wintypes.HANDLE,ctypes.POINTER(P),wintypes.DWORD];k=ctypes.windll.kernel32;k.GetCurrentProcess.restype=wintypes.HANDLE;assert f(k.GetCurrentProcess(),ctypes.byref(x),x.cb);return int(x.peak)
def main():
    hashes=js(OUT/'INPUT_HASHES.json')
    for n,h in hashes['files'].items():assert hashlib.sha256((OLD/n).read_bytes()).hexdigest()==h
    adm=js(OLD/'delivery/FINAL_ADMISSION_LOCK.json');assert adm['mapping']=={'source':[0,1,1,2],'l8':[0,2,1,1],'sparcs':[2,2,0,0,0,1,0]}
    cases=collections.defaultdict(dict)
    for r in rd(OLD/'PER_GROUP.csv'):
        c=matrix(r);key=(r['target'],r['pool'],float(r['alpha']));assert r['group'] not in cases[key];cases[key][r['group']]=c;close(scalar(c),r['mIoU']);assert sum(c)==int(r['valid_pixels']);checks['scalar_group_mIoU']+=1
    for (t,p,a),gs in cases.items():
        assert len(gs)==(107 if p=='source' else 16 if t=='l8' else 10)
        assert set(gs)==set(cases[t,p,0])
        for g,c in gs.items():assert [sum(c[k:k+3]) for k in [0,3,6]]==[sum(cases[t,p,0][g][k:k+3]) for k in [0,3,6]]
    for r in rd(OLD/'ALL_POOL_METRICS.csv'):
        gs=cases[r['target'],r['pool'],float(r['alpha'])];close(scalar(add(gs.values())),r['mIoU']);close(math.fsum(scalar(v) for v in gs.values())/len(gs),r['group_mean_mIoU']);checks['original_pooled_reproduction']+=1
    meta=js(OLD/'TEST_LABEL_METADATA_AUDIT.json');s={x['scene']:x['usgs_shadows'] for x in meta['L8_scene_shadow_status']};supports={'full':sorted(s),'yes':sorted(g for g in s if s[g]=='yes'),'no':sorted(g for g in s if s[g]=='no')}
    assert [len(v) for v in supports.values()]==[16,6,10]
    for r in rd(OUT/'L8_STRATIFIED_RESULTS.csv'):
        a=float(r['alpha']);ids=supports[r['support']];c=add(cases['l8','test',a][g] for g in ids);b=add(cases['l8','test',0][g] for g in ids)
        close(scalar(c),r['mIoU']);close(scalar(c)-scalar(b),r['delta_mIoU_pp']);close(math.fsum(scalar(cases['l8','test',a][g]) for g in ids)/len(ids),r['group_mean_mIoU']);assert sum(c)==int(r['valid_pixels'])
        assert int(r['reference_Shadow_pixels'])==sum(c[6:9])
        if sum(c[6:9])==0:assert r['Shadow_recall']==''
        else:close(r['Shadow_recall'],c[8]/sum(c[6:9]))
        if sum(c[:6])==0:assert r['nominal_FPR']==''
        else:close(r['nominal_FPR'],(c[2]+c[5])/sum(c[:6]))
        unions=[sum(c[k*3+j] for j in range(3))+sum(c[j*3+k] for j in range(3))-c[k*3+k] for k in range(3)]
        assert int(r['effective_IoU_classes'])==sum(x>0 for x in unions)
        for k,name in enumerate(NAMES):
            if not unions[k]:assert r[name+'_IoU']==''
            else:close(r[name+'_IoU'],100*c[4*k]/unions[k])
        checks['stratified_pooled']+=1
    for a in AS:
        yes=add(cases['l8','test',a][g] for g in supports['yes']);no=add(cases['l8','test',a][g] for g in supports['no']);full=add(cases['l8','test',a].values());assert add([yes,no])==full
    for r in rd(OUT/'L8_ERROR_NET_CHANGES.csv'):
        a=float(r['alpha']);ids=supports[r['support']];c=add(cases['l8','test',a][g] for g in ids);b=add(cases['l8','test',0][g] for g in ids)
        for i in range(3):
            for j in range(3):
                if i!=j:assert int(r[NAMES[i]+'_to_'+NAMES[j]+'_net_count'])==c[3*i+j]-b[3*i+j]
        assert int(r['total_misclassified_net_count'])==error(c)-error(b);close(r['total_misclassified_net_pct_valid'],100*(error(c)-error(b))/sum(c));checks['error_flow_changes']+=1
    for r in rd(OUT/'L8_SCENE_RESULTS.csv'):
        c=cases['l8','test',float(r['alpha'])][r['scene']];b=cases['l8','test',0][r['scene']];close(scalar(c),r['mIoU']);close(scalar(c)-scalar(b),r['delta_mIoU_pp'])
        unions=[sum(c[3*k+j] for j in range(3))+sum(c[3*j+k] for j in range(3))-c[4*k] for k in range(3)];assert sum(v>0 for v in unions)==int(r['effective_IoU_classes'])
    calculated={};losses={}
    for t in ['l8','sparcs']:
        for a in AS:
            gs=cases[t,'source',a];base=cases[t,'source',0];losses[t,a]={g:scalar(base[g])-scalar(c) for g,c in gs.items()};calculated[t,a]=sorted(losses[t,a],key=lambda g:(-losses[t,a][g],g))
    for r in rd(OUT/'SOURCE_TAIL_SUMMARY.csv'):
        t,a=r['target'],float(r['alpha']);ls=losses[t,a];order=calculated[t,a];close(r['source_pooled_loss_pp'],scalar(add(cases[t,'source',0].values()))-scalar(add(cases[t,'source',a].values())));close(r['ROI_mean_loss_pp'],math.fsum(ls.values())/107);close(r['median_loss_pp'],quantile(ls.values(),.5));close(r['P90_loss_pp'],quantile(ls.values(),.9));close(r['worst11_mean_loss_pp'],math.fsum(ls[g] for g in order[:11])/11);assert r['worst11_ROI_ids'].split(';')==order[:11] and r['worst_ROI']==order[0];close(r['worst_ROI_loss_pp'],ls[order[0]])
        for k in [1,3,5,10]:assert int(r[f'ROIs_loss_gt_{k}pp'])==sum(v>k for v in ls.values())
        checks['source_tail_summary']+=1
    for name in ['SOURCE_ROI_NET_CHANGES.csv','SOURCE_WORST10_ROIS.csv']:
        for r in rd(OUT/name):
            t,a,g=r['target'],float(r['alpha']),r['ROI'];c=cases[t,'source',a][g];b=cases[t,'source',0][g];close(r['loss_pp'],scalar(b)-scalar(c));assert int(r['total_misclassified_net_count'])==error(c)-error(b);close(r['error_rate_delta_pp'],100*(error(c)-error(b))/sum(c));close(r['Source_mIoU'],scalar(b));close(r['current_mIoU'],scalar(c))
            unions={}
            for label,cc in [('Source',b),('current',c)]:
                unions[label]=[sum(cc[3*k+j] for j in range(3))+sum(cc[3*j+k] for j in range(3))-cc[4*k] for k in range(3)]
                assert int(r[label+'_effective_classes'])==sum(v>0 for v in unions[label])
                for k,name in enumerate(NAMES):
                    if not unions[label][k]:assert r[label+'_'+name+'_IoU']==''
                    else:close(r[label+'_'+name+'_IoU'],100*cc[4*k]/unions[label][k])
            for k,name in enumerate(NAMES):
                assert int(r['reference_'+name+'_pixels'])==sum(c[3*k:3*k+3]);close(r['reference_'+name+'_fraction'],sum(c[3*k:3*k+3])/sum(c))
            for i in range(3):
                for j in range(3):
                    if i!=j:
                        assert int(r[NAMES[i]+'_to_'+NAMES[j]+'_net_count'])==c[3*i+j]-b[3*i+j]
                        close(r[NAMES[i]+'_to_'+NAMES[j]+'_net_pct_valid'],100*(c[3*i+j]-b[3*i+j])/sum(c))
            if 'rank' in r:assert calculated[t,a][int(r['rank'])-1]==g
            checks['ROI_or_worst10_net_changes']+=1
    for r in rd(OUT/'SOURCE_ERROR_CONCENTRATION.csv'):
        t,a=r['target'],float(r['alpha']);v={g:error(c)-error(cases[t,'source',0][g]) for g,c in cases[t,'source',a].items()};pos=sorted([g for g in v if v[g]>0],key=lambda g:(-v[g],g));den=sum(v[g] for g in pos);assert int(r['positive_net_error_sum'])==den and int(r['negative_net_error_sum_signed'])==sum(x for x in v.values() if x<0)
        for k in [1,5,10]:
            if not den:assert r[f'top{k}_share']==''
            else:close(r[f'top{k}_share'],sum(v[g] for g in pos[:k])/den)
            assert r[f'top{k}_ROI_ids']==';'.join(pos[:k])
        checks['additive_error_concentration']+=1
    lock=js(OLD/'ALPHA_SELECTION_LOCK.json')
    for r in rd(OUT/'SOURCE_BUDGET_ROI_OVERLAP.csv'):
        budget=int(r['budget_pp']);sel={t:next(x['chosen_alpha'] for x in lock['rows'] if x['target']==t and x['budget_pp']==budget) for t in ['l8','sparcs']};sets={t:{g for g,l in losses[t,sel[t]].items() if l>5} for t in sel};assert r['intersection_ids']==';'.join(sorted(sets['l8']&sets['sparcs'])) and r['union_ids']==';'.join(sorted(sets['l8']|sets['sparcs']))
        for t in sel:
            m0=scalar(add(cases[t,'source',0].values()));feasible=[a for a in AS if m0-scalar(add(cases[t,'source',a].values()))<=budget];assert max(feasible)==sel[t]
        checks['budget_overlap_and_selection']+=1
    for r in rd(OUT/'SOURCE_DENOMINATOR_AUDIT.csv'):
        t,a=r['target'],float(r['alpha']);change=0;badchange=0
        for g,c in cases[t,'source',a].items():
            b=cases[t,'source',0][g]
            effective=lambda x:sum(sum(x[3*k+j] for j in range(3))+sum(x[3*j+k] for j in range(3))-x[4*k]>0 for k in range(3))
            diff=effective(c)!=effective(b);change+=diff;badchange+=diff and losses[t,a][g]>5
        assert int(r['effective_class_count_changed'])==change and int(r['loss_gt5_with_effective_count_change'])==badchange
        checks['effective_class_denominators']+=1
    for r in rd(OUT/'LEAVE_ONE_GROUP_OUT.csv'):
        t,p,a,g=r['target'],r['pool'],float(r['alpha']),r['left_out'];ids=supports[r['support']] if p=='test' else sorted(cases[t,p,a]);left=[x for x in ids if x!=g];d=scalar(add(cases[t,p,a][x] for x in left))-scalar(add(cases[t,p,0][x] for x in left));close(r['pooled_delta_mIoU_pp'],d);checks['leave_one_group_out']+=1
    los=rd(OUT/'LEAVE_ONE_GROUP_OUT.csv')
    for r in rd(OUT/'LEAVE_ONE_GROUP_OUT_RANGES.csv'):
        vals=[float(x['pooled_delta_mIoU_pp']) for x in los if all(x[k]==r[k] for k in ['target','pool','support','alpha'])];assert vals;close(min(vals),r['LOO_min_delta_pp']);close(max(vals),r['LOO_max_delta_pp'])
    # New target strata only, regenerate paired indices and use an independent einsum/classwise formula.
    rng=np.random.default_rng(214023);draws=collections.defaultdict(list)
    for r in rd(OUT/'L8_BOOTSTRAP_DRAWS.csv'):draws[r['support'],float(r['alpha'])].append((int(r['draw']),float(r['delta_mIoU_pp'])))
    bootrule=js(OUT/'BOOTSTRAP_RULE.json');cirows=rd(OUT/'L8_STRATIFIED_DIFFERENCES_CI.csv')
    for support in ['yes','no']:
        ids=supports[support];n=len(ids);idx=rng.integers(0,n,size=(10000,n));assert hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest()==bootrule['supports'][support]['index_sha256']
        h=np.zeros((10000,n),np.int64)
        for j,ix in enumerate(idx):
            for k in ix:h[j,k]+=1
        base=np.array([cases['l8','test',0][g] for g in ids]);bm=batch_scalar(np.einsum('bg,gi->bi',h,base))
        for a in [.25,.5,.75]:
            c=np.array([cases['l8','test',a][g] for g in ids]);d=batch_scalar(np.einsum('bg,gi->bi',h,c))-bm;observed=sorted(draws[support,a]);assert [x[0] for x in observed]==list(range(10000));assert max(abs(x-v) for (_,x),v in zip(observed,d))<1e-9
            ci=next(r for r in cirows if r['support']==support and float(r['alpha'])==a);close(ci['CI95_low'],quantile(d,.025));close(ci['CI95_high'],quantile(d,.975));checks['bootstrap_all10000_and_manual_quantiles']+=1
    copied=rd(OUT/'SOURCE_EXISTING_213_CI.csv');old=[r for r in rd(OLD/'PAIRED_DIFFERENCES_CI.csv') if r['pool']=='source' and float(r['reference_alpha'])==0];assert copied==old
    # Explicit boundary cases for absent classes and zero denominators.
    assert scalar((10,0,0,0,0,0,0,0,0))==100 and scalar((10,0,1,0,0,0,0,0,0))==50*10/11
    elapsed=time.monotonic()-START;peak=native_peak();assert elapsed<1800 and peak<=4*1024**3
    result={'status':'passed','independent_scalar_classwise_formula':True,'main_script_functions_imported':False,'checks':dict(checks),'absent_class_and_zero_positive_cases_verified':True,'source_CIs_exact_copy_not_resampled':True,'phase213_input_SHA_unchanged':True,'elapsed_seconds':elapsed,'native_peak_working_set_bytes':peak,'new_model_calls':0,'new_fits':0,'remote_connections':0}
    (OUT/'INDEPENDENT_CPU_ACCEPTANCE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()

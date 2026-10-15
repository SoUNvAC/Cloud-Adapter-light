"""Support-first local joins; original references never uploaded to remote."""
import collections,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
from phase122_metrics import evaluate
R=Path(__file__).resolve().parents[2];U=R/'outputs/phase122/phase122_20261015';O=R/'src/research_results/phase122_20261015'
D=R/'data/sentinel2_cloud_mask_catalogue_4172871/phase122_20261015';P=R/'data/pixbox_s2_cmix_5036991/phase122_20261015'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def unpack():
    rec=load(U/'DELIVERY_RECEIPT.json');path=U/'phase122_delivery.zip'
    assert sha(path)==rec['sha256'] and path.stat().st_size==rec['bytes']
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read('DELIVERY_MANIFEST.json'))['files'];assert len(z.namelist())==len(manifest)+1
        for name,h in manifest.items():
            assert not name.startswith('/') and ':' not in name and '\\' not in name and '..' not in Path(name).parts
            raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==h
            dest=((P if '/PixBox/' in name else D)/name) if '/scores/' in name else U/'delivery'/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(raw)
    assert sha(U/'C_decoder_final.safetensors')==rec['decoder']['sha256']
def score(stage,pool,pid):
    base=D/stage;f=base/'scores'/pool/(pid+'.npz')
    seals=load(U/'delivery'/stage/'SCORES_SEALED.json')['scores_sha256'];assert sha(f)==seals[pool+'/'+pid]
    with np.load(f,allow_pickle=False) as z:return {k:z[k].copy() for k in ('scores','valid','flat_indices')}
def mask_truth(group,indices):
    with zipfile.ZipFile(R/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip') as z:raw=z.read('masks/'+group['product']+'.npy')
    assert hashlib.sha256(raw).hexdigest()==group['mask_member_sha256']
    mask=np.load(io.BytesIO(raw),allow_pickle=False);assert mask.shape==(1022,1022,3) and mask.dtype==np.bool_
    assert (mask.reshape(-1,3)[indices].sum(1)==1).all()
    return mask.argmax(-1).reshape(-1)[indices].astype(np.uint8)
def calibrate_acceptance(thresholds,rows):
    locks=load(U/'delivery/THRESHOLD_LOCK.json');report={}
    bindings={v['group_id']:v for v in locks['bindings']}
    for arm in ('R','C'):
        expected=locks['calibration'][arm];t=thresholds[arm];b=expected['unsafe_negative_boundary'];rr=[];lower=[];ties=[];negative=[]
        for row in sorted(rows,key=lambda r:r['group']['group_id']):
            z=score(arm+'_Calibration','Calibration',row['product_id']);assert z['valid'].all()
            y=mask_truth(row['group'],z['flat_indices']);s=z['scores'].astype(np.float64);neg=y!=2;n=int(neg.sum())
            assert n==bindings[row['group']['group_id']]['negative'] and len(y)==bindings[row['group']['group_id']]['valid']
            if n:rr.append(int(((s>=t)&neg).sum())/n);lower.append(int(((s>=b)&neg).sum())/n)
            ties.append(int(((s==b)&neg).sum()));negative.append(n)
        assert len(rr)==expected['negative_groups'] and np.nextafter(t,-np.inf)==b
        assert float(np.mean(rr))<=.01 and float(np.mean(lower))>.01
        assert abs(np.mean(rr)*100-expected['macro_negative_FP_pp'])<1e-10 and ties==expected['atomic_negative_tie_counts']
        report[arm]=dict(negative_groups=len(rr),negative_pixels=sum(negative),macro_FP_pp=float(np.mean(rr))*100,
            adjacent_lower_macro_FP_pp=float(np.mean(lower))*100,atomic_ties=ties,float64_comparisons=True)
    save(O/'CALIBRATION_INDEPENDENT_CPU_ACCEPTANCE.json',dict(status='passed',models=report,model_calls=0,complete_groups_no_sampling=True))
def catalogue(rows,thresholds):
    masks={};frozen=[];cache={}
    for row in rows:
        pid=row['product_id'];a=score('R_Catalogue_reuse','Catalogue',pid);b=score('C_Catalogue','Catalogue',pid)
        assert np.array_equal(a['flat_indices'],b['flat_indices']);common=a['valid']&b['valid'];masks[pid]=common;cache[pid]=(a,b)
        assert len(common)==row['group']['common_valid']
        frozen.append(dict(product=pid,group_id=row['group']['group_id'],original=len(common),common=int(common.sum()),loss=int((~common).sum())))
    save(O/'Catalogue_COMMON_SUPPORT_LOCK.json',dict(before_reference_join=True,rows=frozen,reference_local_only=True))
    groups={}
    for row in rows:
        pid=row['product_id'];a,b=cache[pid];common=masks[pid]
        y=mask_truth(row['group'],a['flat_indices'])[common]
        groups[row['group']['group_id']]=dict(products=pid,truth=y,R=a['scores'][common],C=b['scores'][common])
    summaries,cmp=evaluate(groups,'Catalogue',thresholds,O)
    save(O/'Catalogue_CPU_ACCEPTANCE.json',dict(status='passed',support_loss=sum(r['loss'] for r in frozen),groups=len(groups),
        label_uploads=0,model_calls=0,AP_same_for_both_thresholds=True,independent_AP_ties_and_scalar_bootstrap=True))
    return summaries,cmp
def pixbox(plan,thresholds):
    by={'R':{},'C':{}};pids=[v['product_id'] for v in plan]
    for arm in ('R','C'):
        ss=load((R/'outputs/phase119/phase119_20261015/delivery/R' if arm=='R' else U/'delivery/C_PixBox')/'SCORES_SEALED.json')['scores_sha256']
        for pid in pids:
            path=(R/'data/pixbox_s2_cmix_5036991/phase119_20261015/R/scores/PixBox' if arm=='R' else P/'C_PixBox/scores/PixBox')/pid/'UNLABELLED_POINT_SCORES.json'
            assert sha(path)==ss['PixBox/'+pid]
            for q in load(path)['points']:assert q['point_id'] not in by[arm];by[arm][q['point_id']]=q
        assert len(by[arm])==9500
    support=[]
    for point,q in by['R'].items():
        c=by['C'][point];assert all(str(c[k])==str(q[k]) for k in ('product_id','group_id','row20','col20'))
        ok=all(v['valid'] and v.get('coverage',True) and v.get('author_valid',True) for v in (q,c))
        support.append(dict(point_id=point,common_valid=bool(ok)))
    save(P/'COMMON_SUPPORT_PRIVATE.json',dict(rows=support))
    save(O/'PixBox_COMMON_SUPPORT_LOCK.json',dict(before_reference_join=True,original=9500,common=sum(v['common_valid'] for v in support),
        loss=sum(not v['common_valid'] for v in support),mask_sha256=sha(P/'COMMON_SUPPORT_PRIVATE.json')))
    ref=R/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json'
    assert sha(ref)=='b1448f3fbd05c9e77af7ed0ea50dbde4be7480a35a4c532a2fec45789af25f96'
    refs={q['point_ids'][0]:q for q in load(ref)};assert set(refs)==set(by['R'])
    groups=collections.defaultdict(lambda:dict(products=set(),truth=[],R=[],C=[]));mapping={'negative_surface':0,'negative_cloud':1,'positive_ground_cloud_shadow':2}
    for s in support:
        point=s['point_id'];q=by['R'][point];ref=refs[point]
        assert all(str(q[k])==str(ref[k]) for k in ('product_id','group_id','row20','col20'))
        g=groups[q['group_id']];g['products'].add(q['product_id'])
        if not s['common_valid'] or ref['reference']=='ignore':continue
        g['truth'].append(mapping[ref['reference']])
        for p in ('R','C'):g[p].append(by[p][point]['shadow_confidence'])
    for g in groups.values():
        g['products']=','.join(sorted(g['products']));g['truth']=np.asarray(g['truth'],np.uint8)
        for p in ('R','C'):g[p]=np.asarray(g[p],np.float32)
    summaries,cmp=evaluate(groups,'PixBox',thresholds,O)
    save(O/'PixBox_CPU_ACCEPTANCE.json',dict(status='passed',support_loss=sum(not v['common_valid'] for v in support),groups=len(groups),
        label_uploads=0,model_calls=0,AP_same_for_both_thresholds=True,not_full_scene_FPR=True))
    return summaries,cmp
def main():
    start=time.time()
    from phase122_local_resources import start as resource_guard
    finish=resource_guard(R,[U,D,P],lambda name,v:save(O/name,v))
    for n,h in load(O/'PRE_EXECUTION_LOCK.json')['source_sha256'].items():assert sha(R/n)==h,n
    unpack();seal=load(U/'delivery/ALL_SCORES_SEALED.json');assert not seal['evaluation_labels_joined']
    assert sha(U/'delivery/THRESHOLD_LOCK.json')==seal['thresholds'] and sha(U/'delivery/TRAINING_SEAL.json')==seal['training']
    for stage,h in seal['settings'].items():assert sha(U/'delivery'/stage/'SCORES_SEALED.json')==h
    thresholds=load(U/'delivery/THRESHOLD_LOCK.json')['thresholds'];plan=load(U/'delivery/INPUT_PATCH_PLAN.json')['rows']
    calibrate_acceptance(thresholds,[r for r in plan if r['pool']=='Calibration'])
    data={};data['Catalogue']=catalogue([r for r in plan if r['pool']=='Catalogue'],thresholds)
    data['PixBox']=pixbox([r for r in plan if r['pool']=='PixBox'],thresholds)
    save(O/'LOCAL_CPU_ACCEPTANCE.json',dict(status='passed',elapsed_seconds=time.time()-start,model_calls=0,labels_uploaded=0,
        thresholds_sha256=seal['thresholds'],no_evaluation_threshold_search=True,AP_threshold_invariance=True))
    finish();print(json.dumps(data,indent=2))
if __name__=='__main__':main()

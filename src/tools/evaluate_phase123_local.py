"""Triple-support-first local reference joins; no neural calls or remote labels."""
import collections,hashlib,json,time,zipfile,shutil
from pathlib import Path
import numpy as np
from evaluate_phase122_local import mask_truth
from phase123_metrics import evaluate
R=Path(__file__).resolve().parents[2];U=R/'outputs/phase123/phase123_20261016';O=R/'src/research_results/phase123_20261016'
OLD=R/'outputs/phase122/phase122_20261015';D0=R/'data/sentinel2_cloud_mask_catalogue_4172871/phase122_20261015'
P0=R/'data/pixbox_s2_cmix_5036991/phase122_20261015';D=R/'data/sentinel2_cloud_mask_catalogue_4172871/phase123_20261016';P=R/'data/pixbox_s2_cmix_5036991/phase123_20261016'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def unpack():
    rec=load(U/'DELIVERY_RECEIPT.json');p=U/'phase123_delivery.zip';assert sha(p)==rec['sha256'] and p.stat().st_size==rec['bytes']
    with zipfile.ZipFile(p) as z:
        m=json.loads(z.read('DELIVERY_MANIFEST.json'))['files'];assert len(z.namelist())==len(m)+1
        for n,h in m.items():
            assert not n.startswith('/') and ':' not in n and '\\' not in n and '..' not in Path(n).parts
            b=z.read(n);assert hashlib.sha256(b).hexdigest()==h
            dest=((P if '/PixBox/' in n else D)/n) if '/scores/' in n else U/'delivery'/n
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(b)
    assert sha(U/'C_decoder_final.safetensors')==rec['decoder']['sha256']
def catalogue(rows,thresholds):
    cache={};support=[]
    for row in rows:
        pid=row['product_id'];data={}
        for arm,base,stage,meta in [('R',D0,'R_Catalogue_reuse',OLD),('C122',D0,'C_Catalogue',OLD),('C123',D,'C_Catalogue',U)]:
            f=base/stage/'scores/Catalogue'/(pid+'.npz');assert sha(f)==load(meta/'delivery'/stage/'SCORES_SEALED.json')['scores_sha256']['Catalogue/'+pid]
            with np.load(f,allow_pickle=False) as z:data[arm]={k:z[k].copy() for k in ('scores','valid','flat_indices')}
        a=data['R'];assert all(np.array_equal(v['flat_indices'],a['flat_indices']) for v in data.values())
        common=np.logical_and.reduce([v['valid'] for v in data.values()]);cache[pid]=(data,common)
        support.append(dict(product=pid,group_id=row['group']['group_id'],original=len(common),common=int(common.sum()),loss=int((~common).sum())))
    save(O/'Catalogue_COMMON_SUPPORT_LOCK.json',dict(before_reference_join=True,rows=support))
    groups={}
    for row in rows:
        pid=row['product_id'];data,common=cache[pid];y=mask_truth(row['group'],data['R']['flat_indices'])[common]
        groups[row['group']['group_id']]=dict(products=pid,truth=y,**{arm:v['scores'][common] for arm,v in data.items()})
    result=evaluate(groups,'Catalogue',thresholds,O)
    save(O/'Catalogue_CPU_ACCEPTANCE.json',dict(status='passed',groups=len(groups),positive_groups=int(sum(np.any(v['truth']==2) for v in groups.values())),
        support_loss=sum(v['loss'] for v in support),valid=sum(len(v['truth']) for v in groups.values()),labels_uploaded=0,model_calls=0))
    return result
def pixbox(rows,thresholds):
    by={arm:{} for arm in ('R','C122','C123')}
    for arm,base,meta,stage in [('R',R/'data/pixbox_s2_cmix_5036991/phase119_20261015',R/'outputs/phase119/phase119_20261015','R'),('C122',P0,OLD,'C_PixBox'),('C123',P,U,'C_PixBox')]:
        ss=load(meta/'delivery'/stage/'SCORES_SEALED.json')['scores_sha256']
        for row in rows:
            pid=row['product_id'];f=base/stage/'scores/PixBox'/pid/'UNLABELLED_POINT_SCORES.json';assert sha(f)==ss['PixBox/'+pid]
            for q in load(f)['points']:assert q['point_id'] not in by[arm];by[arm][q['point_id']]=q
        assert len(by[arm])==9500
    support=[]
    for point,r in by['R'].items():
        values=[v[point] for v in by.values()]
        assert all(all(str(q[k])==str(r[k]) for k in ('product_id','group_id','row20','col20')) for q in values)
        support.append(dict(point_id=point,common_valid=all(q['valid'] and q.get('coverage',True) and q.get('author_valid',True) for q in values)))
    save(P/'COMMON_SUPPORT_PRIVATE.json',dict(rows=support))
    save(O/'PixBox_COMMON_SUPPORT_LOCK.json',dict(before_reference_join=True,original=9500,common=sum(v['common_valid'] for v in support),loss=sum(not v['common_valid'] for v in support),mask_sha256=sha(P/'COMMON_SUPPORT_PRIVATE.json')))
    ref=R/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json';assert sha(ref)=='b1448f3fbd05c9e77af7ed0ea50dbde4be7480a35a4c532a2fec45789af25f96'
    refs={q['point_ids'][0]:q for q in load(ref)};assert set(refs)==set(by['R'])
    mapping={'negative_surface':0,'negative_cloud':1,'positive_ground_cloud_shadow':2}
    groups=collections.defaultdict(lambda:dict(products=set(),truth=[],R=[],C122=[],C123=[]))
    for s in support:
        point=s['point_id'];q=by['R'][point];ref=refs[point]
        assert all(str(q[k])==str(ref[k]) for k in ('product_id','group_id','row20','col20'))
        g=groups[q['group_id']];g['products'].add(q['product_id'])
        if not s['common_valid'] or ref['reference']=='ignore':continue
        g['truth'].append(mapping[ref['reference']])
        for arm in by:g[arm].append(by[arm][point]['shadow_confidence'])
    for g in groups.values():
        g['products']=','.join(sorted(g['products']));g['truth']=np.asarray(g['truth'],np.uint8)
        for arm in by:g[arm]=np.asarray(g[arm],np.float32)
    result=evaluate(groups,'PixBox',thresholds,O)
    save(O/'PixBox_CPU_ACCEPTANCE.json',dict(status='passed',groups=len(groups),positive_groups=int(sum(np.any(v['truth']==2) for v in groups.values())),
        support_loss=sum(not v['common_valid'] for v in support),valid=sum(len(v['truth']) for v in groups.values()),labels_uploaded=0,model_calls=0,not_full_scene_FPR=True))
    return result
def main():
    start=time.time()
    from phase122_local_resources import start as guard
    finish=guard(R,[U,D,P],lambda n,v:save(O/n,v))
    cfg=load(O/'PRE_EXECUTION_LOCK.json')
    for n,h in cfg['source_sha256'].items():assert sha(R/n)==h,n
    unpack();seal=load(U/'delivery/ALL_SCORES_SEALED.json');assert not seal['evaluation_labels_joined']
    assert sha(U/'delivery/THRESHOLD_LOCK.json')==seal['thresholds'] and sha(U/'delivery/TRAINING_SEAL.json')==seal['training']
    for name,h in seal['settings'].items():assert sha(U/'delivery'/name/'SCORES_SEALED.json')==h
    check=load(U/'delivery/CALIBRATION_INDEPENDENT_CPU_ACCEPTANCE.json');assert check['status']=='passed' and check['full53_groups_no_sampling'] and check['model_calls']==0
    assert check['source_sha256']==sha(R/'src/tools/collect_phase123_delivery.py') and check['threshold_lock_sha256']==seal['thresholds']
    new=load(U/'delivery/THRESHOLD_LOCK.json');old=load(OLD/'delivery/THRESHOLD_LOCK.json');assert new['thresholds']['R']==old['thresholds']['R'] and new['thresholds']['C122']==old['thresholds']['C']
    t=dict(R=new['thresholds']['R'],C122=old['thresholds']['C'],C123=new['thresholds']['C'])
    plan=load(U/'delivery/INPUT_PATCH_PLAN.json')['rows'];catalogue([v for v in plan if v['pool']=='Catalogue'],t);pixbox([v for v in plan if v['pool']=='PixBox'],t)
    save(O/'LOCAL_CPU_ACCEPTANCE.json',dict(status='passed',model_calls=0,labels_uploaded=0,elapsed_seconds=time.time()-start,AP_threshold_invariance=True,bootstrap_seed=123015))
    finish();print('Phase123 local evaluation passed')
if __name__=='__main__':main()

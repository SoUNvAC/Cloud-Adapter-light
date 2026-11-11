"""Native-label and leave-one-scene-out audit using existing local caches only."""
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/research_audits/label_scope_20261027'
S=ROOT/'outputs/phase230/phase230_20261026/delivery'
R=ROOT/'outputs/phase317/phase317_20261026/remote_backup'
KEY='LC81500412014244LGN00'
HASHES={}
NAMES=['Shadow','Shadow_over_Water','Water','Snow','Land','Cloud','Flooded']


def pin(p):
    h=hashlib.sha256(p.read_bytes()).hexdigest();HASHES[str(p.relative_to(ROOT))]=h;return h


def js(p):
    pin(p);return json.loads(p.read_text(encoding='utf-8'))


def literal(name,path):
    pin(path)
    for x in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in x.targets):
            return ast.literal_eval(x.value.args[0] if isinstance(x.value,ast.Call) else x.value)
    raise ValueError(name)


def metric(cm):
    a=np.asarray(cm,np.float64);support=a.sum(1);u=support+a.sum(0)-np.diag(a)
    return dict(mIoU=float(np.nanmean(np.divide(np.diag(a)*100,u,out=np.full(3,np.nan),where=u>0))),
                Surface_to_Cloud_FPR=float(a[0,1]*100/support[0]),
                Shadow_recall=float(a[2,2]*100/support[2]),Cloud_recall=float(a[1,1]*100/support[1]))


def save(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    support=js(ROOT/'outputs/phase229/phase229_20261026/INPUT_SUPPORT.json')
    refs=js(R/'REFERENCE_LOCK.json')
    pins=js(R/'REFERENCE_INPUT_SHA256.json')
    ref_by_name={x['name']:x for x in refs['rows'] if x['pool']=='SPARCS'}
    map_s=literal('SPARCS_NATIVE_TO_PARENT',ROOT/'src/cloud_adapter/datasets/sparcs_manifest.py')
    map_l=literal('L8_TO_PARENT',ROOT/'src/cloud_adapter/datasets/phase64_parent.py')
    evaluate=ROOT/'src/tools/evaluate_phase64_source_only_parent.py'
    assert list(map_l.values())==literal('L8_TO_PARENT',evaluate)
    assert list(map_s.values())==literal('SPARCS_TO_PARENT',evaluate)
    parent_lut=np.array([map_s[i] for i in range(7)],np.uint8)
    receipt=js(S/'SCORE_RECEIPTS.json')
    b2={r['model']:r['b'] for r in js(ROOT/'outputs/phase231/phase231_20261026/TABLE1.json')}
    b3={r['model']:r['b3'] for r in js(ROOT/'outputs/phase232/phase232_20261027/TABLE1.json')}
    expected232={(x['model'],x['decision'],x['scene']):x['confusion'] for x in js(ROOT/'outputs/phase232/phase232_20261027/VAL_PER_SCENE.json')}
    expected_raw={}
    for model in ['Source','L8_21701','L8_21702','SPARCS_21701','SPARCS_21702']:
        expected_raw[model]=js(ROOT/f'outputs/phase229/phase229_20261026/delivery/evaluation/SPARCS_{model}.json')['per_group']
    hist=[];errors=[];added=[];matrices={};raw_masks={};checks=[]
    for row in support['SPARCS']:
        name=row['name'];scene=row['scene'];ref=ref_by_name[name]
        p=R/'reference'/Path(ref['file']).name
        assert pin(p)==ref['sha256']
        assert pins[row['mask_path']]==row['mask_sha256']
        with np.load(p,allow_pickle=False) as z:
            native=z['native'];truth=z['truth'];valid=z['valid']
        old=S/'labels'/f'{scene}.npy';y=np.load(old,allow_pickle=False);pin(old)
        assert native.shape==y.shape==(512,512) and valid.all()
        assert np.array_equal(parent_lut[native],y) and np.array_equal(truth,y)
        raw_masks[scene]=native
        for k in range(7):hist.append(dict(scene=scene,native_code=k,native_class=NAMES[k],parent=['Surface','Cloud','Shadow'][map_s[k]],pixels=int((native==k).sum())))
        checks.append({'scene':scene,'native_cache_SHA':True,'original_mask_SHA_matches_current_manifest':True,'parent_mapping_equals_Phase230_labels':True})
    for rec in receipt:
        model=rec['model'];scene=rec['scene'];native=raw_masks[scene];y=parent_lut[native]
        p=S/rec['score_path'];assert pin(p)==rec['score_SHA']
        scores=np.load(p,allow_pickle=False).astype(np.float64);raw=scores.argmax(0)
        policies={'Raw':0.}
        if model in b2:policies.update(B2=b2[model],B3=b3[model])
        for policy,b in policies.items():
            q=scores.copy();q[1]*=np.exp(b);pred=q.argmax(0)
            cm=np.bincount((y.astype(np.int64)*3+pred).ravel(),minlength=9).reshape(3,3)
            if policy=='Raw':assert cm.tolist()==expected_raw[model][scene]
            if model in b2:assert cm.tolist()==expected232[model,policy,scene]
            matrices[model,policy,scene]=cm
            for k in range(7):
                mask=native==k
                for j,nn in enumerate(['Surface','Cloud','Shadow']):
                    errors.append(dict(model=model,policy=policy,scene=scene,native_code=k,native_class=NAMES[k],reference_parent=['Surface','Cloud','Shadow'][map_s[k]],prediction=nn,pixels=int((mask&(pred==j)).sum())))
                if map_s[k]==0 and policy!='Raw':
                    added.append(dict(model=model,policy=policy,scene=scene,native_code=k,native_class=NAMES[k],reference_pixels=int(mask.sum()),
                                      new_to_Cloud=int((mask&(raw!=1)&(pred==1)).sum()),
                                      raw_correct_Surface_lost=int((mask&(raw==0)&(pred==1)).sum())))
    excluded=[]
    scenes=sorted(raw_masks)
    for mode in ['all10','exclude_FP_focus_scene']:
        chosen=scenes if mode=='all10' else [x for x in scenes if x!=KEY]
        aggregate={}
        for model,policy,scene in matrices:
            if scene not in chosen:continue
            key=model,policy
            aggregate[key]=aggregate.get(key,np.zeros((3,3),np.int64))+matrices[model,policy,scene]
        for (model,policy),cm in aggregate.items():
            m=metric(cm);base=metric(aggregate[model,'Raw'])
            own=metric(aggregate['SPARCS_'+model.split('_')[-1],'Raw']) if model.startswith('L8_') else None
            src=metric(aggregate['Source',policy]) if model in b2 else metric(aggregate['Source','Raw'])
            row=dict(support=mode,scenes=len(chosen),excluded_scene=KEY if mode!='all10' else '',model=model,policy=policy)
            row.update(m);row.update(delta_mIoU_vs_same_model_Raw=m['mIoU']-base['mIoU'],
                                     delta_Surface_FPR_vs_Raw=m['Surface_to_Cloud_FPR']-base['Surface_to_Cloud_FPR'],
                                     delta_Shadow_recall_vs_Raw=m['Shadow_recall']-base['Shadow_recall'],
                                     delta_mIoU_vs_Source_same_policy=m['mIoU']-src['mIoU'],
                                     delta_mIoU_vs_native_SPARCS_Raw=m['mIoU']-own['mIoU'] if own else None)
            excluded.append(row)
    # Metadata only: do not import or compare the other branch's trained models.
    l8refs={x['name']:x for x in refs['rows'] if x['pool']=='L8'}
    # Another branch uses a different root for byte-identical annotation copies.
    # Match annotation basename AND hash, require uniqueness, never basename alone.
    matches=[]
    for x in support['L8']:
        hits=[(p,h) for p,h in pins.items() if '/ann_dir/' in p and Path(p).name==x['name'] and h==x['mask_sha256']]
        if x['name'] in l8refs and len(hits)==1:matches.append(x)
    source_l8=js(ROOT/'outputs/phase229/phase229_20261026/delivery/evaluation/L8_Source.json')
    old_margins={x['name']:np.asarray(x['confusion']).sum(1) for x in source_l8['per_input']}
    l8support=[]
    for x in matches:
        rr=l8refs[x['name']];pp=R/'reference'/Path(rr['file']).name
        assert pin(pp)==rr['sha256']
        with np.load(pp,allow_pickle=False) as z:
            raw=z['native'];truth=z['truth'];valid=z['valid']
        assert np.isin(raw,[0,64,128,192,255]).all() and np.array_equal(valid,raw!=0)
        parent=np.zeros(raw.shape,np.uint8);parent[raw==64]=2;parent[np.isin(raw,[192,255])]=1
        assert np.array_equal(parent[valid],truth[valid]) and np.all(truth[~valid]==255)
        assert np.array_equal(np.bincount(parent.ravel(),minlength=3),old_margins[x['name']])
        assert int((raw==0).sum())==rr['invalid']
        l8support.append(dict(scene=x['scene'],name=x['name'],raw_fill_pixels=int((raw==0).sum()),
                              raw_clear_pixels=int((raw==128).sum()),thin_cloud_pixels=int((raw==192).sum()),
                              cloud_pixels=int((raw==255).sum()),shadow_pixels=int((raw==64).sum()),
                              current_reference_pixels=int(raw.size)))
    meta={'current_L8_val_inputs':len(support['L8']),'matched_raw_reference_metadata_inputs':len(matches),
          'metadata_reported_raw_fill_pixels':sum(l8refs[x['name']]['invalid'] for x in matches),
          'all_current_masks_exactly_reproduced_in_reference_receipt':refs['L8_all_PNG_reproduced_exactly'],
          'native_arrays_checked':len(l8support),
          'current_confusion_reference_margins_match_fill_as_Surface':True,
          'raw_fill_fraction_of_original_validation':sum(x['raw_fill_pixels'] for x in l8support)/sum(x['current_reference_pixels'] for x in l8support),
          'status':'Native arrays and original mask hashes checked; no L8 predictions available for fill-excluded performance recomputation.'}
    save('SPARCS_NATIVE_SUPPORT.csv',hist)
    save('NATIVE_CLASS_PREDICTION_COUNTS.csv',errors)
    save('ADDED_CLOUD_ERRORS_BY_NATIVE_CLASS.csv',added)
    save('WITH_WITHOUT_FP_SCENE.csv',excluded)
    save('L8_NATIVE_REFERENCE_SUPPORT.csv',l8support)
    (OUT/'CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    (OUT/'L8_FILL_METADATA_REVIEW.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    for name in ['src/tools/reference_phase317.py','src/dataset/l8_biome.py','src/dataset/l8_biome_crop.py','src/research_plans/protocol_data/phase64_dataset_registry.json']:
        pin(ROOT/name)
    (OUT/'AUDIT.json').write_text(json.dumps({'inputs':HASHES,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                             'new_model_calls':0,'training':0,'new_CI':0,'remote_connections':0,
                                             'other_branch_use':'Only native reference labels and provenance for the exact current SPARCS val scenes; no other branch models or performance.',
                                             'L8_native_arrays_read':len(l8support),'L8_new_performance_computed':False},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'focus_histogram':[x for x in hist if x['scene']==KEY],
                      'focus_added_B2':[x for x in added if x['scene']==KEY and x['policy']=='B2'],
                      'L8_fill_metadata':meta},indent=2))


if __name__=='__main__':main()

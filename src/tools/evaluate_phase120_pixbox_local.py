"""Local five-policy PixBox support freeze before original-reference join."""
import collections,hashlib,json,zipfile,csv
from pathlib import Path
import numpy as np
from phase120_metrics import deliver
R=Path(__file__).resolve().parents[2];U=R/'outputs/phase120/phase120_20261015'
O=R/'src/research_results/phase120_20261015';D=R/'data/pixbox_s2_cmix_5036991/phase120_20261015'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,o):
    p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('x',encoding='utf-8') as f:json.dump(o,f,indent=2,allow_nan=False);f.write('\n')
def unpack():
    rec=load(U/'DELIVERY_RECEIPT.json');archive=U/'phase120_delivery.zip'
    assert archive.stat().st_size==rec['bytes'] and sha(archive)==rec['sha256']
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('DELIVERY_MANIFEST.json'))['files'];assert len(z.namelist())==len(manifest)+1
        for name,h in manifest.items():
            assert not name.startswith('/') and ':' not in name and '\\' not in name and '..' not in Path(name).parts
            raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==h
            dest=(D/name) if '/scores/PixBox/' in name else U/'delivery'/name
            dest.parent.mkdir(exist_ok=True,parents=True)
            with dest.open('xb') as f:f.write(raw)
    return rec

def main():
    for n,h in load(O/'EVALUATION_LOCK.json')['sources'].items():assert sha(R/n)==h,n
    rec=unpack();seal=load(U/'delivery/ALL_SCORES_SEALED.json')
    assert not seal['PixBox_labels_joined'] and seal['training']==sha(U/'delivery/TRAINING_SEAL.json')
    models=load(U/'delivery/TRAINING_SEAL.json')['models']
    for p,h in models.items():assert sha(U/(p+'_decoder_final.safetensors'))==h==rec['decoder_checkpoints'][p]['sha256']
    pids=[p['product_id'] for p in load(R/'src/research_results/phase104_20261013/object_manifest.json')['products']]
    by={p:{} for p in ('E','R','N','C','D')}
    for p in by:
        if p=='E':scores=load(R/'src/research_results/phase112_20261014/SCORES_SEALED.json')['score_sha256']
        elif p in ('R','N'):scores=load(R/'outputs/phase119/phase119_20261015/delivery'/p/'SCORES_SEALED.json')['scores_sha256']
        else:
            ss=U/'delivery'/(p+'_PixBox')/'SCORES_SEALED.json';assert sha(ss)==seal['settings'][p+'_PixBox'];scores=load(ss)['scores_sha256']
        for pid in pids:
            if p=='E':path=R/'data/pixbox_s2_cmix_5036991/phase112_20261014/omni'/pid/'UNLABELLED_POINT_SCORES.json'
            elif p in ('R','N'):path=R/'data/pixbox_s2_cmix_5036991/phase119_20261015'/p/'scores/PixBox'/pid/'UNLABELLED_POINT_SCORES.json'
            else:path=D/(p+'_PixBox')/'scores/PixBox'/pid/'UNLABELLED_POINT_SCORES.json'
            assert sha(path)==scores[pid if p=='E' else 'PixBox/'+pid]
            for q in load(path)['points']:
                assert q['point_id'] not in by[p];by[p][q['point_id']]=q
        assert len(by[p])==9500
    support=[];counts=collections.defaultdict(collections.Counter)
    for point,q in by['R'].items():
        ok=True
        for p in by:
            v=by[p][point];assert all(str(v[k])==str(q[k]) for k in ('product_id','row20','col20'))
            if 'group_id' in v:assert str(v['group_id'])==str(q['group_id'])
            ok &= v['valid'] and v.get('coverage',True) and v.get('author_valid',True)
        support.append({'point_id':point,'common_valid':bool(ok)})
        counts[q['group_id']]['original']+=1;counts[q['group_id']]['common']+=int(ok)
    save(D/'COMMON_SUPPORT_PRIVATE.json',{'rows':support})
    save(O/'PixBox_COMMON_SUPPORT_LOCK.json',{'before_original_reference_join':True,'settings':list(by),'original':9500,
        'common_valid':sum(v['common_valid'] for v in support),'rows_by_group':dict(counts),'mask_sha256':sha(D/'COMMON_SUPPORT_PRIVATE.json'),
        'no_extra_Source_six_band_filter':True})
    path=R/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json'
    assert sha(path)=='b1448f3fbd05c9e77af7ed0ea50dbde4be7480a35a4c532a2fec45789af25f96'
    refs={v['point_ids'][0]:v for v in load(path)};assert set(refs)==set(by['R'])
    groups=collections.defaultdict(lambda:{'products':set(),'truth':[],**{p:[] for p in by}})
    for v in support:
        point=v['point_id'];q=by['R'][point];r=refs[point];g=groups[q['group_id']];g['products'].add(q['product_id'])
        assert all(str(r[k])==str(q[k]) for k in ('product_id','group_id','row20','col20'))
        if not v['common_valid'] or r['reference']=='ignore':continue
        g['truth'].append(r['reference']=='positive_ground_cloud_shadow')
        for p in by:g[p].append(by[p][point]['shadow_confidence'])
    for g in groups.values():
        g['products']=','.join(sorted(g['products']));g['truth']=np.array(g['truth'],bool)
        for p in by:g[p]=np.array(g[p],np.float32)
    per,summaries,cmp=deliver(O,'PixBox',groups)
    expected=list(csv.DictReader((R/'src/research_results/phase119_20261015/PixBox_summary.csv').open()))
    if all(v['common_valid'] for v in support):
        for old in expected:
            new=next(v for v in summaries if v['setting']==old['setting'])
            assert abs(new['macro_AP_pp']-float(old['macro_AP_pp']))<1e-8 and abs(new['pooled_AP_pp']-float(old['pooled_AP_pp']))<1e-8
    save(O/'PixBox_CPU_ACCEPTANCE.json',{'status':'passed','groups':len(per),'AP_groups':summaries[0]['AP_groups'],
        'original_E_R_N_recomputed':True,'support_loss':sum(not v['common_valid'] for v in support),
        'independent_tied_AP_and_scalar_bootstrap':True,'model_calls':0,'threshold':.5,'not_full_scene_FPR':True})
    print(json.dumps({'summaries':summaries,'comparisons':cmp},indent=2))
if __name__=='__main__':main()

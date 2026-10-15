"""Local original PixBox reference join, only after all E/R/N scores are sealed."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from phase119_metrics import deliver

R=Path(__file__).resolve().parents[2]
U=R/'outputs/phase119/phase119_20261015'
D=R/'data/pixbox_s2_cmix_5036991/phase119_20261015'
O=R/'src/research_results/phase119_20261015'


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')


def main():
    cfg=load(O/'EVALUATION_LOCK.json')
    for n,h in cfg['source_sha256'].items():assert sha(R/n)==h,n
    seal=load(U/'delivery/ALL_SCORES_SEALED.json')
    assert not seal['labels_joined']
    by={p:{} for p in ('E','R','N')};oldseal=load(R/'src/research_results/phase112_20261014/SCORES_SEALED.json')
    pids=[p['product_id'] for p in load(R/'src/research_results/phase104_20261013/object_manifest.json')['products']]
    for p in by:
        if p!='E':
            ss=U/'delivery'/p/'SCORES_SEALED.json';assert sha(ss)==seal['settings'][p];scores=load(ss)['scores_sha256']
        for pid in pids:
            path=(R/'data/pixbox_s2_cmix_5036991/phase112_20261014/omni'/pid/'UNLABELLED_POINT_SCORES.json') if p=='E' else D/p/'scores/PixBox'/pid/'UNLABELLED_POINT_SCORES.json'
            assert sha(path)==(oldseal['score_sha256'][pid] if p=='E' else scores['PixBox/'+pid])
            for q in load(path)['points']:
                assert q['point_id'] not in by[p];by[p][q['point_id']]=q
        assert len(by[p])==9500
    support=[];loss=collections.defaultdict(collections.Counter)
    for point_id,q in by['R'].items():
        n=by['N'][point_id];e=by['E'][point_id]
        assert all(str(q[k])==str(v[k]) for v in (e,n) for k in ('product_id','row20','col20'))
        ok=q['valid'] and n['valid'] and e['valid'] and e['coverage'] and e['author_valid']
        support.append({'point_id':point_id,'common_valid':bool(ok)})
        loss[q['group_id']]['original']+=1;loss[q['group_id']]['common']+=int(ok)
    save(D/'COMMON_SUPPORT_PRIVATE.json',{'rows':support})
    save(O/'PixBox_COMMON_SUPPORT_LOCK.json',{'before_original_reference_join':True,'all_three_settings_sealed':True,
         'common_valid':sum(r['common_valid'] for r in support),'original':9500,'mask_sha256':sha(D/'COMMON_SUPPORT_PRIVATE.json'),
         'loss_by_group':dict(loss),'no_extra_Source_six_band_mask':True})
    path=R/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json'
    assert sha(path)==cfg['original_reference_sha256']
    refs={q['point_ids'][0]:q for q in load(path)};assert set(refs)==set(by['R'])
    groups=collections.defaultdict(lambda:{'products':set(),'truth':[], 'E':[],'R':[],'N':[]})
    baseline=collections.defaultdict(lambda:{'products':set(),'truth':[], 'E':[],'R':[],'N':[]})
    joined=[]
    for mask in support:
        point_id=mask['point_id'];q=by['R'][point_id];r=refs[point_id]
        assert all(str(q[k])==str(r[k]) for k in ('product_id','group_id','row20','col20'))
        gid=q['group_id'];groups[gid]['products'].add(q['product_id']);baseline[gid]['products'].add(q['product_id'])
        label=r['reference'];scores={p:by[p][point_id]['shadow_confidence'] for p in by}
        joined.append(dict(mask,reference=label,**scores))
        if label=='ignore':continue
        positive=label=='positive_ground_cloud_shadow'
        if by['E'][point_id]['valid']:
            baseline[gid]['truth'].append(positive)
            for p in by:baseline[gid][p].append(scores['E'])
        if not mask['common_valid']:continue
        groups[gid]['truth'].append(positive)
        for p in by:groups[gid][p].append(scores[p])
    save(D/'JOINED_REFERENCE_SCORES_PRIVATE.json',{'rows':joined})
    for collection in (groups,baseline):
        for g in collection.values():
            g['products']=','.join(sorted(g['products']));g['truth']=np.array(g['truth'],bool)
            for p in by:g[p]=np.array(g[p],np.float32)
    from phase119_metrics import evaluate
    _,oldsummary,_=evaluate(baseline,'PixBox_original_E')
    import csv
    with (R/'src/research_results/phase118_20261014/three_setting_summary.csv').open() as f:
        expected=next(r for r in csv.DictReader(f) if r['setting']=='O20')
    assert abs(oldsummary[0]['macro_AP_pp']-float(expected['macro_AP_pp']))<1e-8
    assert abs(oldsummary[0]['pooled_AP_pp']-float(expected['pooled_AP_pp']))<1e-8
    per,summary,comparisons=deliver(O,'PixBox',groups)
    save(O/'PixBox_CPU_ACCEPTANCE.json',{'status':'passed','old_E_baseline_reproduced':True,
         'old_E_macro_AP_pp':oldsummary[0]['macro_AP_pp'],'old_E_pooled_AP_pp':oldsummary[0]['pooled_AP_pp'],
         'independent_tied_AP_and_scalar_group_bootstrap':True,'all10_original_groups_retained':len(per)==10,
         'actual_AP_groups':summary[0]['AP_groups'],'common_valid':sum(r['common_valid'] for r in support),
         'threshold':.5,'negative_FP_definition':'selected negative reference points; not full-scene FPR',
         'support_sealed_before_original_reference_join':True,'Omni_calls':0})
    print(json.dumps({'summary':summary,'comparisons':comparisons},indent=2))


if __name__=='__main__':main()

"""Catalogue-only CPU evaluation after all model scores and support are sealed."""
import gc
from pathlib import Path
import sys
import numpy as np
from phase119_common import ROOT,OLD,REPO,setup,load,write,sha,catalog_image
from phase119_metrics import ap,deliver


def main():
    setup();assert (ROOT/'EXIT_CODE').read_text().strip()=='0'
    for name,h in load(REPO/'src/research_results/phase119_20261015/EVALUATION_LOCK.json')['source_sha256'].items():
        assert sha(REPO/name)==h,name
    seal=load(ROOT/'ALL_SCORES_SEALED.json')
    assert not seal['labels_joined'] and len(seal['settings'])==3
    for p,h in seal['settings'].items():assert sha(ROOT/p/'SCORES_SEALED.json')==h
    bindings=[r for r in load(ROOT/'INPUT_PATCH_PLAN.json')['rows'] if r['pool']=='Catalogue']
    masks={};support=[]
    for row in bindings:
        pid=row['product_id'];vectors=[]
        assert sha(OLD/'scores'/f'O_{pid}.npz')==row['old_score_sha256']
        for p in ('R','N'):
            path=ROOT/p/'scores/Catalogue'/(pid+'.npz')
            assert sha(path)==load(ROOT/p/'SCORES_SEALED.json')['scores_sha256']['Catalogue/'+pid]
            with np.load(path,allow_pickle=False) as z:
                vectors.append((z['valid'],z['flat_indices']))
        assert np.array_equal(vectors[0][1],vectors[1][1])
        common=vectors[0][0]&vectors[1][0]
        assert len(common)==row['group']['common_valid']
        path=ROOT/'common_support'/(pid+'.npy');path.parent.mkdir(exist_ok=True)
        with path.open('xb') as f:np.save(f,common,allow_pickle=False)
        masks[pid]=common
        support.append({'group_id':row['group']['group_id'],'product':pid,'original_valid':len(common),
                        'common_valid':int(common.sum()),'loss':int((~common).sum()),'mask_sha256':sha(path)})
    write(ROOT/'Catalogue_COMMON_SUPPORT_LOCK.json',{'before_reference_semantics_join':True,'groups':support,
          'all_three_settings_scores_sealed':True,'old_Phase117_support_preserved':True})
    groups={};baseline_scores=[];baseline_truth=[];baseline_ap=[];cpu=[]
    for row in bindings:
        pid=row['product_id'];g=row['group'];common=masks[pid]
        with np.load(OLD/'scores'/f'O_{pid}.npz',allow_pickle=False) as z:
            truth=z['truth'];old=z['scores']
        _,valid,original_truth=catalog_image(g,read_truth=True)
        assert np.array_equal(truth,original_truth) and len(truth)==len(common)
        raw=[];record={'products':pid,'truth':(truth[common]==2),'E':old[common]}
        for p in ('R','N'):
            with np.load(ROOT/p/'scores/Catalogue'/(pid+'.npz'),allow_pickle=False) as z:
                assert np.array_equal(z['flat_indices'],np.flatnonzero(valid))
                record[p]=z['scores'][common];raw.append(z['raw_logits'][:,common])
        mean=(raw[0]+raw[1])*np.float32(.5)
        soft=np.exp(mean-mean.max(0,keepdims=True));soft/=soft.sum(0,keepdims=True)
        reconstructed=np.clip(soft[3]+np.float32(.001),.001,.999)
        delta=float(np.max(np.abs(reconstructed-old[common]))) if common.any() else 0.
        assert delta<=1e-5,(pid,delta)
        cpu.append({'product':pid,'E_from_real_single_raw_logits_max_error':delta,'no_confidence_inversion':True})
        groups[g['group_id']]=record
        baseline_scores.append(old);baseline_truth.append(truth==2)
        a=ap(truth==2,old)
        if a is not None:baseline_ap.append(a)
        del raw,mean,soft,reconstructed;gc.collect()
    expected=load(REPO/'src/research_results/phase117_20261014/FIVE_POLICY_SUMMARY.json')['O']
    macro=float(np.mean(baseline_ap))*100;pooled=ap(np.concatenate(baseline_truth),np.concatenate(baseline_scores))*100
    assert len(baseline_ap)==expected['AP_groups']==7
    assert abs(macro-expected['macro_AP_pp'])<1e-8 and abs(pooled-expected['pooled_AP_pp'])<1e-8
    result,summary,comparison=deliver(ROOT/'aggregate','Catalogue',groups)
    write(ROOT/'Catalogue_CPU_ACCEPTANCE.json',{'status':'passed','baseline_E_macro_AP_pp':macro,'baseline_E_pooled_AP_pp':pooled,
          'old_baseline_reproduced':True,'all_single_mosaics_independently_checked':True,'E_reconstruction':cpu,
          'independent_tied_AP_groups_and_pool':True,'group_bootstrap_scalar_replay':True,
          'common_support_locked_before_reference_semantics_join':True,'evaluated_fit_groups':0,
          'AP_groups':sum(r['E_AP'] is not None for r in result),'all_groups':len(result),
          'support_loss':sum(r['loss'] for r in support),'threshold':.5,
          'negative_FP_definition':'dense released-reference nonshadow pixels, not PixBox selected points'})
    print({'summary':summary,'comparisons':comparison},flush=True)


if __name__=='__main__':main()

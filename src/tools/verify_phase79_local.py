"""Verify report hashes, reconstruct frozen B11 bins, and reaggregate all AUCs."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np
from run_phase79 import darkness_bins,TASKS,NAMES


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reports',type=Path,required=True);a=ap.parse_args();out=a.reports
    hashes=json.loads((out/'artifact_hashes.json').read_text())
    for rel,h in hashes.items():assert (out/rel).resolve().is_relative_to(out.resolve()) and sha(out/rel)==h,rel
    lock=json.loads((out/'input_code_lock.json').read_text());summary=json.loads((out/'summary.json').read_text());assert summary['status']=='completed' and (out/'EXIT_CODE').read_text().strip()=='0'
    for rel,key in [('src/tools/run_phase79.py','code_sha256'),('src/tools/run_phase76.py','audit_helper_sha256'),('src/tools/verify_phase78_local.py','AP_helper_sha256'),('src/tools/test_phase79.py','test_sha256'),('src/tools/run_phase79.sh','runner_sha256'),('src/research_plans/PHASE79_CONDITIONAL_DIRECTION_20261011.md','protocol_sha256'),('src/research_plans/PHASE79_INPUT_BINDINGS_20261011.json','bindings_sha256')]:
        assert hashlib.sha256(subprocess.check_output(['git','show',lock['git_commit']+':'+rel])).hexdigest()==lock[key],rel
    prior=Path('outputs/phase78/phase78_20261011');oldlock=json.loads((prior/'input_formula_lock.json').read_text());old=json.loads((prior/'summary.json').read_text())
    for rel,h in lock['previous_inputs'].items():assert sha(prior/rel)==h,rel
    remote_arrays=json.loads((prior/'REMOTE_ARRAY_VERIFIED.json').read_text())['native_arrays_retained_shared']
    for rel,h in lock['previous_artifact_hashes'].items():
        if rel in remote_arrays:assert remote_arrays[rel]==h
        else:assert sha(prior/rel)==h
    previous=Path('outputs/phase76/phase76_20261010');assert sha(previous/'artifact_hashes.json')==oldlock['previous']['artifact_hashes.json'];oldhash=json.loads((previous/'artifact_hashes.json').read_text())
    coverage=table(out/'bin_coverage.csv');bins=table(out/'bin_AUC.csv');scene=summary['scenes'];checked_bins=0
    with zipfile.ZipFile('data/sentinel2_cloud_mask_catalogue_4172871/subscenes.zip') as images,zipfile.ZipFile('data/sentinel2_cloud_mask_catalogue_4172871/masks.zip') as masks:
        for r in lock['selected']:
            product=r['product'];p=previous/product/'B11_full_residual.npy';assert sha(p)==oldhash[product+'/B11_full_residual.npy']
            raw=images.read('subscenes/'+product+'.npy');assert hashlib.sha256(raw).hexdigest()==r['member_sha256']['image']
            cube=np.load(io.BytesIO(raw),allow_pickle=False);residual=np.load(p,allow_pickle=False)
            # Recover exact original integer DN from already verified cube+residual,
            # then apply unchanged quantification=10000. No direction recomputation.
            native=np.rint((cube[:,:,11].astype('float64')-residual)*10000)/10000;dark=-native
            common=np.load(prior/product/'common_mask.npy',allow_pickle=False);ev=np.load(prior/product/'evaluation_mask.npy',allow_pickle=False)
            edges,ids,q,n=darkness_bins(dark,common);saved=json.loads((out/product/'darkness_edges.json').read_text())
            assert n==saved['actual_bins'];np.testing.assert_allclose(q,saved['original_linear_quantiles'],atol=1e-15,rtol=0)
            np.testing.assert_allclose(edges,saved['unique_edges'],atol=1e-15,rtol=0)
            # Serialized frozen boundaries are authoritative across NumPy versions.
            ids[common]=np.searchsorted(np.array(saved['unique_edges'])[1:-1],dark[common],side='right')
            raw=masks.read('masks/'+product+'.npy');assert hashlib.sha256(raw).hexdigest()==r['member_sha256']['mask']
            mask=np.load(io.BytesIO(raw),allow_pickle=False);truth=mask.argmax(-1);assert np.array_equal(ev,common&(mask.sum(-1)==1))
            for b in range(n):
                cov=next(x for x in coverage if x['product']==product and int(x['bin'])==b)
                assert int(cov['common_centers'])==int((ids==b).sum()) and int(cov['evaluation_centers'])==int(((ids==b)&ev).sum())
                for task in TASKS:
                    v=(ids==b)&ev
                    if task!=TASKS[0]:v &= (truth==2)|(truth==(0 if task==TASKS[1] else 1))
                    pos=int(((truth==2)&v).sum());neg=int(v.sum())-pos
                    matches=[x for x in bins if x['product']==product and x['task']==task and int(x['bin'])==b];assert len(matches)==6
                    for x in matches:
                        assert int(x['Shadow'])==pos and int(x['nonShadow'])==neg and int(x['pair_weight'])==pos*neg
                        assert (x['AUC']=='')==(pos*neg==0)
                    checked_bins+=1
    for x in scene:
        selected=[b for b in bins if b['product']==x['product'] and b['task']==x['task'] and b['score']==x['score']]
        assert len(selected)==x['total_bins']
        den=sum(int(b['pair_weight']) for b in selected);num=sum(float(b['AUC'])*int(b['pair_weight']) for b in selected if b['AUC']!='')
        assert den==x['pair_denominator']
        if den:assert abs(num/den-x['conditional_AUC'])<1e-14
        else:assert x['conditional_AUC'] is None
        assert sum(b['AUC']!='' for b in selected)==x['supported_bins']
        for label in ['Shadow','nonShadow']:
            total=sum(int(b[label]) for b in selected);supported=sum(int(b[label]) for b in selected if b['AUC']!='')
            assert total==x['total_'+label] and supported==x['supported_'+label]
            if total:assert abs(supported/total-x[label+'_coverage'])<1e-14
    for m in summary['macro']:
        other=m['comparison'].split('-',1)[1];delta=[]
        for r in lock['selected']:
            values={x['score']:x['conditional_AUC'] for x in scene if x['product']==r['product'] and x['task']==m['task']}
            delta.append((values['sun']-values[other])*100 if values['sun'] is not None and values[other] is not None else None)
        if all(v is not None for v in delta):assert abs(float(np.mean(delta))-m['mean_delta_AUC_pp'])<1e-12
        else:assert m['mean_delta_AUC_pp'] is None
        assert sum(v is not None and v>0 for v in delta)==m['positive_scenes']
    for r in table(out/'phase78_reproduction.csv'):
        original=next(x for x in old['metrics'] if x['product']==r['product'] and x['score']==r['score'])
        for key in ['AP','ROC_AUC']:assert abs(float(r[key])-original[key])<1e-12
    primary=[x for x in summary['macro'] if x['task']==TASKS[0]];supported=all(x['defined_scenes']==3 for x in primary)
    passed=supported and all(x['mean_delta_AUC_pp']>0 and x['positive_scenes']>=2 for x in primary if x['comparison']!='sun-all')
    assert supported==summary['support_sufficient'] and passed==summary['screen_passed'] and summary['phase78_failure_unchanged']
    result=dict(artifact_SHA_verified=len(hashes),Git_blobs_verified=7,previous_input_and_mask_SHA_verified=True,all_three_darkness_boundaries_reproduced_from_fixed_local_B11=True,
        per_bin_per_task_support_locally_verified=checked_bins,conditional_AUC_weighted_aggregates_locally_verified=len(scene),macro_differences_verified=len(summary['macro']),original_AP_ROC_reproduction_verified=18,stop_rule_verified=True,
        bin_AUC_pixels_computed_shared='AUC implementation validated against synthetic brute-force comparisons; actual 18 unstratified AUCs reproduced original sklearn; local reaggregation validates reported per-bin AUC weights, not re-reading shared directional score arrays')
    with (out/'LOCAL_VERIFIED.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()

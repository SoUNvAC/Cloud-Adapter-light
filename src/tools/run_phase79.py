"""Fixed 20-quantile, within-darkness-pair diagnostic; no fit or new scores."""
import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np
import run_phase76 as audit
from verify_phase78_local import independent_AP

NAMES=('sun','orthogonal90','reverse','orthogonal270','all','dark')
TASKS=('Shadow_vs_nonShadow','Shadow_vs_Surface','Shadow_vs_Cloud')


def auc(y,score):
    """Mann-Whitney probability, ties contribute .5; missing class => null."""
    y=np.asarray(y,dtype=bool);score=np.asarray(score)
    assert len(y)==len(score) and np.isfinite(score).all()
    pos=int(y.sum());neg=len(y)-pos
    if not pos or not neg:return None
    order=np.argsort(score,kind='stable');s=score[order];yy=y[order]
    starts=np.r_[0,np.flatnonzero(s[:-1]!=s[1:])+1]
    count=np.diff(np.r_[starts,len(y)]);p=np.add.reduceat(yy.astype('int64'),starts);n=count-p
    before=np.cumsum(n,dtype='int64')-n
    return float(np.sum(p*(before+.5*n))/(pos*neg))


def darkness_bins(dark,common):
    values=dark[common];assert len(values)>0 and np.isfinite(values).all()
    original=np.quantile(values,np.linspace(0,1,21),method='linear')
    edges=np.unique(original)
    ids=np.full(dark.shape,-1,dtype='int16')
    ids[common]=np.searchsorted(edges[1:-1],values,side='right')
    # Degenerate all-equal darkness is a single bin, not zero supported bins.
    count=max(1,len(edges)-1)
    assert (ids[common]>=0).all() and (ids[common]<count).all()
    return edges,ids,original,count


def conditional(truth,score,ids,count,task):
    selected=np.ones(truth.shape,dtype=bool) if task==TASKS[0] else ((truth==2)|(truth==(0 if task==TASKS[1] else 1)))
    y=truth==2;table=[];num=0.;den=0;sp=sn=0
    for b in range(count):
        v=selected&(ids==b);p=int(y[v].sum());n=int(v.sum())-p;weight=p*n;a=auc(y[v],score[v])
        table.append(dict(bin=b,Shadow=p,nonShadow=n,pair_weight=weight,AUC=a))
        if a is not None:num+=a*weight;den+=weight;sp+=p;sn+=n
    totalp=int((selected&y).sum());totaln=int((selected&~y).sum())
    return table,dict(conditional_AUC=num/den if den else None,unstratified_AUC=auc(y[selected],score[selected]),
        supported_bins=sum(r['AUC'] is not None for r in table),total_bins=count,pair_denominator=den,
        supported_Shadow=sp,total_Shadow=totalp,Shadow_coverage=sp/totalp if totalp else None,
        supported_nonShadow=sn,total_nonShadow=totaln,nonShadow_coverage=sn/totaln if totaln else None)


def differences(rows):
    paired=[];macro=[]
    for task in TASKS:
        products=list(dict.fromkeys(r['product'] for r in rows if r['task']==task))
        assert len(products)==3
        for product in products:
            values={r['score']:r['conditional_AUC'] for r in rows if r['product']==product and r['task']==task}
            for name in NAMES[1:]:
                delta=(values['sun']-values[name])*100 if values['sun'] is not None and values[name] is not None else None
                paired.append(dict(product=product,tile=product.split('_')[5],task=task,comparison='sun-'+name,delta_AUC_pp=delta))
        for name in NAMES[1:]:
            vals=[r['delta_AUC_pp'] for r in paired if r['task']==task and r['comparison']=='sun-'+name]
            macro.append(dict(task=task,comparison='sun-'+name,mean_delta_AUC_pp=float(np.mean(vals)) if all(v is not None for v in vals) else None,positive_scenes=sum(v is not None and v>0 for v in vals),defined_scenes=sum(v is not None for v in vals),scene_count=3))
    return paired,macro


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--bindings',type=Path,required=True);a=ap.parse_args()
    for p in (a.data,a.previous,a.output,a.bindings):assert any(p.resolve().is_relative_to(Path(x)) for x in ('/home/scv/shared','/home/scv/Cloud-Adapter-light'))
    a.output.mkdir(exist_ok=False);out=a.output
    try:
        expected=json.loads(a.bindings.read_text())
        for rel,h in expected['phase78'].items():assert audit.sha(a.previous/rel)==h,rel
        hashes=json.loads((a.previous/'artifact_hashes.json').read_text())
        for rel,h in hashes.items():
            p=a.previous/rel;assert p.resolve().is_relative_to(a.previous.resolve()) and audit.sha(p)==h,rel
        oldlock=json.loads((a.previous/'input_formula_lock.json').read_text());old=json.loads((a.previous/'summary.json').read_text())
        assert old['status']=='completed' and old['screen_passed'] is False and len(oldlock['selected'])==3
        for rel,key in [('src/tools/run_phase78.py','code_sha256'),('src/tools/run_phase76.py','helper_sha256'),('src/research_plans/PHASE78_INPUT_BINDINGS_20261011.json','bindings_sha256')]:
            assert hashlib.sha256(subprocess.check_output(['git','show',oldlock['git_commit']+':'+rel])).hexdigest()==oldlock[key]
        lock=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            code_sha256=audit.sha(Path(__file__)),bindings_sha256=audit.sha(a.bindings),audit_helper_sha256=audit.sha(Path(audit.__file__)),AP_helper_sha256=audit.sha(Path('src/tools/verify_phase78_local.py')),
            test_sha256=audit.sha(Path('src/tools/test_phase79.py')),runner_sha256=audit.sha(Path('src/tools/run_phase79.sh')),protocol_sha256=audit.sha(Path('src/research_plans/PHASE79_CONDITIONAL_DIRECTION_20261011.md')),
            previous_inputs=expected['phase78'],previous_artifact_hashes=hashes,selected=oldlock['selected'],
            quantiles='np.quantile(dark[common_mask],np.linspace(0,1,21),method=linear); unique edges; no label-dependent boundaries',
            assignment='searchsorted(unique_edges[1:-1],dark,side=right); [left,right) except final includes max; all-equal => one bin; outside common => -1',
            conditional_AUC='sum(AUC_bin*n_Shadow*n_negative)/sum(n_Shadow*n_negative) over defined bins; only same-bin pairs; ties=.5; six scores share bins, valid mask and weights',
            tasks=TASKS,scores=NAMES,scope='exact Phase78 three fit scenes/six scores; no new directions, imagery, fitting, models or confirmation; Phase78 remains failed',
            continue_rule='primary sun minus reverse/orthogonal90/orthogonal270/dark macro conditional AUC each >0 and each positive >=2/3; any undefined scene => insufficient support; all fully reported, no substitution',
            no_significance_tests=True,budget_cutoff='2026-10-20 Asia/Shanghai',numpy_version=np.__version__,new_conditional_analysis_not_yet_computed=True)
        audit.write(out/'input_code_lock.json',lock)
        scenes=[];bin_audit=[]
        # Freeze all three label-independent partitions before any new label read.
        for r in oldlock['selected']:
            product=r['product'];dest=out/product;dest.mkdir()
            with np.load(a.previous/product/'native_scores.npz',allow_pickle=False) as f:
                assert set(f.files)==set(NAMES);scores={n:f[n] for n in NAMES}
            common=np.load(a.previous/product/'common_mask.npy',allow_pickle=False);ev=np.load(a.previous/product/'evaluation_mask.npy',allow_pickle=False)
            assert common.dtype==np.bool_ and ev.dtype==np.bool_ and common.shape==(1022,1022) and not (ev&~common).any()
            assert all(np.isfinite(scores[n][common]).all() for n in NAMES)
            edges,ids,original,count=darkness_bins(scores['dark'],common)
            edge_lock=dict(product=product,requested_bins=20,actual_bins=count,original_linear_quantiles=original.tolist(),unique_edges=edges.tolist(),common_count=int(common.sum()),common_sha256=hashes[product+'/common_mask.npy'],labels_not_yet_read=True)
            audit.write(dest/'darkness_edges.json',edge_lock)
            for b in range(count):bin_audit.append(dict(product=product,tile=product.split('_')[5],bin=b,lower=float(edges[min(b,len(edges)-1)]),upper=float(edges[min(b+1,len(edges)-1)]),upper_inclusive=b==count-1,common_centers=int((ids==b).sum()),evaluation_centers=int(((ids==b)&ev).sum())))
            scenes.append((r,scores,common,ev,ids,count))
        audit.csvwrite(out/'bin_coverage.csv',bin_audit);print('ALL THREE DARKNESS PARTITIONS FROZEN WITHOUT LABELS',flush=True)
        reproduction=[];prepared=[]
        with zipfile.ZipFile(a.data/'masks.zip') as masks:
            for r,scores,common,ev,ids,count in scenes:
                product=r['product'];raw=masks.read('masks/'+product+'.npy');assert hashlib.sha256(raw).hexdigest()==r['member_sha256']['mask']
                mask=np.load(io.BytesIO(raw),allow_pickle=False);assert mask.shape==(1022,1022,3) and mask.dtype==np.bool_
                assert np.array_equal(ev,common&(mask.sum(-1)==1));truth=mask.argmax(-1)[ev]
                for name in NAMES:
                    original=next(m for m in old['metrics'] if m['product']==product and m['score']==name)
                    ap=independent_AP(truth==2,scores[name][ev]);roc=auc(truth==2,scores[name][ev]);assert abs(ap-original['AP'])<1e-12 and abs(roc-original['ROC_AUC'])<1e-12
                    reproduction.append(dict(product=product,score=name,AP=ap,original_AP=original['AP'],ROC_AUC=roc,original_ROC_AUC=original['ROC_AUC']))
                prepared.append((r,{n:scores[n][ev] for n in NAMES},truth,ids[ev],count))
        audit.csvwrite(out/'phase78_reproduction.csv',reproduction);print('ALL 18 ORIGINAL AP AND ROC AUC REPRODUCED',flush=True)
        scene_rows=[];bin_rows=[];support=[]
        for r,scores,truth,ids,count in prepared:
            product=r['product'];tile=product.split('_')[5]
            for task in TASKS:
                shared=None
                for name in NAMES:
                    table,m=conditional(truth,scores[name],ids,count,task)
                    weights=[(x['Shadow'],x['nonShadow'],x['pair_weight']) for x in table]
                    if shared is None:shared=weights
                    else:assert shared==weights
                    scene_rows.append(dict(product=product,tile=tile,task=task,score=name,**m))
                    for x in table:bin_rows.append(dict(product=product,tile=tile,task=task,score=name,**x))
                support.append(dict(product=product,tile=tile,task=task,**{k:v for k,v in m.items() if k not in ('conditional_AUC','unstratified_AUC')}))
            print('SCENE COMPLETE',tile,[(x['score'],x['conditional_AUC']) for x in scene_rows if x['product']==product and x['task']==TASKS[0]],flush=True)
        paired,macro=differences(scene_rows);primary=[x for x in macro if x['task']==TASKS[0]]
        sufficient=all(x['defined_scenes']==3 for x in primary)
        passed=sufficient and all(x['mean_delta_AUC_pp']>0 and x['positive_scenes']>=2 for x in primary if x['comparison']!='sun-all')
        action='retain_candidate_hypothesis_only_requires_new_protocol' if passed else ('insufficient_support_no_resampling' if not sufficient else 'close_this_direction_increment_hypothesis')
        for name,rows in [('scene_conditional_AUC.csv',scene_rows),('bin_AUC.csv',bin_rows),('support_audit.csv',support),('paired_differences.csv',paired),('macro_differences.csv',macro)]:audit.csvwrite(out/name,rows)
        audit.write(out/'summary.json',dict(status='completed',support_sufficient=sufficient,screen_passed=passed,next_action=action,phase78_failure_unchanged=True,scenes=scene_rows,macro=macro,support=support,new_download_bytes=0,no_models_or_confirmation_access=True))
        audit.write(out/'artifact_hashes.json',{str(p.relative_to(out)):audit.sha(p) for p in out.rglob('*') if p.is_file()})
    except Exception as e:
        audit.write(out/'failure.json',dict(type=type(e).__name__,error=str(e)));raise


if __name__=='__main__':main()

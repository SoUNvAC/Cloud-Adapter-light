"""Frozen 65b exploratory NIR/SWIR risk evidence, not six-band segmentation."""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from prepare_phase65a_explore import convert,digest,SPLIT_SHA,SOURCE_SHA
from summarize_phase65a_explore import fixed_logistic,pairwise_auroc

ADAPTED_SHA='82f88d235948ef0a9202cebaa026e7f9fb1e1ad5c0aa361d9045c3e6991ee16d'
FIT_SOURCE_SHA='f0704e969a2f7e88bde0c0790d594b4e92cebcd7e906160fe8affd9d97abe309'
FIT_ADAPTED_SHA='362f2e219518d80e87084b7d756618b490fc84c2629dd6593161818675c99f80'


def authorized(path):
    path=Path(path).resolve()
    if not any(path.is_relative_to(Path(r)) for r in ['/home/scv/shared','/home/scv/Cloud-Adapter-light']):
        raise ValueError('Outside authorized roots')
    return path


def prepare(a):
    root=authorized(a.root);out=authorized(a.output);split=authorized(a.split)
    assert digest(split)==SPLIT_SHA
    d=json.loads(split.read_text());assert d['status']=='stopped_insufficient_h1_support'
    assert json.loads((root/'download_status.json').read_text())['status']=='all_files_verified'
    selected=[r for r in d['proposed_rows'] if r['split']=='65b_confirmation'];assert len(selected)==64
    assert digest(authorized(a.adapted_checkpoint))==ADAPTED_SHA
    out.mkdir(exist_ok=False);(out/'arrays').mkdir();rows=[]
    tags={r['scene']:r for r in csv.DictReader((root/'classification_tags.csv').open())}
    with zipfile.ZipFile(root/'subscenes.zip') as images,zipfile.ZipFile(root/'masks.zip') as masks:
        for group in selected:
            product=group['representative'];assert tags[product]['shadows_marked']=='1'
            rb=images.read(f'subscenes/{product}.npy');mb=masks.read(f'masks/{product}.npy')
            raw=np.load(io.BytesIO(rb),allow_pickle=False);mask=np.load(io.BytesIO(mb),allow_pickle=False)
            rgb,target,valid,spectra=convert(raw,mask)
            if not (target!=255).any():raise ValueError('Empty valid labels')
            prefix=f'arrays/{product}'
            for suffix,array in [('_rgb.npy',rgb),('_mask.npy',target),('_image_valid.npy',valid)]:
                np.save(out/(prefix+suffix),array)
            rows.append(dict(product=product,group_id=group['group_id'],original_split=group['split'],
                split='phase65b_evidence',is_representative=True,h1_public_support=group['h1_supported'],
                rgb_path=prefix+'_rgb.npy',mask_path=prefix+'_mask.npy',image_valid_path=prefix+'_image_valid.npy',
                member_sha256={'image':hashlib.sha256(rb).hexdigest(),'mask':hashlib.sha256(mb).hexdigest()},
                prepared_sha256={s:digest(out/(prefix+s)) for s in ['_rgb.npy','_mask.npy','_image_valid.npy']},
                observable_spectra=spectra,valid_pixels=int((target!=255).sum()),
                label_counts=np.bincount(target[target!=255],minlength=3).tolist(),
                image_valid_sha256=hashlib.sha256(valid.tobytes()).hexdigest(),
                matched_grid_shape=list(raw.shape[:2]),radiometry='official float32 TOA, no clipping',
                geometric_matching='RGB and NIR/SWIR from same official cube; external CRS/transform not available in npy'))
            print('prepared65b',len(rows),product,flush=True)
    report=dict(scope='exploratory_phase65b_evidence_only',authorization='user_20261008_advance65b',
                source_sha256=SOURCE_SHA,adapted_sha256=ADAPTED_SHA,original_split_sha256=SPLIT_SHA,
                original_65a_confirmation_reads=False,phase65c_reads=False,sealed_test_reads=False,rows=rows)
    (out/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))


def pair_rows(source,adapted,cohort):
    left=[r for r in source['scenes'] if r['cohort']==cohort]
    right={r['product']:r for r in adapted['scenes'] if r['cohort']==cohort}
    if set(right)!={r['product'] for r in left}:raise ValueError('Paired products mismatch')
    rows=[]
    for s in left:
        t=right[s['product']];sc=np.asarray(s['confusion']);tc=np.asarray(t['confusion'])
        if s['group_id']!=t['group_id'] or not (sc.sum(1)==tc.sum(1)).all():raise ValueError('Pair denominator mismatch')
        rows.append(dict(product=s['product'],group_id=s['group_id'],features=s['source_observable_features'],
            risk=bool(tc[2,0]>sc[2,0]),shadow_to_surface_delta=int(tc[2,0]-sc[2,0]),
            delta_miou_pp=t['miou_percent']-s['miou_percent']))
    if len({r['group_id'] for r in rows})!=len(rows):raise ValueError('Repeated statistical group')
    return rows


def paired_inference(y,full,rgb,iterations=10000):
    """Paired stratified scene bootstrap and paired-score permutation; fixed seeds."""
    positive=np.flatnonzero(y);negative=np.flatnonzero(~y)
    rng=np.random.default_rng(65);deltas=[]
    for _ in range(iterations):
        index=np.r_[rng.choice(positive,len(positive)),rng.choice(negative,len(negative))]
        deltas.append(pairwise_auroc(y[index],full[index])-pairwise_auroc(y[index],rgb[index]))
    observed=pairwise_auroc(y,full)-pairwise_auroc(y,rgb)
    rng=np.random.default_rng(65);count=0
    for _ in range(iterations):
        swap=rng.random(len(y))<.5
        a=np.where(swap,rgb,full);b=np.where(swap,full,rgb)
        count+=pairwise_auroc(y,a)-pairwise_auroc(y,b)>=observed
    p=(count+1)/(iterations+1)
    return dict(delta_auroc=observed,paired_stratified_ci95=np.quantile(deltas,[.025,.975]).tolist(),
                one_sided_paired_score_permutation_p=p,holm_family3_adjusted_p=min(1.,3*p),iterations=iterations)


def probe(a):
    out=authorized(a.output)
    if out.exists():raise FileExistsError('Preserve prior evidence report')
    fs=authorized(a.fit_source);fa=authorized(a.fit_adapted)
    assert digest(fs)==FIT_SOURCE_SHA and digest(fa)==FIT_ADAPTED_SHA
    es=authorized(a.evidence_source);ea=authorized(a.evidence_adapted)
    source=json.loads(es.read_text());adapted=json.loads(ea.read_text())
    assert source['scope']==adapted['scope']=='exploratory_phase65b_evidence_only'
    assert source['checkpoint_sha256']==SOURCE_SHA and adapted['checkpoint_sha256']==ADAPTED_SHA
    assert source['prepared_manifest_sha256']==adapted['prepared_manifest_sha256']
    fit=pair_rows(json.loads(fs.read_text()),json.loads(fa.read_text()),'fit_representatives')
    held=pair_rows(source,adapted,'phase65b_evidence');assert len(fit)==153 and len(held)==64
    assert not {r['group_id'] for r in fit}&{r['group_id'] for r in held}
    x=np.asarray([r['features'] for r in fit]);z=np.asarray([r['features'] for r in held])
    y=np.asarray([r['risk'] for r in fit]);yh=np.asarray([r['risk'] for r in held])
    assert x.shape==(153,12) and z.shape==(64,12) and np.isfinite(x).all() and np.isfinite(z).all()
    report=dict(status='complete_exploratory_evidence_probe',primary_confirmation_test=False,phase65c_authorized=False,
        fit_groups=153,heldout_groups=64,fit_positive=int(y.sum()),fit_negative=int((~y).sum()),
        heldout_positive=int(yh.sum()),heldout_negative=int((~yh).sum()),
        definition='Risk truth: adapted minus source Shadow-to-Surface count > 0; GT used only for offline target',
        feature_groups={'rgb':list(range(9)),'rgb_nir_swir':list(range(12))},
        heldout_rows=held,source_aggregate=source['aggregate'],adapted_aggregate=adapted['aggregate'],
        matched_input='Same 13-band cube, native grid, TOA, joint six-band observation mask and checkpoints; fit-only standardization',
        formal_input_gates_pending=['external CRS/footprint verification','independent hard-case label review',
            'matched RGB/six segmentation tiny-fit>=95','matched six-band source retention<=1pp'],
        skipped_evidence={'solar_geometry':'No verified solar/cloud-height inputs','large_context':'No separately frozen context probe'},
        interpretation='Extra spectral risk features only; not a six-band segmentation result, not prior 65a gate passing',
        artifacts={str(p):digest(p) for p in [fs,fa,es,ea]})
    if min(y.sum(),(~y).sum())<8:
        report['risk_probe_status']='fit_support_insufficient_not_fitted'
    else:
        rgb,rd=fixed_logistic(x[:,:9],y,z[:,:9]);full,fd=fixed_logistic(x,y,z)
        shuffled=z.copy();shuffled[:,9:]=z[np.random.default_rng(65).permutation(len(z)),9:]
        details=fd;mean=np.asarray(details['fit_mean']);scale=np.asarray(details['fit_scale']);theta=np.asarray(details['coefficients'])
        from scipy.special import expit
        shuffle=expit(((shuffled-mean)/scale)@theta[:-1]+theta[-1])
        report['frozen_models']={'rgb':rd,'rgb_nir_swir':fd}
        report['probabilities']={'rgb':rgb.tolist(),'rgb_nir_swir':full.tolist(),'shuffled_nir_swir':shuffle.tolist()}
        if min(yh.sum(),(~yh).sum())<8:
            report['risk_probe_status']='heldout_support_insufficient_no_auroc_claim'
        else:
            report['risk_probe_status']='heldout_support_sufficient_exploratory_only'
            report['auroc']={n:pairwise_auroc(yh,p) for n,p in [('rgb',rgb),('rgb_nir_swir',full),('shuffled_nir_swir',shuffle)]}
            report['spectral_increment']=paired_inference(yh,full,rgb)
            evidence=report['spectral_increment']
            report['spectral_increment_exploratory_support']=evidence['paired_stratified_ci95'][0]>0 and evidence['holm_family3_adjusted_p']<.05
    out.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ['heldout_rows','probabilities','frozen_models','artifacts']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for name in ['root','split','adapted-checkpoint','output']:q.add_argument('--'+name,required=True)
    q=sub.add_parser('probe')
    for name in ['fit-source','fit-adapted','evidence-source','evidence-adapted','output']:q.add_argument('--'+name,required=True)
    a=p.parse_args();prepare(a) if a.command=='prepare' else probe(a)

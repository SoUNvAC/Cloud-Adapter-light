"""One frozen Phase66 evaluation. No fitting, threshold selection or cohort expansion."""
import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from freeze_phase66 import digest, code_digest, write, SPLIT_SHA, RECOVERY_SHA
from prepare_phase65a_explore import convert
from phase65d_metrics import curve, confusion, measures, corrected
from phase65d_recovery_math import logit_score, predict_logit
from phase66_statistics import interval, primary, fpr_constraint

REPO = Path('/home/scv/Cloud-Adapter-light')
DATA = Path('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871')
ROOT = DATA/'phase66_20261010'
PREP = DATA/'phase65a_explore_20261008'
RECOVERY = DATA/'phase65d_recovery_20261008'
CHECKPOINTS = {
    'source': REPO/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth',
    'msre': REPO/'src/work_dirs/phase65a_explore_20261008/msre_seed65/best_mIoU_iter_2000.pth'}
LEVELS = {'L1': ('msre_calibration', 1), 'L2': ('msre_source', 2),
          'L3': ('msre_source_spectral', 5)}
NAMES = ['Source', 'MsRE', 'L1', 'L2', 'L3']


def read(path):
    return json.loads(Path(path).read_text())


def lock(bundle):
    frozen = read(bundle/'phase66_lock.json')
    assert digest(bundle/'recovery_lock.json') == RECOVERY_SHA
    assert digest(REPO/'src/work_dirs/phase65a/split_lock.json') == SPLIT_SHA
    for rel, sha in frozen['code_sha256'].items():
        assert code_digest(REPO/rel) == sha, ('code/config drift', rel)
    assert digest(RECOVERY/'recovery_lock.json') == RECOVERY_SHA
    for rel, sha in frozen['fit_input_sha256'].items():
        assert digest(RECOVERY/rel) == sha
    assert digest(PREP/'manifest.json') == frozen['prepared_fit_development_manifest_sha256']
    return frozen


def prepare(bundle, groups_path):
    frozen = lock(bundle)
    assert not ROOT.exists(), 'Preserve any existing Phase66 attempt; no automatic rerun'
    groups = read(groups_path)
    assert digest(groups_path) == read(REPO/'src/work_dirs/phase65a/split_lock.json')['group_sha256']
    assert groups_path.resolve().is_relative_to(Path('/home/scv'))
    semantic = hashlib.sha256(json.dumps(groups['groups'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert semantic == frozen['group_semantics_sha256']
    assert sorted(groups['input_sha256'].values()) == frozen['provenance_input_sha256']
    provenance = {}
    for path, sha in groups['input_sha256'].items():
        path = Path(path)
        assert any(path.resolve().is_relative_to(r) for r in [REPO, Path('/home/scv/shared')])
        assert digest(path) == sha, ('provenance dependency changed', str(path))
        provenance[str(path)] = sha
    eligible = {g['group_id']: g for g in groups['groups'] if g['eligible']}
    selected = frozen['rows']
    for r in selected:
        assert set(r['products']) == set(eligible[r['group_id']]['shadow_valid_products'])
    products = {r['representative'] for r in selected}
    # Only manifests/index metadata, never old sealed pixels or scores.
    access_metadata = {}
    for folder in sorted(DATA.glob('phase65*')):
        if not folder.is_dir():
            continue
        paths = set(folder.rglob('manifest.json')) | set(folder.rglob('index*.json'))
        for path in sorted(paths):
            d = read(path)
            rows = d.get('rows', [])
            if isinstance(rows, list):
                prior = {r.get('product') for r in rows if isinstance(r, dict)}
                assert not products & prior, ('Prior confirmation access; stop', str(path))
            access_metadata[str(path)] = digest(path)
    status = read(DATA/'download_status.json')
    assert status['status'] == 'all_files_verified'
    for name, reference in frozen['official_files'].items():
        assert status['verified_files'][name]['sha256'] == reference['sha256']
        assert (DATA/name).stat().st_size == reference['bytes']
    for name, path in CHECKPOINTS.items():
        assert digest(path) == frozen[name+'_checkpoint_sha256']
    # Receipt precedes the first read of a selected ZIP member.
    ROOT.mkdir(); (ROOT/'candidate').mkdir(); (ROOT/'prepared').mkdir(); (ROOT/'delivery').mkdir()
    for name in ['phase66_lock.json', 'recovery_lock.json']:
        shutil.copyfile(bundle/name, ROOT/'candidate'/name)
    for name, path in CHECKPOINTS.items():
        target = ROOT/'candidate'/(name+'.pth')
        shutil.copyfile(path, target)
        assert digest(target) == frozen[name+'_checkpoint_sha256']
    from importlib.metadata import version
    write(ROOT/'unseal_receipt.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        phase66_lock_sha256=digest(bundle/'phase66_lock.json'), provenance=provenance,
        package_versions={n:version(n) for n in ['numpy','torch','mmengine','mmsegmentation','mmcv']},
        access_metadata_sha256=access_metadata, old_h1_unchanged=True,
        authorized_cohort='65a_confirmation_64_original_representatives',
        other_reserved_cohorts_read=False))
    tags = {r['scene']: r for r in csv.DictReader((DATA/'classification_tags.csv').open())}
    rows = []; prepared = ROOT/'prepared'; (prepared/'arrays').mkdir()
    with zipfile.ZipFile(DATA/'subscenes.zip') as images, zipfile.ZipFile(DATA/'masks.zip') as masks:
        for group in selected:
            product = group['representative']; assert tags[product]['shadows_marked'] == '1'
            rb = images.read('subscenes/'+product+'.npy'); mb = masks.read('masks/'+product+'.npy')
            raw = np.load(io.BytesIO(rb), allow_pickle=False)
            rgb, truth, valid, _ = convert(raw, np.load(io.BytesIO(mb), allow_pickle=False))
            assert (truth != 255).any(), 'Empty valid representative: stop without replacement'
            actual_shadow = bool((truth == 2).any())
            # Record discrepancies, do not replace, resplit or hide zero-support groups.
            prefix = 'arrays/'+product
            arrays = {'_rgb.npy': rgb, '_mask.npy': truth,
                      '_spectral.npy': raw[...,[7,11,12]].copy()}
            for suffix, array in arrays.items():
                np.save(prepared/(prefix+suffix), array, allow_pickle=False)
            rows.append(dict(product=product, group_id=group['group_id'], original_split='65a_confirmation',
                rgb_path=prefix+'_rgb.npy', mask_path=prefix+'_mask.npy', spectral_path=prefix+'_spectral.npy',
                label_counts=np.bincount(truth[truth!=255], minlength=3).tolist(),
                public_support=group['h1_supported'], actual_shadow_support=actual_shadow,
                prepared_sha256={suffix: digest(prepared/(prefix+suffix)) for suffix in arrays},
                member_sha256=dict(image=hashlib.sha256(rb).hexdigest(), mask=hashlib.sha256(mb).hexdigest())))
            print('prepared Phase66',len(rows),64,product,flush=True)
    write(prepared/'manifest.json', dict(scope='phase66_once_independent_64',
        phase66_lock_sha256=digest(bundle/'phase66_lock.json'), rows=rows))


def extract(name):
    frozen = lock(ROOT/'candidate'); prepared = ROOT/'prepared'; md = read(prepared/'manifest.json')
    assert len(md['rows']) == 64 and md['scope'] == 'phase66_once_independent_64'
    out = ROOT/('scores_'+name); out.mkdir(exist_ok=False)
    import torch
    sys.path.insert(0, str(REPO/'src'))
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmseg.utils import register_all_modules
    register_all_modules()
    import cloud_adapter.models, cloud_adapter.datasets.phase65_catalogue
    from mmseg.registry import MODELS
    from mmseg.datasets import BaseSegDataset

    class IndependentDataset(BaseSegDataset):
        METAINFO = dict(classes=('surface_visible','cloud','shadow'),
                        palette=[[80,130,70],[245,245,245],[70,70,70]])

        def load_data_list(self):
            return [dict(img_path=str(prepared/r['rgb_path']), seg_map_path=str(prepared/r['mask_path']),
                         label_map=None,reduce_zero_label=False,seg_fields=[]) for r in md['rows']]

    ck = ROOT/'candidate'/(name+'.pth')
    assert digest(ck) == frozen[name+'_checkpoint_sha256']
    os.environ['PHASE65_PREPARED_MANIFEST'] = str(PREP/'manifest.json')
    os.environ['PHASE65_SOURCE_CHECKPOINT'] = str(ROOT/'candidate/source.pth')
    torch.manual_seed(65); np.random.seed(65)
    cfg = Config.fromfile(str(REPO/'src/configs/protocol/phase65a_explore_rgb.py'))
    model = MODELS.build(cfg.model); model.init_weights()
    missing, unexpected = model.load_state_dict(torch.load(ck,map_location='cpu')['state_dict'],strict=False)
    assert not unexpected and all(k.startswith(('backbone.target_msre.','backbone.target_head_delta.')) for k in missing)
    if name == 'msre': assert not missing
    model.train(); model.backbone.set_target_enabled(name=='msre'); model.eval().cuda()
    ds = IndependentDataset(pipeline=cfg.test_pipeline, reduce_zero_label=False)
    rows = []
    for i, r in enumerate(md['rows']):
        prefix = r['rgb_path'].removesuffix('_rgb.npy')
        for suffix, sha in r['prepared_sha256'].items():
            assert digest(prepared/(prefix+suffix)) == sha
        sample = ds[i]
        assert Path(sample['data_samples'].img_path).name == r['product']+'_rgb.npy'
        with torch.no_grad(): p = model.test_step(pseudo_collate([sample]))[0]
        native = p.seg_logits.data.cpu().numpy()
        arrays = dict(score=p.seg_logits.data.softmax(0).cpu().numpy()[2].astype(np.float32),
                      pred=p.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8),
                      nonshadow=native[:2].argmax(0).astype(np.uint8))
        assert np.isfinite(arrays['score']).all()
        assert arrays['score'].shape == np.load(prepared/r['mask_path']).shape
        files = {}
        for key, array in arrays.items():
            path = out/(r['product']+'_'+key+'.npy'); np.save(path,array,allow_pickle=False)
            files[key] = dict(path=str(path.relative_to(ROOT)),sha256=digest(path))
        rows.append(dict(product=r['product'],group_id=r['group_id'],files=files))
        print('Phase66 scores',name,i+1,64,r['product'],flush=True)
    write(out/'index.json',dict(checkpoint_sha256=digest(ck),manifest_sha256=digest(prepared/'manifest.json'),rows=rows))


def load_score(row,key):
    f=row['files'][key];path=ROOT/f['path'];assert digest(path)==f['sha256']
    return np.load(path,allow_pickle=False)


def brief(c):
    return {k:v for k,v in c.items() if k not in ['precision','recall','fpr','thresholds']}


def evaluate():
    frozen = lock(ROOT/'candidate'); recovery = read(ROOT/'candidate/recovery_lock.json')
    md = read(ROOT/'prepared/manifest.json'); indices=[]
    for name in ['source','msre']:
        index=read(ROOT/('scores_'+name)/'index.json')
        assert index['checkpoint_sha256']==frozen[name+'_checkpoint_sha256']
        assert index['manifest_sha256']==digest(ROOT/'prepared/manifest.json')
        rows={r['product']:r for r in index['rows']};assert set(rows)=={r['product'] for r in md['rows']}
        indices.append(rows)
    policy_rows={n:[] for n in NAMES};pool_scores={n:[] for n in NAMES};pool_labels=[]
    for r in md['rows']:
        product=r['product'];truth=np.load(ROOT/'prepared'/r['mask_path']);valid=truth!=255;yy=truth[valid]==2
        prefix=r['rgb_path'].removesuffix('_rgb.npy')
        for suffix,sha in r['prepared_sha256'].items():assert digest(ROOT/'prepared'/(prefix+suffix))==sha
        ss=load_score(indices[0][product],'score')[valid];ms=load_score(indices[1][product],'score')[valid]
        spectra=np.load(ROOT/'prepared'/r['spectral_path'])[valid]
        x=np.column_stack([logit_score(ms),logit_score(ss),spectra]).astype(np.float32)
        assert np.isfinite(x).all()
        scores={'Source':ss,'MsRE':ms}
        preds={'Source':load_score(indices[0][product],'pred')[valid],
               'MsRE':load_score(indices[1][product],'pred')[valid]}
        ns=load_score(indices[1][product],'nonshadow')[valid]
        for name,(key,n) in LEVELS.items():
            model=recovery['models'][key];scores[name]=predict_logit(x[:,:n],model)
            preds[name]=corrected(scores[name],model['threshold'],ns)
        for name in NAMES:
            metrics=measures(confusion(truth[valid],preds[name]));rank=brief(curve(scores[name],yy,fprs=(.01,.05)))
            policy_rows[name].append(dict(product=product,group_id=r['group_id'],ranking=rank,**metrics,
                predicted_shadow_fraction=float((preds[name]==2).mean()),
                shadow_to_surface_fraction=metrics['shadow_to_surface']/metrics['positive'] if metrics['positive'] else None,
                shadow_to_cloud_fraction=metrics['shadow_to_cloud']/metrics['positive'] if metrics['positive'] else None))
            pool_scores[name].append(scores[name])
        pool_labels.append(yy);print('Phase66 metrics',len(pool_labels),64,product,flush=True)
    policies={};labels=np.concatenate(pool_labels)
    for name in NAMES:
        rows=policy_rows[name];no_shadow=[r for r in rows if not r['positive']]
        pooled=measures(np.sum([r['confusion'] for r in rows],axis=0))
        pooled['ranking']=brief(curve(np.concatenate(pool_scores.pop(name)),labels,fprs=(.01,.05)))
        macros={key:interval([r[key] for r in rows if r[key] is not None]) for key in ['recall','precision','fpr']}
        policies[name]=dict(rows=rows,n_groups=len(rows),pooled=pooled,macro=macros,
            macro_ap=interval([r['ranking']['ap'] for r in rows if r['ranking']['ap'] is not None]),
            macro_miou_percent=interval([r['miou_percent'] for r in rows]),
            macro_iou_percent=[interval([r['iou_percent'][j] for r in rows if r['iou_percent'][j] is not None]) for j in range(3)],
            fpr_constraint=fpr_constraint([r['fpr'] for r in rows if r['fpr'] is not None]),
            no_shadow_groups=len(no_shadow),no_shadow_fpr=interval([r['fpr'] for r in no_shadow if r['fpr'] is not None]),
            undefined_precision_groups=sum(r['precision'] is None for r in rows))
    comparisons={}
    for before,after in [('L2','L3'),('L1','L2'),('Source','L3'),('MsRE','L3')]:
        pairs=[]
        for a,b in zip(policy_rows[before],policy_rows[after]):
            assert a['group_id']==b['group_id']
            if a['ranking']['ap'] is not None:
                assert b['ranking']['ap'] is not None
                pairs.append(dict(product=a['product'],group_id=a['group_id'],delta_ap=b['ranking']['ap']-a['ranking']['ap'],
                                  delta_recall=b['recall']-a['recall']))
        ap_summary=primary(pairs,frozen['minimum_ap_groups'])
        if (before,after)!=('L2','L3'):
            ap_summary.pop('decision')
        comparisons[after+'_minus_'+before]=dict(pairs=pairs,
            ap=ap_summary,
            max_recall_decline=min(pairs,key=lambda r:r['delta_recall']) if pairs else None,
            role='primary' if (before,after)==('L2','L3') else 'auxiliary_not_alternative_primary')
    apdecision=comparisons['L3_minus_L2']['ap']['decision']
    fpr=policies['L3']['fpr_constraint']
    report=dict(status='completed_once_independent_phase66',policies=policies,comparisons=comparisons,
        primary_decision=apdecision,
        joint_increment_and_budget_supported=apdecision=='positive_increment_independently_supported' and fpr['upper_bound_meets_budget'] is True,
        public_shadow_support=sum(r['public_support'] for r in md['rows']),
        actual_shadow_support=sum(r['actual_shadow_support'] for r in md['rows']),
        public_actual_support_disagreements=[r['product'] for r in md['rows'] if r['public_support']!=r['actual_shadow_support']],
        small_sample_limitation='Expected only 15 Shadow-supported groups; conditional bootstrap is not a power guarantee or label-quality certification.',
        phase66_lock_sha256=digest(ROOT/'candidate/phase66_lock.json'),
        prepared_manifest_sha256=digest(ROOT/'prepared/manifest.json'),
        score_index_sha256={n:digest(ROOT/('scores_'+n)/'index.json') for n in ['source','msre']},
        old_h1_failure_unchanged=True,phase65c_authorized=False)
    write(ROOT/'delivery/phase66_report.json',report)
    flat=[]
    for name in NAMES:
        for r in policy_rows[name]:
            flat.append(dict(policy=name,product=r['product'],group_id=r['group_id'],ap=r['ranking']['ap'],
                **{k:r[k] for k in ['positive','negative','recall','precision','fpr','miou_percent','shadow_to_surface','shadow_to_cloud','predicted_shadow_fraction']},
                **{str(j)+'_iou_percent':r['iou_percent'][j] for j in range(3)},
                **{'oracle_recall_fpr_'+str(rate):r['ranking']['matched_fpr'].get(str(rate),{}).get('recall') for rate in [.01,.05]}))
    with (ROOT/'delivery/groups.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print('Phase66 complete',apdecision,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','extract','evaluate'])
    p.add_argument('--bundle',type=Path);p.add_argument('--groups',type=Path)
    p.add_argument('--model',choices=['source','msre']);a=p.parse_args()
    if a.action=='prepare':
        assert a.bundle and a.groups
        assert a.bundle.resolve().is_relative_to(Path('/home/scv/shared'))
        assert any(a.groups.resolve().is_relative_to(r) for r in [REPO,Path('/home/scv/shared')])
        prepare(a.bundle,a.groups)
    elif a.action=='extract':assert a.model;extract(a.model)
    else:evaluate()


if __name__=='__main__':main()

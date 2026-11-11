"""CPU-only post-hoc diagnostics of frozen score caches; no fitting/selection."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'outputs/phase231/phase231_20261026'
S = ROOT / 'outputs/phase230/phase230_20261026/delivery'
OUT = ROOT / 'outputs/research_audits/phase231_local_tradeoffs_20261026'
IDENTITIES = {}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def load_json(path):
    IDENTITIES[str(path.relative_to(ROOT))] = digest(path)
    return json.loads(path.read_text(encoding='utf-8'))


def metric(cm):
    cm = np.asarray(cm, dtype=np.float64)
    support = cm.sum(1)
    u = support + cm.sum(0) - cm.diagonal()
    iou = np.divide(cm.diagonal()*100, u, out=np.full(3, np.nan), where=u>0)
    return dict(mIoU=float(np.nanmean(iou)),
                Cloud_recall=float(cm[1, 1]*100/support[1]) if support[1] else None,
                Surface_to_Cloud_FPR=float(cm[0, 1]*100/support[0]) if support[0] else None,
                Shadow_recall=float(cm[2, 2]*100/support[2]) if support[2] else None)


def save(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Freeze the diagnostic grid before reading score arrays. No optimum chosen.
    design = {'bias_grid': [float(x) for x in np.arange(-1, 2.5001, .125)],
              'models': ['Source', 'L8_21701', 'L8_21702'],
              'purpose': 'Post-hoc sensitivity, not deployment selection or fitting.',
              'extra_points': 'Original Phase231 frozen b for each model.',
              'training': 0, 'new_forwards': 0, 'new_intervals': 0}
    (OUT / 'DIAGNOSTIC_DESIGN.json').write_text(json.dumps(design, indent=2)+'\n', encoding='utf-8')
    biases = {r['model']: r['b'] for r in load_json(P/'TABLE1.json')}
    expected = {(r['model'], r['scene'], r['decision']): r for r in load_json(P/'VAL_PER_SCENE.json')}
    transitions = {(r['model'], r['scene']): r for r in load_json(P/'DIRECT_PIXEL_TRANSITIONS.json')}
    receipts = load_json(S/'SCORE_RECEIPTS.json')
    rows, details, checks = [], [], []
    for model in design['models']:
        for receipt in sorted((r for r in receipts if r['model']==model), key=lambda r:r['scene']):
            scene = receipt['scene']
            sp, lp = S/receipt['score_path'], S/receipt['label_path']
            assert digest(sp)==receipt['score_SHA'] and digest(lp)==receipt['label_SHA']
            IDENTITIES[str(sp.relative_to(ROOT))]=receipt['score_SHA']
            IDENTITIES[str(lp.relative_to(ROOT))]=receipt['label_SHA']
            p = np.load(sp).astype(np.float64)
            y = np.load(lp)
            assert p.shape==(3,512,512) and y.shape==(512,512) and np.isfinite(p).all()
            valid = y != 255
            assert np.isin(y[valid],[0,1,2]).all()
            yy = y[valid].astype(np.int64)
            raw = p.argmax(0)
            assert hashlib.sha256(raw.astype(np.uint8).tobytes()).hexdigest()==receipt['prediction_SHA']
            frozen_prediction = None
            for b in sorted(set(design['bias_grid']+[biases[model]])):
                q=p.copy(); q[1]*=np.exp(b)
                pred=q.argmax(0)
                cm=np.bincount(yy*3+pred[valid],minlength=9).reshape(3,3)
                row=dict(model=model,scene=scene,b=b,point_type='frozen_training_bias' if b==biases[model] else 'diagnostic_grid')
                row.update(metric(cm))
                row.update({f'cm_{i}{j}':int(cm[i,j]) for i in range(3) for j in range(3)})
                rows.append(row)
                if b==0 or b==biases[model]:
                    label='raw' if b==0 else 'calibrated'
                    assert cm.tolist()==expected[model,scene,label]['confusion']
                    for k in ['mIoU','Cloud_recall','Surface_to_Cloud_FPR','Shadow_recall']:
                        a,z=row[k],expected[model,scene,label][k]
                        assert (a is None and z is None) or abs(a-z)<1e-10
                    if label=='calibrated': frozen_prediction=pred
            assert frozen_prediction is not None
            jj=np.bincount(yy*9+raw[valid]*3+frozen_prediction[valid],minlength=27).reshape(3,3,3)
            assert jj.tolist()==transitions[model,scene]['reference_raw_calibrated_J']
            # Positive Cloud-only bias can only change another prediction to Cloud.
            assert all(jj[:,i,j].sum()==0 for i in range(3) for j in range(3) if i!=j and j!=1)
            before=expected[model,scene,'raw']; after=expected[model,scene,'calibrated']
            cloud_saved=int(jj[1,0,1]+jj[1,2,1])
            shadow_lost=int(jj[2,2,1])
            surface_new_fp=int(jj[0,0,1]+jj[0,2,1])
            details.append(dict(model=model,scene=scene,b=biases[model],
                                delta_mIoU_pp=after['mIoU']-before['mIoU'],
                                Surface_pixels=before['Surface_pixels'], Cloud_pixels=before['Cloud_pixels'], Shadow_pixels=before['Shadow_pixels'],
                                new_Surface_to_Cloud=surface_new_fp,
                                original_correct_Surface_to_Cloud=int(jj[0,0,1]),
                                original_Shadow_prediction_on_Surface_to_Cloud=int(jj[0,2,1]),
                                rescued_Cloud=cloud_saved,
                                original_correct_Shadow_to_Cloud=shadow_lost,
                                original_Surface_prediction_on_Shadow_to_Cloud=int(jj[2,0,1]),
                                correct_Shadow_before=int(np.asarray(before['confusion'])[2,2]),
                                Shadow_recall_loss_pp=100*shadow_lost/before['Shadow_pixels'] if before['Shadow_pixels'] else None))
            checks.append({'model':model,'scene':scene,'score_SHA':True,'label_SHA':True,
                           'raw_prediction_SHA':True,'raw_and_frozen_metrics':True,'direct_transition_J':True})
    pooled=[]
    for model in design['models']:
        for b in sorted(set(design['bias_grid']+[biases[model]])):
            rr=[r for r in rows if r['model']==model and r['b']==b]
            assert len(rr)==10
            cm=np.array([[sum(r[f'cm_{i}{j}'] for r in rr) for j in range(3)] for i in range(3)])
            row=dict(model=model,b=b,point_type=rr[0]['point_type'])
            row.update(metric(cm)); pooled.append(row)
    summaries=[]
    for model in design['models']:
        rr=[r for r in details if r['model']==model]
        ordered=sorted(rr,key=lambda x:x['new_Surface_to_Cloud'],reverse=True)
        total=sum(r['new_Surface_to_Cloud'] for r in rr)
        shadow=sum(r['original_correct_Shadow_to_Cloud'] for r in rr)
        correct=sum(r['correct_Shadow_before'] for r in rr)
        summaries.append(dict(model=model,new_Surface_to_Cloud=total,
                              top1_scene=ordered[0]['scene'],top1_share_pct=100*ordered[0]['new_Surface_to_Cloud']/total,
                              top3_share_pct=100*sum(r['new_Surface_to_Cloud'] for r in ordered[:3])/total,
                              rescued_Cloud=sum(r['rescued_Cloud'] for r in rr),
                              original_correct_Shadow_to_Cloud=shadow,
                              fraction_original_correct_Shadow_lost_pct=100*shadow/correct,
                              mIoU_improved_scenes=sum(r['delta_mIoU_pp']>1e-10 for r in rr),
                              mIoU_worse_scenes=sum(r['delta_mIoU_pp']< -1e-10 for r in rr)))
    save('BIAS_CURVES_PER_SCENE.csv',rows)
    save('BIAS_CURVES_POOLED.csv',pooled)
    save('FROZEN_BIAS_DIRECT_COSTS.csv',details)
    save('CONCENTRATION_SUMMARY.csv',summaries)
    sensitivity=[]
    for model in design['models']:
        baseline={r['scene']:r['mIoU'] for r in rows if r['model']==model and r['b']==0}
        for b in design['bias_grid']:
            rr=[r for r in rows if r['model']==model and r['b']==b]
            deltas=[r['mIoU']-baseline[r['scene']] for r in rr]
            sensitivity.append(dict(model=model,b=b,improved_scenes=sum(x>1e-10 for x in deltas),
                                    worse_scenes=sum(x< -1e-10 for x in deltas),
                                    unchanged_scenes=sum(abs(x)<=1e-10 for x in deltas),
                                    min_scene_delta_mIoU_pp=min(deltas),
                                    mean_scene_delta_mIoU_pp=float(np.mean(deltas))))
    save('GRID_SCENE_SIGNS.csv',sensitivity)
    (OUT/'CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    (OUT/'AUDIT.json').write_text(json.dumps({'inputs':IDENTITIES,'script_sha256':digest(Path(__file__)),
                                             'model_calls':0,'fitting':0,'new_intervals':0,'diagnostic_grid_points':len(design['bias_grid']),
                                             'curves_rows':len(rows),'separate_scene_count':10,'seed_interpretation':'two repeats, not 20 independent scenes'},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()

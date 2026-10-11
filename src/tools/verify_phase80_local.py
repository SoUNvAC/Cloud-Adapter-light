"""Independent delivery statistics and Git blob checks; no reserved pixels."""
import argparse,csv,json,subprocess,hashlib
from pathlib import Path
import numpy as np
from phase80_common import read,digest

def interval(vals):
    x=np.asarray(vals,np.float64)
    out=dict(mean=float(np.sum(x)/len(x)) if len(x) else None,ci95=None,upper95_one_sided=None)
    if len(x)>1:
        samples=np.random.default_rng(80011).choice(x,size=(10000,len(x)),replace=True)
        means=np.sum(samples,axis=1)/len(x)
        out.update(ci95=np.percentile(means,[2.5,97.5]).tolist(),upper95_one_sided=float(np.percentile(means,95)))
    return out

def near(a,b):
    assert (a is None and b is None) or (a is not None and b is not None and np.allclose(a,b,rtol=0,atol=1e-10)),(a,b)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--execution-commit',default='1c91e75');a=p.parse_args();root=a.root
    hashes=read(root/'artifact_hashes.json')
    for rel,h in hashes.items():assert digest(root/rel)==h,rel
    lock=read(root/'training_lock.json')
    for rel,h in lock['code_sha256'].items():
        blob=subprocess.check_output(['git','show',a.execution_commit+':'+rel])
        assert hashlib.sha256(blob).hexdigest()==h,rel
    pre=read(root/'preflight.json');train=read(root/'training_completed.json');r=read(root/'phase80_report.json')
    calibration=read(root/'calibration_lock.json')
    assert digest(root/'baseline_reproduction.json')==calibration['baseline_reproduction_sha256']
    assert (root/'EXIT_CODE').read_text().strip()=='0'
    assert (root/'VERIFICATION_EXIT_CODE').read_text().strip()=='0'
    assert pre['base_tensors_identical'] and train['base_tensors_still_identical']
    assert all(v<=1e-6 for v in pre['zero_residual_max_absolute_logit_error'].values())
    assert len(pre['trainable_names'])==96 and all(g['finite'] and g['connected'] for g in pre['gradients'].values())
    curve=train['curve'];assert [x['iteration'] for x in curve]==list(range(500,4001,500))
    assert max(curve,key=lambda v:(v['metrics']['mIoU'],-v['iteration']))['iteration']==train['best_iteration']
    assert len(train['checkpoints'])==9
    expected_names={'M','M+S','L','L+S','Source','MsRE','LoRA_argmax','L3'}
    assert set(r['policies'])==expected_names
    csvrows=list(csv.DictReader((root/'development_groups.csv').open()))
    assert len(csvrows)==256
    lookup={(n,v['product']):v for n,q in r['policies'].items() for v in q['rows']}
    assert len({(v['policy'],v['product']) for v in csvrows})==256
    for row in csvrows:
        ref=lookup[row['policy'],row['product']];assert row['group_id']==ref['group_id']
        for k in ['positive','negative','recall','precision','fpr','miou_percent','shadow_to_surface','shadow_to_cloud','predicted_shadow_fraction']:
            near(float(row[k]) if row[k] else None,ref[k])
        near(float(row['ap']) if row['ap'] else None,ref['ranking']['ap'])
    for n,q in r['policies'].items():
        rr=q['rows'];assert len(rr)==len({v['group_id'] for v in rr})==32
        supported=[x for x in rr if x['positive']];assert len(supported)==7
        for row in rr:
            cm=np.asarray(row['confusion']);p=int(cm[2].sum());neg=int(cm[:2].sum());tp=int(cm[2,2]);fp=int(cm[:2,2].sum())
            assert row['positive']==p and row['negative']==neg
            near(row['recall'],tp/p if p else None);near(row['fpr'],fp/neg)
            near(row['precision'],tp/(tp+fp) if tp+fp else None)
            if not p:assert row['ranking']['ap'] is None
            union=cm.sum(0)+cm.sum(1)-np.diag(cm)
            iou=np.divide(np.diag(cm),union,out=np.full(3,np.nan),where=union>0)
            near(row['miou_percent'],float(np.nanmean(iou)*100))
            assert row['shadow_to_surface']==int(cm[2,0]) and row['shadow_to_cloud']==int(cm[2,1])
        metrics={'macro_ap':[x['ranking']['ap'] for x in supported],
                 'macro_recall':[x['recall'] for x in supported],
                 'macro_precision':[x['precision'] for x in rr if x['precision'] is not None],
                 'macro_miou_percent':[x['miou_percent'] for x in rr],
                 'macro_fpr':[x['fpr'] for x in rr],
                 'shadow_supported_fpr':[x['fpr'] for x in supported],
                 'no_shadow_fpr':[x['fpr'] for x in rr if not x['positive']]}
        for key,vals in metrics.items():
            check=interval(vals)
            for k in ['mean','ci95','upper95_one_sided']:near(q[key][k],check[k])
        assert q['fpr_above_one_percent_groups']==sum(x['fpr']>.01 for x in rr)
        cm=np.sum([x['confusion'] for x in rr],axis=0)
        assert np.array_equal(cm,np.asarray(q['confusion']))
        near(q['fpr'],cm[:2,2].sum()/cm[:2].sum());near(q['recall'],cm[2,2]/cm[2].sum())
    for key,v in r['comparisons'].items():
        a_name,b_name=key.split(' minus ');aa=r['policies'][a_name]['rows'];bb=r['policies'][b_name]['rows']
        assert [x['group_id'] for x in aa]==[x['group_id'] for x in bb]
        values={'ap_pp':[100*(a['ranking']['ap']-b['ranking']['ap']) for a,b in zip(aa,bb) if a['positive']],
                'recall_pp':[100*(a['recall']-b['recall']) for a,b in zip(aa,bb) if a['positive']],
                'fpr_pp':[100*(a['fpr']-b['fpr']) for a,b in zip(aa,bb)],
                'miou_pp':[a['miou_percent']-b['miou_percent'] for a,b in zip(aa,bb)]}
        for kind,vals in values.items():
            check=interval(vals)
            for k in ['mean','ci95','upper95_one_sided']:near(v[kind][k],check[k])
        aps=np.asarray(values['ap_pp']);assert v['improved']==int((aps>1).sum()) and v['declined']==int((aps< -1).sum())
        near(v['worst_ap_pp'],float(aps.min()));near(v['worst_recall_pp'],min(values['recall_pp']))
    out=dict(status='verified',artifact_count=len(hashes),git_blob_files=len(lock['code_sha256']),
        groups=32,shadow_groups=7,policies=8,comparisons=4,bootstrap_resamples=10000,seed=80011,
        full_local_pixel_cache_recomputation=False,work_dirs_not_transferred=True,
        evidence='artifact hashes, preflight, checkpoint selection, all 256 confusions, macro/paired group bootstrap')
    with (root/'LOCAL_VERIFIED.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out))

if __name__=='__main__':main()

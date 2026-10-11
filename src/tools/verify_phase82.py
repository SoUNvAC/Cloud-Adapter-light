"""Independent delivered metrics/bootstrap/selection/KD-budget verification."""
import argparse,csv,json,hashlib,subprocess
from pathlib import Path
import numpy as np
from phase80_common import read,write,digest
from verify_phase81 import near,independent_metrics

def stats(values,reported):
    x=np.asarray(values,np.float64);bs=np.random.default_rng(82011).choice(x,size=(10000,len(x)),replace=True).mean(1)
    near(float(x.mean()),reported['mean']);near(np.percentile(bs,[2.5,97.5]),reported['ci95']);near(float(np.percentile(bs,95)),reported['upper95_one_sided'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    hashes=read(root/'artifact_hashes.json')
    for rel,h in hashes.items():assert digest(root/rel)==h,rel
    commit=(root/'GIT_COMMIT').read_text().strip();lock=read(root/'training_lock.json')
    for rel,h in lock['code_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+rel])).hexdigest()==h,rel
    assert lock['KD_temperature']==lock['KD_weight']==1
    assert read(root/'baseline_reproduction.json')['status']=='all_96_reference_rows_exactly_reproduced'
    assert (root/'TRAINING_EXIT_CODE').read_text().strip()==(root/'EXIT_CODE').read_text().strip()=='0'
    pre=read(root/'preflight.json');train=read(root/'training_completed.json');cal=read(root/'calibration_lock.json')
    assert pre['strict_load'] and pre['teacher_frozen_unchanged'] and train['teacher_still_identical'] and train['base_tensors_still_identical']
    assert train['teacher_not_in_student_checkpoint'] and len(pre['trainable_names'])==96 and pre['trainable_parameters']==294912
    assert all(v<=1e-6 for v in pre['zero_residual_max_absolute_logit_error'].values())
    assert set(cal['models'])=={'D+S'} and cal['models']['D+S']['fit_macro_fpr']<=.01
    assert cal['models']['D+S']['columns']==[0,1,2,3]
    curve=train['curve'];assert [x['iteration'] for x in curve]==list(range(500,4001,500))
    assert max(curve,key=lambda x:(x['metrics']['mIoU'],-x['iteration']))['iteration']==train['best_iteration']
    assert len(train['checkpoints'])==9
    trace=[json.loads(s) for s in (root/'loss_gradient_curve.jsonl').read_text().splitlines()]
    assert [x['iteration'] for x in trace]==list(range(1,4001))
    assert all(len(x['gradient_norms'])==96 and np.isfinite(list(x['gradient_norms'].values())).all() and np.isfinite(x['kd']) and np.isfinite(x['supervised']) for x in trace)
    cost=read(root/'cost_audit.json');assert max(x['cuda_peak_allocated_bytes'] for x in trace)==cost['training_peak_allocated_bytes']
    report=read(root/'phase82_report.json');policies=report['policies'];assert set(policies)=={'MM','LL','LM','D+S'}
    csvrows=list(csv.DictReader((root/'development_groups.csv').open()));assert len(csvrows)==128
    for n,p in policies.items():
        rr=p['rows'];assert len(rr)==32 and sum(r['positive']>0 for r in rr)==7
        for r in rr:
            c=np.asarray(r['confusion']);iou,miou=independent_metrics(c);near(miou,r['miou_percent'])
            for x,y in zip(iou,r['iou_percent']):near(x,y)
            assert c.sum()==r['valid_pixels'] and c.sum(1).tolist()==r['reference_support']
            near(int(c[2,2])/int(c[2].sum()) if c[2].sum() else None,r['recall'])
            near(int(c[:2,2].sum())/int(c[:2].sum()),r['fpr'])
            x=next(v for v in csvrows if v['policy']==n and v['product']==r['product'])
            for key in ['ap','recall','fpr','miou_percent']:near(float(x[key]) if x[key] else None,r[key])
        total=np.sum([r['confusion'] for r in rr],axis=0);assert total.tolist()==p['confusion'];near(independent_metrics(total)[1],p['miou_percent'])
        for field,key in [('ap','macro_ap'),('recall','macro_recall'),('fpr','macro_fpr'),('miou_percent','macro_miou_percent')]:stats([r[field] for r in rr if r[field] is not None],p[key])
    for key,c in report['comparisons'].items():
        ref=key.split(' minus ')[1];a=policies['D+S'];b=policies[ref]
        for field,name,scale in [('ap','AP_pp',100),('recall','recall_pp',100),('miou_percent','mIoU_pp',1),('fpr','FPR_pp',100)]:
            stats([scale*(x[field]-y[field]) for x,y in zip(a['rows'],b['rows']) if x[field] is not None],c[name])
        near(a['miou_percent']-b['miou_percent'],c['pooled_mIoU_pp'])
    d=policies['D+S'];l=policies['LL']
    expected=[d['miou_percent']-l['miou_percent']>=1,
        100*(d['macro_ap']['mean']-l['macro_ap']['mean'])>=-1,
        100*(d['macro_recall']['mean']-l['macro_recall']['mean'])>=-1,
        d['macro_miou_percent']['mean']-l['macro_miou_percent']['mean']>=-1,d['macro_fpr']['mean']<=.01]
    assert list(report['continuation_gate']['checks'].values())==expected and report['continuation_gate']['retain_fixed_scheme']==all(expected)
    assert read(root/'SHARED_PIXEL_VERIFIED.json')['status']=='passed'
    write(root/'LOCAL_VERIFIED.json',dict(status='passed',artifact_SHA=len(hashes),code_git_blobs=True,all_128_confusions_and_metrics=True,
        independent_group_bootstrap=True,five_screening_checks=True,all_4000_loss_gradient_records=True,checkpoint_selection=True,
        shared_independent_pixel_AP_and_confusion_checks=True,pixels_transferred=False,weights_transferred=False))
    print('Phase82 local independent verification passed')

if __name__=='__main__':main()

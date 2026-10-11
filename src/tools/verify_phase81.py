"""Independent local confusion/statistics/decision-table and execution SHA audit."""
import argparse,csv,hashlib,subprocess
from pathlib import Path
import numpy as np
from phase80_common import read,write,digest

def near(a,b):
    assert (a is None and b is None) or (a is not None and b is not None and np.allclose(a,b,rtol=0,atol=1e-10)),(a,b)

def stats(values,reported):
    x=np.asarray(values,np.float64);bs=np.random.default_rng(81011).choice(x,size=(10000,len(x)),replace=True).mean(1)
    near(float(x.mean()),reported['mean']);near(np.percentile(bs,[2.5,97.5]),reported['ci95']);near(float(np.percentile(bs,95)),reported['upper95_one_sided'])

def independent_metrics(cm):
    c=np.asarray(cm,dtype=np.int64);union=c.sum(0)+c.sum(1)-c.diagonal()
    iou=[None if not u else 100*float(c[i,i])/int(u) for i,u in enumerate(union)]
    return iou,float(np.mean([x for x in iou if x is not None]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    hashes=read(root/'artifact_hashes.json')
    for rel,h in hashes.items():assert digest(root/rel)==h,rel
    lock=read(root/'input_configuration_lock.json')
    for rel,h in lock['code_sha256'].items():
        blob=subprocess.check_output(['git','show',lock['git_commit']+':'+rel]);assert hashlib.sha256(blob).hexdigest()==h,rel
    assert not any(lock[k] for k in ('training','fitting','threshold_search','new_inference','fit_pixels','confirmation_access','work_dirs_transfer'))
    report=read(root/'phase81_report.json');policies=report['policies'];assert set(policies)=={'MM','LL','LM','ML'}
    table=list(csv.DictReader((root/'development_groups.csv').open()))
    assert len(table)==128 and len({r['group_id'] for r in table})==32
    for name,p in policies.items():
        rows=p['rows'];assert len(rows)==32
        for r in rows:
            c=np.asarray(r['confusion']);iou,miou=independent_metrics(c)
            near(miou,r['miou_percent'])
            for x,y in zip(iou,r['iou_percent']):near(x,y)
            assert c.sum()==r['valid_pixels'] and c.sum(1).tolist()==r['reference_support']
            assert sum(v is not None for v in iou)==r['defined_iou_classes']
            near(int(c[2,2])/int(c[2].sum()) if c[2].sum() else None,r['recall'])
            near(int(c[:2,2].sum())/int(c[:2].sum()),r['fpr'])
            rr=next(v for v in table if v['policy']==name and v['product']==r['product'])
            for key in ('ap','fpr','miou_percent','recall','precision'):
                near(float(rr[key]) if rr[key] else None,r[key])
            for i,cname in enumerate(('Surface','Cloud','Shadow')):
                for j,pname in enumerate(('Surface','Cloud','Shadow')):assert int(rr[cname+'_to_'+pname])==c[i,j]
        total=np.sum([r['confusion'] for r in rows],axis=0);assert total.tolist()==p['confusion']
        near(independent_metrics(total)[1],p['miou_percent'])
        for field,key in [('ap','macro_ap'),('recall','macro_recall'),('fpr','macro_fpr'),('miou_percent','macro_miou_percent')]:
            stats([r[field] for r in rows if r[field] is not None],p[key])
        stats([r['fpr'] for r in rows if r['positive']],p['shadow_fpr'])
        stats([r['fpr'] for r in rows if not r['positive']],p['no_shadow_fpr'])
        assert sum(r['fpr']>.01 for r in rows)==p['fpr_above_one_percent_groups']
    for a,b in [('LL','LM'),('MM','ML')]:
        near(policies[a]['pooled_ap'],policies[b]['pooled_ap'])
        for x,y in zip(policies[a]['rows'],policies[b]['rows']):
            for key in ('ap','recall','fpr','precision'):near(x[key],y[key])
            near(x['iou_percent'][2],y['iou_percent'][2])
            assert np.array_equal(np.asarray(x['confusion'])[:,2],np.asarray(y['confusion'])[:,2])
    decisions=read(root/'decision_combinations.json');sums=np.sum([r['counts'] for r in decisions['rows']],axis=0)
    assert sums.tolist()==decisions['pooled_counts']
    for d in decisions['rows']:
        c=np.asarray(d['counts']);rr={n:next(v for v in p['rows'] if v['product']==d['product']) for n,p in policies.items()}
        assert c.sum(0).tolist()==rr['LL']['reference_support']
        assert (c[0]+c[1]).tolist()==np.asarray(rr['LL']['confusion'])[:,2].tolist()
        assert (c[0]+c[2]).tolist()==np.asarray(rr['MM']['confusion'])[:,2].tolist()
    for key,v in report['comparisons'].items():
        a,b=key.split(' minus ');delta=[x['miou_percent']-y['miou_percent'] for x,y in zip(policies[a]['rows'],policies[b]['rows'])]
        stats(delta,v['macro_miou_pp']);near(policies[a]['miou_percent']-policies[b]['miou_percent'],v['pooled_miou_pp'])
    assert read(root/'invariant_checks.json')['status']=='passed'
    assert read(root/'reproduction_checks.json')['status']=='all_64_scene_confusions_AP_and_metrics_exact'
    assert (root/'EXIT_CODE').read_text().strip()=='0'
    write(root/'LOCAL_VERIFIED.json',dict(status='passed',artifacts_SHA=len(hashes),execution_git_blobs=True,confusions=128,
          statistics_independently_recomputed=True,decision_combinations_consistent=True,group_bootstrap_seed=81011,
          shared_pixel_AP_and_confusions_independently_verified_in_execution=True,no_pixels_transferred=True))
    print('Phase81 local verification passed')

if __name__=='__main__':main()

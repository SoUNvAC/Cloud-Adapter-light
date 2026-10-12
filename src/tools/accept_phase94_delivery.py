"""Additional Phase94 acceptance against the pre-failure G7 hash and sealed group IDs."""
import argparse, hashlib, json
from pathlib import Path
from verify_phase94_delivery import verify

def accept(root):
    result=verify(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8-sig'))
    digest=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    sl=read('split_input_lock.json');ml=read('two_model_lock.json');tl=read('four_threshold_lock.json');report=read('phase94_report.json');repair=read('repair_authorization_lock.json')
    original='2e11a30996b04768b03d6668346d8a34ffda4100c93bbeded7d684e59091c6e6'
    assert digest('G7_100.pkl')==repair['G7_preserved_sha256']==original
    assert digest('repair_authorization_lock.json')==ml['repair_authorization_lock_sha256']
    assert ml['new_fits']==report['new_fits']==3 and report['final_models']==2
    identities={s:[r['group_id'] for r in sl[k]] for s,k in [('in','fit'),('cal','calibration')]}
    products={s:[r['product'] for r in sl[k]] for s,k in [('in','fit'),('cal','calibration')]}
    for n,m in tl['models'].items():
        for s,t in m.items():
            assert [r['group_id'] for r in t['rows']]==identities[s]
            assert [r['product'] for r in t['rows']]==products[s]
    for row in read('train_calibration_diagnostics.json')['rows']:
        assert [r['group_id'] for r in row['rows']]==identities[row['data_subset']]
        if row['data_subset']==row['threshold_source']:
            assert row['macro_fpr']['mean']==tl['models'][row['model']][row['threshold_source']]['macro_fpr']
    for c in read('calibration_cpu_checks.json')['rows']:
        assert c['max_error']<=(6e-8 if c['model']=='G7_100' else 1e-4)
    for domain,v in report['domains'].items():
        for c in v['comparisons'].values():
            fprs=[r for r in c['pairs'] if r['delta_fpr_pp'] is not None]
            assert c['largest_fpr_increase']==max(fprs,key=lambda r:r['delta_fpr_pp'],default=None)
        for model in ['G7_100','SourceS_100']:
            a=v['policies'][model+'_T_in'];b=v['policies'][model+'_T_cal']
            assert a['macro']['ap']==b['macro']['ap']
    return dict(**result,preserved_G7_exact_original_SHA=True,calibration_and_fit_identity_order_verified=True,
        actual_fit_attempts=3,original_failure_preserved=True,formal_confirmation=False)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);args=a.parse_args();result=accept(args.root)
    path=args.root/'LOCAL_ACCEPTANCE.json'
    with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

"""Independent delivery arithmetic and gate checks; no remote pixels."""
import argparse,hashlib,json,csv
from pathlib import Path
import numpy as np
from phase65d_metrics import measures
from phase74_math import interval,continue_conditions


def verify(root):
    read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    r=read('phase74_report.json');lock=read('policy_lock.json');initial=read('input_lock.json')
    sha=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    assert r['policy_lock_sha256']==sha('policy_lock.json')
    for n in ['input_lock','feature_index','fit_audit']:assert lock[n+'_sha256']==sha(n+'.json')
    assert initial['max_fits']==lock['fit_attempts']==r['fits']==2
    assert r['new_network_forwards']==0 and not r['confirmation_access']
    assert read('baseline_reproduction.json')['all_32_L3_and_B_exact']
    groups=None
    for name,p in r['policies'].items():
        rows=p['rows'];ids=[x['group_id'] for x in rows]
        assert len(rows)==len(set(ids))==32
        if groups is None:groups=ids
        else:assert groups==ids
        pos=[x for x in rows if x['positive']];assert len(pos)==7
        assert p['macro_ap']==np.mean([x['ranking']['ap'] for x in pos])
        assert p['macro_recall']==np.mean([x['recall'] for x in pos])
        total=np.sum([x['confusion'] for x in rows],axis=0)
        for key,val in measures(total).items():assert p[key]==val,(name,key)
        for x in rows:
            for key,val in measures(x['confusion']).items():assert x[key]==val,(name,x['product'],key)
        assert p['fpr_strata']['all']==interval([x['fpr'] for x in rows])
    for c in r['comparisons'].values():
        valid=[p for p in c['pairs'] if p['delta_ap'] is not None]
        assert c['ap']==interval([x['delta_ap'] for x in valid])
        assert c['recall']==interval([x['delta_recall'] for x in valid])
    assert continue_conditions(r['policies'])==r['continue_conditions']
    assert r['continue_branch']==all(r['continue_conditions'].values())
    audit=read('neighbor_audit.json');assert len(audit)==370
    for a in audit:
        assert sum(a['excluded_gt_crosscount_offline_only'].values())==a['excluded_image_valid']
        assert 0<=a['fallback_eval_fraction']<=1
    for a,b in zip(audit[::2],audit[1::2]):
        assert a['product']==b['product'] and a['excluded_image_valid']==b['excluded_image_valid']
    with (root/'development_groups.csv').open(encoding='utf-8') as f:assert len(list(csv.DictReader(f)))==192
    return dict(status='verified',groups=32,positive_groups=7,neighbor_audits=370,policies=6,
        hashes_and_confusions_and_group_bootstrap_and_gates=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');a=parser.parse_args();print(json.dumps(verify(Path(a.root)),indent=2))

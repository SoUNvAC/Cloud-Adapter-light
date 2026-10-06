"""Reject mechanism development until verified preceding evidence is available.

Gate JSON must link real immutable reports; this script is not a trainer.
"""
import argparse
import json
from pathlib import Path
from phase65a_freeze import allowed, digest


def validate(a, b):
    if a['status'] != 'complete' or b['status'] != 'complete':
        raise ValueError('Both earlier phases must be complete')
    if not (a['h1_r_ci95'][0] > 0 and a['h1_prediction_rmse'] < a['constant_rmse']
            and a['h2_auc_ci95'][0] > 0.5):
        raise ValueError('65a evidence gate failed; do not redefine endpoints')
    if b['matched_source_loss_pp'] > 1 or b['tiny_fit_miou'] < 95:
        raise ValueError('65b source/input gate failed')
    supported = [e['name'] for e in b['evidence']
                 if e['delta_auc_ci95'][0] > 0 and e['holm_adjusted_p'] < 0.05]
    if not supported:
        raise ValueError('No independently supported extra evidence')
    return supported


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase65a', type=Path, required=True)
    p.add_argument('--phase65b', type=Path, required=True)
    args = p.parse_args()
    reports = []
    for path in (args.phase65a, args.phase65b):
        report = json.loads(allowed(path).read_text(encoding='utf-8'))
        if not report['artifacts']:
            raise ValueError('Missing source artifacts')
        for artifact in report['artifacts']:
            if digest(artifact['path']) != artifact['sha256']:
                raise ValueError('Artifact SHA mismatch')
        reports.append(report)
    print(json.dumps({'supported_evidence': validate(*reports),
                      'status': 'eligible_to_preregister_one_mechanism',
                      'training_authorized': False}))

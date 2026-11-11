"""Read existing metrics/code; no model, new selection, or confidence intervals."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/research_audits/interpolation_endpoint_and_head_20261027'


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    p=ROOT/'outputs/phase227/phase227_20261025/TABLE_B_ALL_CANDIDATES.csv'
    with p.open(encoding='utf-8-sig',newline='') as f: raw=list(csv.DictReader(f))
    assert len(raw)==100
    index={(r['domain'],r['seed'],r['arm'],float(r['alpha'])):r for r in raw}
    rows=[]
    for r in raw:
        end=index[r['domain'],r['seed'],r['arm'],1.]
        for split,col in [('val','val_mIoU'),('test','test_mIoU')]:
            rows.append(dict(domain=r['domain'],seed=r['seed'],arm=r['arm'],alpha=float(r['alpha']),split=split,
                             mIoU=float(r[col]),endpoint_mIoU=float(end[col]),
                             delta_vs_endpoint_pp=float(r[col])-float(end[col]),
                             support_status='HISTORICAL_OLD_FILL_INCLUDED_NOT_CLEAN_REEVALUATED' if r['domain']=='L8' else 'ORIGINAL_SPARCS_SUPPORT',
                             role='Fixed candidate, not selected by viewed test'))
    with (OUT/'PER_SEED_ENDPOINT_COMPARISONS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    means=[]
    for domain in ['L8','SPARCS']:
        for split in ['val','test']:
            for arm in ['C','D','H','W1','W10']:
                for alpha in [0,.25,.5,.75,1]:
                    rr=[r for r in rows if (r['domain'],r['split'],r['arm'],r['alpha'])==(domain,split,arm,alpha)]
                    assert len(rr)==2
                    means.append(dict(domain=domain,split=split,arm=arm,alpha=alpha,
                                      mIoU=mean(r['mIoU'] for r in rr),endpoint_mIoU=mean(r['endpoint_mIoU'] for r in rr),
                                      delta_vs_endpoint_pp=mean(r['delta_vs_endpoint_pp'] for r in rr),support_status=rr[0]['support_status']))
    with (OUT/'MEAN_ENDPOINT_COMPARISONS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(means[0]));w.writeheader();w.writerows(means)
    source_files=[p,ROOT/'src/configs/protocol/phase64_source_parent_rgb.py',ROOT/'src/configs/protocol/phase22_clean_v8_l1c.py',
                  ROOT/'src/configs/light/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c.py',
                  ROOT/'src/configs/light/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_l1c.py',
                  ROOT/'src/configs/light/cloud_adapter_dinov2_s_mask2former_lite_q25_l1c.py',
                  ROOT/'src/configs/light/cloud_adapter_dinov2_s_mask2former_lite_l1c.py',
                  ROOT/'src/configs/control/cloud_adapter_dinov2_s_mask2former_4x_l1c.py',
                  ROOT/'src/tools/export_phase12_onnx.py',ROOT/'src/tools/execute_phase216.py']
    audit=dict(inputs={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in source_files},
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               new_forwards=0,new_training=0,new_CI=0,new_deployment_selection=0,
               head='Entire 1,316,516-parameter nonlinear Mask2Former decode_head; 25 queries, query decoder2, pixel encoder2, hidden128, final3 semantic classes.',
               output='Sum over query softmax class scores (no-object removed) times sigmoid mask scores; not per-pixel raw affine logits.',
               consequence='No strict weight-interpolation/output-mixture identity. Approximation magnitude not measured.')
    (OUT/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([r for r in means if r['arm']=='C' and r['split']=='test'],indent=2))


if __name__=='__main__':main()

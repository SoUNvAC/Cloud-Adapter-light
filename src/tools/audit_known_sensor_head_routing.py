"""Construct known-domain routing comparison from frozen tables; no inference."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/research_audits/head_routing_deployment_20261026'
locks={}


def csvread(path):
    p=ROOT/path;locks[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def jsread(path):
    p=ROOT/path;locks[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding='utf-8-sig'))


def save(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


meta=csvread('outputs/phase65/source_metadata/acquired/val_metadata.csv')
assert len(meta)==535 and len({r['roi_id'] for r in meta})==107
assert all(r['s2_id'].startswith(('S2A_MSIL1C_','S2B_MSIL1C_')) for r in meta)
allc=csvread('outputs/phase227/phase227_20261025/TABLE_B_ALL_CANDIDATES.csv')
l8rule=csvread('outputs/phase222/phase222_20261024/JOINT_RULE.csv')
sprule=csvread('outputs/phase217/phase217_20261023/JOINT_RULE.csv')
source=jsread('outputs/phase222/phase222_20261024/EXPECTED_Source__source.json')
cm=source['confusion']
den=[sum(cm[i])+sum(row[i] for row in cm)-cm[i][i] for i in range(3)]
source_miou=100*sum(cm[i][i]/den[i] for i in range(3))/3
assert abs(source_miou-77.81522179292283)<1e-10

rows=[]
for domain in ['L8','SPARCS']:
    for seed in ['21701','21702']:
        if domain=='L8':
            choice=next(r for r in l8rule if r['seed']==seed and r['arm']=='D' and r['budget_pp']=='1')
            alpha=float(choice['alpha'])
        else:
            choice=next(r for r in sprule if r['seed']==seed and r['context']=='B1')
            alpha=float(choice['D_alpha'])
        point=next(r for r in allc if r['domain']==domain and r['seed']==seed and r['arm']=='D' and float(r['alpha'])==alpha)
        for deployment in ['shared_D_head','known_sensor_route_Source_D']:
            routed=deployment.startswith('known_sensor')
            loss=0. if routed else float(point['pool_loss_pp'])
            rows.append(dict(domain=domain,seed=seed,deployment=deployment,target_head=f'D_alpha{alpha:g}',
                             target_test_mIoU=float(point['test_mIoU']),source_mIoU=source_miou-loss,
                             source_pooled_loss_pp=loss,source_T11_pp=0. if routed else float(point['T11_pp']),
                             source_worst_ROI_loss_pp=0. if routed else float(point['worst_ROI_loss_pp']),
                             extra_heads=1 if routed else 0,
                             extra_FP32_parameter_bytes=1316516*4 if routed else 0,
                             status='Counterfactual assembly of cached exact branch results, no new routing execution.',
                             assumption='Reliable Sentinel-2-vs-Landsat8 metadata; one specified target domain per deployment.'))
summary=[]
for domain in ['L8','SPARCS']:
    for deployment in ['shared_D_head','known_sensor_route_Source_D']:
        rr=[r for r in rows if r['domain']==domain and r['deployment']==deployment]
        assert len(rr)==2
        summary.append(dict(domain=domain,deployment=deployment,
                            **{k:sum(r[k] for r in rr)/2 for k in ['target_test_mIoU','source_mIoU','source_pooled_loss_pp','source_T11_pp','source_worst_ROI_loss_pp']},
                            extra_FP32_parameter_MiB=rr[0]['extra_FP32_parameter_bytes']/1024**2))
for domain in ['L8','SPARCS']:
    for seed in ['21701','21702']:
        a,b=[r for r in rows if r['domain']==domain and r['seed']==seed]
        assert a['target_test_mIoU']==b['target_test_mIoU']
        assert b['source_pooled_loss_pp']==b['source_T11_pp']==b['source_worst_ROI_loss_pp']==0

OUT.mkdir(parents=True,exist_ok=True)
save('PER_SEED_ROUTING_COMPARISON.csv',rows)
save('MEAN_ROUTING_COMPARISON.csv',summary)
specs=[dict(scenario='One source domain and one specified target domain',heads=2,extra_heads=1,extra_FP32_parameter_MiB=1316516*4/1024**2,routing='Sentinel-2 vs Landsat8 metadata',status='Structurally sufficient under stated input contract; no routed system benchmark.'),
       dict(scenario='Source plus separate L8-Biome and SPARCS heads',heads=3,extra_heads=2,extra_FP32_parameter_MiB=2*1316516*4/1024**2,routing='Sensor alone cannot select between the two Landsat8 heads',status='Dataset identity would be oracle routing unless operational provenance makes the distinction available.')]
save('STORAGE_AND_ROUTING_SCOPE.csv',specs)
for path,h in locks.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
audit=dict(date='2026-10-26',inputs=locks,input_hashes_unchanged=True,source_metadata_rows=535,source_ROIs=107,
           new_training=0,new_inference=0,remote_connections=0,new_intervals=0,
           source_head_identity='Source unchanged under frozen backbone and original preprocessing; zero loss is a conditional structural deduction.',
           storage_scope='Trainable FP32 head parameters only, excludes buffers/serialization/activation/runtime overhead.',
           latency='One backbone and one selected head per input; no switching, loading or end-to-end time benchmark performed.',
           selection='Existing source-budget B1 heads reused; no target-test-based selection. Historical source-val selection still disclosed.',
           unresolved=['No cross-evaluation of each Landsat head on both target pools in the audited current tables.',
                       'No concrete requirement forbidding metadata routing or an extra head has been established.',
                       'No claim for missing/corrupted metadata, unknown future domains or a global guarantee on all Sentinel-2 imagery.'])
(OUT/'AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='SHA256.json'},indent=2),encoding='utf-8')
print(json.dumps(dict(output=str(OUT),summary=summary,scenarios=specs),indent=2))

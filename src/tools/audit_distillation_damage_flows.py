"""Read-only local audit of aggregate ROI confusion matrices from 216/217."""
import csv, hashlib, json, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/research_audits/distillation_damage_flow_20261023'
OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[21601,21701,21702]; ALPHAS=[.25,.5,.75,1.]
LABELS=['Surface','Cloud','Shadow']; records={}; input_locks=[]; checks=0
def load(p):
    input_locks.append((p,hashlib.sha256(p.read_bytes()).hexdigest()))
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def metrics(cm):
    den=[sum(cm[j])+sum(r[j] for r in cm)-cm[j][j] for j in range(3)]
    assert all(d>0 for d in den)
    iou=[100*cm[j][j]/den[j] for j in range(3)]
    return iou,sum(iou)/3
def save(name,rr):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)

start=time.perf_counter()
for phase,default in [(216,21601),(217,None)]:
    for r in load(ROOT/f'outputs/phase{phase}/phase{phase}_20261023/PER_GROUP.csv'):
        if r['pool']=='source':
            records[(int(r.get('seed',default)),r['arm'],float(r['alpha']),r['group'])]=json.loads(r['confusion'])
rois=sorted(g for s,arm,a,g in records if s==21601 and arm=='C' and a==0)
assert len(rois)==107
all_rows=[];flows=[];sets={};summary=[]
for seed in SEEDS:
    for a in ALPHAS:
        worsened=set();crossed=set()
        for roi in rois:
            src=records[(seed,'C',0.,roi)]
            C=records[(seed,'C',a,roi)];D=records[(seed,'D',a,roi)]
            assert [sum(r) for r in src]==[sum(r) for r in C]==[sum(r) for r in D]
            si,sm=metrics(src);ci,cm=metrics(C);di,dm=metrics(D)
            cl=sm-cm;dl=sm-dm
            if dl-cl>1: worsened.add(roi)
            if cl<=5 and dl>5: crossed.add(roi)
            diffs=[(D[i][j]-C[i][j],f'{LABELS[i]}_to_{LABELS[j]}') for i in range(3) for j in range(3) if i!=j]
            dom=max(diffs)
            row=dict(seed=seed,alpha=a,roi=roi,Source_mIoU=sm,C_mIoU=cm,D_mIoU=dm,
                C_loss=cl,D_loss=dl,D_minus_C_mIoU=dm-cm,D_loss_worse_gt1=dl-cl>1,
                crossed_gt5=cl<=5 and dl>5,valid_pixels=sum(map(sum,src)),
                C_errors=sum(C[i][j] for i in range(3) for j in range(3) if i!=j),
                D_errors=sum(D[i][j] for i in range(3) for j in range(3) if i!=j),
                largest_net_error_increase_flow=dom[1],largest_net_error_increase_count=dom[0])
            for i,label in enumerate(LABELS):
                row[f'{label}_reference_pixels']=sum(src[i])
                row[f'{label}_Source_IoU']=si[i];row[f'{label}_C_IoU']=ci[i];row[f'{label}_D_IoU']=di[i]
                row[f'{label}_D_minus_C_IoU']=di[i]-ci[i]
            for i in range(3):
                for j in range(3):
                    if i==j: continue
                    n=sum(src[i]);delta=D[i][j]-C[i][j]
                    flows.append(dict(seed=seed,alpha=a,roi=roi,flow=f'{LABELS[i]}_to_{LABELS[j]}',
                        reference_pixels=n,Source_count=src[i][j],C_count=C[i][j],D_count=D[i][j],
                        D_minus_C_net_count=delta,D_minus_C_rate_pp=100*delta/n if n else None,
                        C_minus_Source_net_count=C[i][j]-src[i][j],D_minus_Source_net_count=D[i][j]-src[i][j]))
            # The sum of all off-diagonal differences must equal diagonal loss.
            assert sum(x[0] for x in diffs)==sum(C[i][i]-D[i][i] for i in range(3));checks+=1
            all_rows.append(row)
        sets[(seed,a,'worse')]=worsened;sets[(seed,a,'crossed')]=crossed
        summary.append(dict(seed=seed,alpha=a,worse_gt1_count=len(worsened),crossed_gt5_count=len(crossed),
            worse_gt1_rois=';'.join(sorted(worsened)),crossed_gt5_rois=';'.join(sorted(crossed))))

common=[]
for a in ALPHAS:
    for kind in ['worse','crossed']:
        ss=[sets[(s,a,kind)] for s in SEEDS]
        newcommon=ss[1]&ss[2];allcommon=set.intersection(*ss)
        common.append(dict(alpha=a,definition=kind,new_seed_intersection=';'.join(sorted(newcommon)),
            all_three_intersection=';'.join(sorted(allcommon)),all_three_union=';'.join(sorted(set.union(*ss)))))

focus=set.intersection(*(sets[(s,1.,'worse')] for s in SEEDS))
focus_rows=[r for r in all_rows if r['alpha']==1 and r['roi'] in focus]
focus_flows=[r for r in flows if r['alpha']==1 and r['roi'] in focus]
save('ALL_ROI_CLASS_METRICS.csv',all_rows);save('ALL_ROI_ERROR_FLOWS.csv',flows)
save('REPEATED_DAMAGE_SUMMARY.csv',summary);save('CROSS_SEED_OVERLAP.csv',common)
save('REPEATED_ENDPOINT_ROIS.csv',focus_rows);save('REPEATED_ENDPOINT_ERROR_FLOWS.csv',focus_flows)
for p,h in input_locks: assert hashlib.sha256(p.read_bytes()).hexdigest()==h
manifest=dict(inputs=[dict(path=str(p.relative_to(ROOT)),sha256=h) for p,h in input_locks],
    source_rois=107,seeds=SEEDS,checks=checks,seconds=time.perf_counter()-start,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    comparisons='D versus C at same alpha; crossed_gt5 is relative to Source in each arm',
    limitations=['Post-hoc ROI selection; no independent confirmation',
        'ROI identifiers are not land-cover labels', 'Aggregate net error changes do not identify pixelwise new errors',
        'Fixed Source/replay pool; seed repeats do not establish a physical mechanism'],
    no_remote=True,no_neural_calls=True,no_pixels=True)
(OUT/'AUDIT.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(output=str(OUT),endpoint_overlap=[r for r in common if r['alpha']==1],
    repeated_endpoint_rows=[{k:r[k] for k in ['seed','roi','C_loss','D_loss','D_minus_C_mIoU',
        'Surface_D_minus_C_IoU','Cloud_D_minus_C_IoU','Shadow_D_minus_C_IoU',
        'largest_net_error_increase_flow','largest_net_error_increase_count','C_errors','D_errors']} for r in focus_rows]),indent=2))

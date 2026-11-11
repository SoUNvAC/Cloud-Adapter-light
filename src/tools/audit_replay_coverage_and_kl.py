"""Metadata and existing per-step KL audit only; no raster or network access."""
import collections,csv,hashlib,json,math,re,statistics,time
from pathlib import Path
R=Path(__file__).resolve().parents[2]
O=R/'outputs/research_audits/replay_coverage_kl_20261023';O.mkdir(parents=True,exist_ok=True)
locks=[]
def text(p):
    b=p.read_bytes();locks.append((p,hashlib.sha256(b).hexdigest()));return b.decode('utf-8-sig')
def csvread(p):return list(csv.DictReader(text(p).splitlines()))
def save(name,rows):
    with (O/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def quantile(v,q):
    a=sorted(v);p=(len(a)-1)*q;i=int(p);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(p-i)
def tile(r):return r['s2_id'].split('_')[5]
def point(r):
    a=list(map(float,re.findall(r'-?\d+(?:\.\d+)?',r['proj_centroid'])))
    assert len(a)==2 and -180<=a[0]<=180 and -90<=a[1]<=90;return a
def distance(p,q):
    x,y=map(math.radians,p);xx,yy=map(math.radians,q)
    h=math.sin((yy-y)/2)**2+math.cos(y)*math.cos(yy)*math.sin((xx-x)/2)**2
    return 6371*2*math.asin(math.sqrt(min(1,h)))

start=time.perf_counter()
pool=json.loads(text(R/'outputs/phase216/phase216_20261023/SOURCE_POOL_REQUEST.json'))
selected=pool['selected'];assert len(selected)==256
byid={r['sample_id']:r for r in selected};assert len(byid)==256
val=csvread(R/'outputs/phase65/source_metadata/acquired/val_metadata.csv')
assert len(val)==535
assert locks[-1][1]==pool['metadata_SHA']['val']
valgroups=collections.defaultdict(list)
for r in val:valgroups[r['roi_id']].append(r)
assert len(valgroups)==107
focus={'ROI_0050','ROI_4813','ROI_1013','ROI_1404'}
coverage=[]
for roi,rr in sorted(valgroups.items()):
    roi_matches=[s for s in selected if s['roi_id']==roi]
    tile_matches=[s for s in selected if tile(s) in {tile(v) for v in rr}]
    product_matches=[s for s in selected if s['s2_id'] in {v['s2_id'] for v in rr}]
    nearest=min((distance(point(v),point(s)),s['sample_id'],s['roi_id']) for v in rr for s in selected)
    coverage.append(dict(roi=roi,focus=roi in focus,val_images=len(rr),same_replay_ROI_images=len(roi_matches),
        same_replay_tile_images=len(tile_matches),same_replay_product_images=len(product_matches),
        nearest_replay_centroid_km=nearest[0],nearest_replay_sample_id=nearest[1],nearest_replay_roi=nearest[2]))
save('VAL_ROI_REPLAY_COVERAGE.csv',coverage)

samples=[];steps=[];summaries=[];checks=0
paths={21601:R/'outputs/phase216/phase216_20261023/delivery/D_LOSSES.jsonl'}
paths.update({s:R/f'outputs/phase217/phase217_20261023/delivery/seed{s}/D_LOSSES.jsonl' for s in [21701,21702]})
source_sequences=[]
for seed,p in paths.items():
    rows=[json.loads(x) for x in text(p).splitlines() if x.strip()];assert len(rows)==4000
    assert [r['step'] for r in rows]==list(range(1,4001))
    seq=[r['source_sample_id'] for r in rows];source_sequences.append(seq)
    assert set(seq)==set(byid)
    grouped=collections.defaultdict(list)
    for r in rows:
        sid=r['source_sample_id'];k=float(r['source_KL']);tl=float(r['target_loss'])
        assert math.isfinite(k) and k>=-1e-7 and math.isfinite(tl)
        assert abs(float(r['total_loss'])-tl-k)<1e-4
        grouped[sid].append(r)
        steps.append(dict(seed=seed,step=r['step'],source_sample_id=sid,replay_roi=byid[sid]['roi_id'],
            KL=k,target_loss=tl,KL_over_target_loss=k/tl if tl>0 else None));checks+=1
    for sid,rr in sorted(grouped.items()):
        ks=[float(q['source_KL']) for q in rr]
        assert len(ks) in [15,16]
        samples.append(dict(seed=seed,sample_id=sid,roi=byid[sid]['roi_id'],observations=len(ks),
            first_observed_KL=ks[0],last_observed_KL=ks[-1],mean_observed_KL=statistics.fmean(ks),
            max_observed_KL=max(ks),sum_observed_KL=sum(ks),
            first_step=rr[0]['step'],last_step=rr[-1]['step']))
    sk=[float(q['source_KL']) for q in rows];own=[r for r in samples if r['seed']==seed]
    ranked=sorted(own,key=lambda r:r['sum_observed_KL'],reverse=True);total=sum(sk)
    summaries.append(dict(seed=seed,mean_KL=statistics.fmean(sk),max_KL=max(sk),
        first100_mean_KL=statistics.fmean(sk[:100]),last100_mean_KL=statistics.fmean(sk[-100:]),
        first_observation_mean=statistics.fmean(r['first_observed_KL'] for r in own),
        last_observation_mean=statistics.fmean(r['last_observed_KL'] for r in own),
        per_image_last_observed_KL_median=statistics.median(r['last_observed_KL'] for r in own),
        last_greater_than_first_images=sum(r['last_observed_KL']>r['first_observed_KL'] for r in own),
        top5_images_share_of_KL=sum(r['sum_observed_KL'] for r in ranked[:5])/total,
        top26_images_share_of_KL=sum(r['sum_observed_KL'] for r in ranked[:26])/total,
        top5_ids=';'.join(r['sample_id'] for r in ranked[:5])))
assert all(s==source_sequences[0] for s in source_sequences)
save('PER_IMAGE_KL.csv',samples);save('PER_STEP_KL.csv',steps);save('KL_SUMMARY.csv',summaries)
overview=dict(source_replay_images=256,source_replay_unique_ROIs=len({r['roi_id'] for r in selected}),
    val_ROIs=107,val_ROIs_with_same_replay_ROI=sum(r['same_replay_ROI_images']>0 for r in coverage),
    val_ROIs_with_same_replay_tile=sum(r['same_replay_tile_images']>0 for r in coverage),
    val_ROIs_with_same_replay_product=sum(r['same_replay_product_images']>0 for r in coverage),
    focus=[r for r in coverage if r['focus']],KL=summaries)
manifest=dict(inputs=[dict(path=str(p.relative_to(R)),sha256=h) for p,h in locks],
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=checks,
    seconds=time.perf_counter()-start,no_neural_calls=True,no_remote=True,no_pixels=True,
    limitations=['Centroid distance and tile/ROI identity are not appearance or semantic coverage',
    'Per-step losses are online observations under changing weights, not final-checkpoint replay evaluations',
    'KL magnitudes versus supervised loss do not determine gradient strength',
    'No validation-ROI KL or pixelwise teacher/student probability cache is present in these inputs'])
for p,h in locks:assert hashlib.sha256(p.read_bytes()).hexdigest()==h
(O/'AUDIT.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(O/'SUMMARY.json').write_text(json.dumps(overview,indent=2),encoding='utf-8')
print(json.dumps(overview,indent=2))

"""Local aggregate-only sensitivity audit. No models, pixels or remote I/O."""
import csv, hashlib, json, math, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/research_audits/source_selection_sensitivity_20261023'
OUT.mkdir(parents=True, exist_ok=True)
ALPHAS = [0., .25, .5, .75, 1.]
inputs = []
def read_csv(p):
    inputs.append((p, hashlib.sha256(p.read_bytes()).hexdigest()))
    with p.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def save_csv(name, rows):
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def miou(cm):
    tp=np.diagonal(cm, axis1=-2, axis2=-1)
    u=cm.sum(-1)+cm.sum(-2)-tp
    assert np.all(u>0)
    return (tp/u).mean(-1)*100
def scalar(cm):
    return sum(cm[j][j]/(sum(cm[j])+sum(r[j] for r in cm)-cm[j][j]) for j in range(3))*100/3

start=time.perf_counter(); groups={}; candidates={}
for phase,seed0 in [(216,21601),(217,None)]:
    base=ROOT/f'outputs/phase{phase}/phase{phase}_20261023'
    rows=read_csv(base/'PER_GROUP.csv')
    for r in rows:
        if r['pool']!='source': continue
        seed=int(r.get('seed',seed0)); key=(seed,r['arm'])
        groups.setdefault(key,{})[(float(r['alpha']),r['group'])]=r
    for r in read_csv(base/'ALL_CANDIDATES.csv'):
        candidates[(int(r.get('seed',seed0)),r['arm'],float(r['alpha']))]=r

models={}; base_roi=None; ids=None; checks=0
for key, gg in groups.items():
    gids=sorted(g for a,g in gg if a==0.)
    assert len(gids)==107 and len(gg)==535
    if ids is None: ids=gids
    assert gids==ids
    cm=np.array([[json.loads(gg[(a,g)]['confusion']) for g in ids] for a in ALPHAS],dtype=np.int64)
    assert (cm>=0).all() and np.all(cm.sum(-1)==cm[0].sum(-1))
    if base_roi is None: base_roi=cm[0].copy()
    assert np.array_equal(base_roi,cm[0])
    gm=miou(cm); losses=gm[0]-gm
    pooled=miou(cm[0].sum(0))-miou(cm.sum(1))
    tail=np.sort(losses,axis=1)[:,-11:].mean(1)
    for i,a in enumerate(ALPHAS):
        r=candidates[(*key,a)]
        assert abs(pooled[i]-float(r['pool_loss_pp']))<1e-9
        assert abs(tail[i]-float(r['T11_pp']))<1e-9
        for j in [0,53,106]:
            assert abs(gm[i,j]-scalar(cm[i,j].tolist()))<1e-10; checks+=1
    models[key]=(cm,losses,pooled,tail)

rng=np.random.default_rng(230217)
draws=rng.integers(0,107,(10000,107))
weights=np.zeros((10000,107),dtype=np.int64)
for j,idx in enumerate(draws): weights[j]=np.bincount(idx,minlength=107)
loo=np.ones((107,107),dtype=np.int64)-np.eye(107,dtype=np.int64)
selection_rows=[]; detail_rows=[]; boundary=[]; boot_choices={}

for key,(cm,losses,pool,tail) in sorted(models.items()):
    seed,arm=key
    for mode, ww in [('leave_one',loo),('bootstrap',weights)]:
        summed=(ww @ cm.transpose(1,0,2,3).reshape(107,45)).reshape(-1,5,3,3)
        vals=miou(summed); pp=vals[:,0,None]-vals
        if mode=='leave_one':
            tt=np.array([np.sort(np.delete(losses,j,axis=1),axis=1)[:,-11:].mean(1) for j in range(107)])
        else:
            tt=np.empty((10000,5))
            for i in range(5):
                sampled=losses[i][draws]
                tt[:,i]=np.partition(sampled,96,axis=1)[:,96:].mean(1)
            # Independent scalar reconstruction for first 12 resamples.
            for j in range(12):
                for i in range(5):
                    v=sorted(float(losses[i,k]) for k in draws[j])[-11:]
                    assert abs(sum(v)/11-tt[j,i])<1e-10
                    assert abs(scalar(summed[j,0].tolist())-scalar(summed[j,i].tolist())-pp[j,i])<1e-10
                    checks+=2
        for B in [1,3,5]:
            feasible=(pp<=B)&(tt<=5)
            assert feasible[:,0].all()
            chosen=4-np.argmax(feasible[:,::-1],axis=1)
            actual=max(i for i in range(5) if pool[i]<=B and tail[i]<=5)
            if mode=='bootstrap': boot_choices[(*key,B)]=chosen.copy()
            for i,a in enumerate(ALPHAS):
                selection_rows.append(dict(seed=seed,arm=arm,budget=B,mode=mode,alpha=a,
                    count=int((chosen==i).sum()),draws=len(chosen),frequency=float((chosen==i).mean()),
                    full_sample_alpha=ALPHAS[actual],changed=int((chosen!=actual).sum())))
            if mode=='leave_one':
                for j in range(107):
                    detail_rows.append(dict(seed=seed,arm=arm,budget=B,omitted_roi=ids[j],
                        alpha=ALPHAS[chosen[j]],changed=bool(chosen[j]!=actual)))
        if mode=='bootstrap' and arm=='D':
            boundary.append(dict(seed=seed,alpha=.5,T11=float(tail[2]),pool_loss=float(pool[2]),
                bootstrap_T11_q025=float(np.quantile(tt[:,2],.025)),bootstrap_T11_q975=float(np.quantile(tt[:,2],.975)),
                tail_pass_frequency=float((tt[:,2]<=5).mean()),pool_B1_pass_frequency=float((pp[:,2]<=1).mean()),
                joint_B1_pass_frequency=float(((tt[:,2]<=5)&(pp[:,2]<=1)).mean()),
                meaning='Empirical source-ROI resampling sensitivity, not population safety probability'))

damage=[]; transfer=[]; choice_pairs=[]
for seed in [21601,21701,21702]:
    lc=models[(seed,'C')][1]; ld=models[(seed,'D')][1]
    for i,a in enumerate(ALPHAS[1:],1):
        C=lc[i];D=ld[i];delta=D-C
        damage.append(dict(seed=seed,alpha=a,C_gt5=int((C>5).sum()),D_gt5=int((D>5).sum()),
            improved_below5=int(((C>5)&(D<=5)).sum()),new_D_gt5=int(((C<=5)&(D>5)).sum()),
            both_gt5=int(((C>5)&(D>5)).sum()),D_loss_lower_by_gt1=int((delta < -1).sum()),
            D_loss_higher_by_gt1=int((delta>1).sum()),worst_D_roi=ids[int(np.argmax(D))],
            worst_D_loss=float(D.max())))
        for j,g in enumerate(ids):
            transfer.append(dict(seed=seed,alpha=a,roi=g,C_loss=float(C[j]),D_loss=float(D[j]),
                D_minus_C_loss=float(delta[j]),new_D_gt5=bool(C[j]<=5 and D[j]>5)))
    for B in [1,3,5]:
        cc=boot_choices[(seed,'C',B)];dd=boot_choices[(seed,'D',B)]
        for i,a in enumerate(ALPHAS):
            for j,b in enumerate(ALPHAS):
                n=int(((cc==i)&(dd==j)).sum())
                if n: choice_pairs.append(dict(seed=seed,budget=B,C_alpha=a,D_alpha=b,count=n,frequency=n/10000))

save_csv('SELECTION_FREQUENCIES.csv',selection_rows)
save_csv('LEAVE_ONE_CHOICES.csv',detail_rows)
save_csv('D_ALPHA05_BOUNDARY.csv',boundary)
save_csv('DAMAGE_TRANSFER_SUMMARY.csv',damage)
save_csv('ROI_DAMAGE_CHANGES.csv',transfer)
save_csv('PAIRED_SELECTION_FREQUENCIES.csv',choice_pairs)
for p,h in inputs: assert hashlib.sha256(p.read_bytes()).hexdigest()==h
manifest=dict(inputs=[dict(path=str(p.relative_to(ROOT)),sha256=h) for p,h in inputs],
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=checks,
    source_groups=107,draws=10000,random_seed=230217,seconds=time.perf_counter()-start,
    no_remote=True,no_neural_calls=True,no_pixels=True,
    scope='Historical exploratory source-ROI deletion/resampling sensitivity; no new alpha selection policy or test confirmation',
    limitations=['Fixed training seeds, replay pool and Source', '107 ROIs are not certified independent geographic groups',
                 'Empirical resampling frequencies are not deployment guarantees', 'Paired confusion counts do not identify pixelwise corrected/new errors'])
(OUT/'AUDIT.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(output=str(OUT),boundary=boundary,endpoint_damage=[r for r in damage if r['alpha']==1],checks=checks,seconds=manifest['seconds']),indent=2))

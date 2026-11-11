"""Freeze image weights from existing SPARCS train logs only; no model/pixel access."""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'src/research_plans/phase226_frozen_20261025'
locks={};scores={};counts={}
for seed in ['21701','21702']:
    p=ROOT/f'outputs/phase217/phase217_20261023/delivery/seed{seed}/D_LOSSES.jsonl'
    locks[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    rows=[json.loads(s) for s in p.read_text().splitlines()]
    assert len(rows)==4000 and [r['step'] for r in rows]==list(range(1,4001))
    values=collections.defaultdict(list)
    for r in rows:values[str(r['source_sample_id'])].append(float(r['source_KL']))
    assert len(values)==256
    counts[seed]={k:len(v) for k,v in values.items()}
    scores[seed]={k:sum(v)/len(v) for k,v in values.items()}
assert counts['21701']==counts['21702']
frequency=counts['21701'];assert set(frequency.values())=={15,16}
score={k:(scores['21701'][k]+scores['21702'][k])/2 for k in frequency}
ranked=sorted(score,key=lambda k:(-score[k],k));top=set(ranked[:64])
raw={k:2. if k in top else 2./3 for k in frequency}
norm=sum(frequency[k]*raw[k] for k in frequency)/4000
R={k:raw[k]/norm for k in frequency}
Q={};mapped={}
for n in [15,16]:
    ids=sorted(k for k in frequency if frequency[k]==n)
    perm=sorted(ids,key=lambda k:(hashlib.sha256(f'phase225|{n}|{k}'.encode()).digest(),k))
    for dst,src in zip(ids,perm):Q[dst]=R[src];mapped[dst]=src
    assert sorted(R[k] for k in ids)==sorted(Q[k] for k in ids)
assert abs(sum(frequency[k]*R[k] for k in frequency)/4000-1)<1e-12
assert abs(sum(frequency[k]*Q[k] for k in frequency)/4000-1)<1e-12
p=ROOT/'outputs/phase220/phase220_20261024/PER_IMAGE_CLASS_WEIGHT_EXPOSURE.csv'
locks[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
with p.open(encoding='utf-8-sig',newline='') as f:
    exposure=[r for r in csv.DictReader(f) if r['pool']=='replay256' and r['basis']=='teacher_prediction_geometry']
classrows=[]
for label in ['Surface','Cloud','Shadow']:
    rr=[r for r in exposure if r['class']==label]
    assert len(rr)==256 and {r['sample_id'] for r in rr}==set(frequency)
    total=sum(int(r['support'])*frequency[r['sample_id']] for r in rr)
    classrows.append(dict(teacher_prediction_class=label,exposure_pixels=total,
                          R_mean_weight=sum(int(r['support'])*frequency[r['sample_id']]*R[r['sample_id']] for r in rr)/total,
                          Q_mean_weight=sum(int(r['support'])*frequency[r['sample_id']]*Q[r['sample_id']] for r in rr)/total))
OUT.mkdir(parents=True,exist_ok=True)
rows=[dict(sample_id=k,rank=i+1,mean_online_KL_seed21701=scores['21701'][k],mean_online_KL_seed21702=scores['21702'][k],
           score_mean=score[k],repeat_count=frequency[k],top64=k in top,R_weight=R[k],Q_weight=Q[k],Q_weight_source_id=mapped[k]) for i,k in enumerate(ranked)]
def save(name,rr):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
save('REPLAY_WEIGHTS.csv',rows)
save('TEACHER_CLASS_WEIGHT_EXPOSURE.csv',classrows)
lock=dict(inputs=locks,weight_normalizer=norm,weight_rule='top64=2; other192=2/3; divide by 4000-step exposure-weighted mean',
          Q_rule='Hash permutation within repeat15/repeat16 stratum, lexicographic sample IDs as destination order.',
          historical_training_required='Two SPARCS D 4000-step runs in Phase217; weights are not training-free.',
          labels_read=False,model_calls=0,raw_source_pixels_read=False,
          limitation='Source KL is teacher disagreement, not a reference error/damage score; class distribution can change.')
(OUT/'WEIGHT_LOCK.json').write_text(json.dumps(lock,indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='SHA256.json'},indent=2),encoding='utf-8')
print(json.dumps(dict(output=str(OUT),normalizer=norm,class_exposure=classrows,top64_weight=R[ranked[0]],other_weight=R[ranked[-1]]),indent=2))

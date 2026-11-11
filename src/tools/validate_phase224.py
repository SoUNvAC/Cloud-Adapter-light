"""Independent H-D all-point metrics, paired resampling and four-axis acceptance."""
from pathlib import Path
import csv,json,hashlib,collections,time,functools
import numpy as np
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase224/phase224_20261024';D=O/'delivery';E=D/'evaluation';SS=[21701,21702];AA=[0,.25,.5,.75,1];CS=['Surface','Cloud','Shadow']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(v) for v in p.read_text().splitlines()]
def rows(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def score(c):
 c=np.asarray(c,np.float64);p=np.diagonal(c,axis1=-2,axis2=-1);u=c.sum(-2)+c.sum(-1)-p;return np.nanmean(np.divide(p,u,out=np.full(p.shape,np.nan),where=u>0),axis=-1)*100
def main():
 clock=time.monotonic();m=js(D/'DELIVERY_MANIFEST.json');assert all((D/n).stat().st_size==v['bytes'] and sha(D/n)==v['SHA'] for n,v in m.items());assert js(D/'REMOTE_CPU_ACCEPTANCE.json')['status']=='passed';complete=js(D/'COMPLETE.json');assert complete['units']==100460 and complete['GPU_seconds']<21600;lock=js(D/'EXPERIMENT_LOCK.json');strata=js(D/'L8_SUPPORT_ACCEPTED.json')['test_strata'];cases={};raw={};stats={};selected={};checks=0
 for target in ['SPARCS','L8']:
  base={p:js(D/f'historical/{target}/evaluation/Source__{p}.json') for p in ['source','val','test']};assert [len(base[p]['per_group']) for p in ['source','val','test']]==([107,10,10] if target=='SPARCS' else [107,8,16])
  for s in SS:
   for arm in ['C','D','H']:
    for a in AA:
     for pool in ['source','val','test']:
      v=base[pool] if a==0 else js(E/f'{target}_{s}_H_a{a}__{pool}.json') if arm=='H' else js(D/f'historical/{target}/evaluation/{s}_{arm}_a{a}__{pool}.json');g={k:np.asarray(c,np.int64) for k,c in v['per_group'].items()};assert set(g)==set(base[pool]['per_group']);aggregate=collections.defaultdict(lambda:np.zeros((3,3),np.int64))
      for q in v['per_input']:aggregate[q.get('scene',q.get('group'))]+=np.asarray(q['confusion'],np.int64)
      assert set(g)==set(aggregate) and all(np.array_equal(c,aggregate[k]) for k,c in g.items()) and sum(g.values()).tolist()==v['confusion'];assert all(np.array_equal(c.sum(1),np.asarray(base[pool]['per_group'][k]).sum(1)) for k,c in g.items());cases[target,s,arm,a,pool]=g;raw[target,s,arm,a,pool]=v
     gg=cases[target,s,arm,a,'source'];losses=sorted([score(base['source']['per_group'][g])-score(c) for g,c in gg.items()],reverse=True);stats[target,s,arm,a]={'pool_loss_pp':float(score(base['source']['confusion'])-score(sum(gg.values()))),'T11_pp':float(np.mean(losses[:11])),'worst_ROI_loss_pp':float(losses[0]),'ROI_loss_gt5':sum(v>5 for v in losses)}
    for b in [1,3,5]:selected[target,s,arm,b]=max(a for a in AA if stats[target,s,arm,a]['pool_loss_pp']<=b and stats[target,s,arm,a]['T11_pp']<=5)
 for v in rows('ALL_POOL_METRICS.csv')+rows('PER_GROUP.csv'):
  t,s,a,f,p=v['target'],int(v['seed']),v['arm'],float(v['alpha']),v['pool'];g=cases[t,s,a,f,p];c=g[v['group']] if 'group' in v else sum(g[k] for k in (sorted(g) if v['support']=='full' else strata[v['support']]));assert c.tolist()==json.loads(v['confusion']);assert abs(score(c)-float(v['mIoU']))<1e-10;tp=np.diag(c);u=c.sum(0)+c.sum(1)-tp;assert int(v['effective_IoU_classes'])==(u>0).sum()
  for i,cl in enumerate(CS):assert int(v[cl+'_reference_pixels'])==c[i].sum() and int(v[cl+'_prediction_pixels'])==c[:,i].sum() and int(v[cl+'_IoU_union_pixels'])==u[i];assert (not v[cl+'_IoU'] and u[i]==0) or abs(float(v[cl+'_IoU'])-100*tp[i]/u[i])<1e-10
  for key,n,den in [('Shadow_recall',c[2,2],c[2].sum()),('nonShadow_to_Shadow_FPR',c[:2,2].sum(),c[:2].sum())]:assert (not v[key] and den==0) or abs(float(v[key])-n/den)<1e-12
  checks+=1
 for v in rows('TABLE_C_ALL_CANDIDATES.csv'):
  st=stats[v['target'],int(v['seed']),v['arm'],float(v['alpha'])]
  for k,x in st.items():assert abs(float(v[k])-x)<1e-10
 for v in rows('JOINT_RULE.csv'):assert float(v['alpha'])==selected[v['target'],int(v['seed']),v['arm'],int(v['budget_pp'])]
 for job,meta in lock['jobs'].items():
  cfg=js(D/job/'TRAINING_CONFIG_LOCK.json');old=js(D/job/'HISTORICAL_TRAINING_CONFIG_LOCK.json');old['KL']['lambda']=4;assert cfg==old;assert sha(D/job/'TARGET_SOURCE_SCHEDULE.json')==meta['original_schedule_SHA'];g=js(D/job/'GATE_ACCEPTED.json');assert g['target_loss_H']==g['historical_target_loss_D'] and g['lambda']==4 and abs(g['initial_KL'])<1e-6 and g['gate_updates']==0 and all(n.startswith('decode_head.') for n in g['grad_H']);assert sha(D/job/'heads/H_iter4000.pth')==js(D/job/'H_TRAIN_COMPLETE.json')['head_SHA']
  losses=lines(D/job/'H_LOSSES.jsonl');schedule=js(D/job/'TARGET_SOURCE_SCHEDULE.json');assert len(losses)==4000
  for i,v in enumerate(losses):assert v['step']==i+1 and v['target_input_SHA']==schedule[i]['image_SHA'] and v['finite_gradients'] and v['lambda']==4 and np.isfinite(v['total_loss']) and abs(v['source_KL_scaled4']-4*v['source_KL'])<1e-10
 # Explicit all81 cells and independent official per-input margins.
 by={};source={v['name']:v for v in raw['SPARCS',21701,'D',0,'source']['per_input']};joint=collections.defaultdict(lambda:np.zeros((3,3,3,3),np.int64));cell_count=0
 for t in ['SPARCS','L8']:
  for s in SS:
   for arm in ['D','H']:by[t,s,arm]={v['name']:v for v in raw[t,s,arm,1,'source']['per_input']}
 for v in lines(E/'JOINT_COUNTS.jsonl'):
  j=np.asarray(v['J'],np.int64);assert j.shape==(3,3,3,3) and (j>=0).all() and j.sum()==v['valid_pixels'];cm={k:np.zeros((3,3),np.int64) for k in ['teacher','D','H']}
  for r in range(3):
   for t in range(3):
    for d in range(3):
     for h in range(3):n=int(j[r,t,d,h]);cm['teacher'][r,t]+=n;cm['D'][r,d]+=n;cm['H'][r,h]+=n;cell_count+=1
  assert cm['teacher'].tolist()==source[v['name']]['confusion']
  for arm in ['D','H']:
   old=by[v['target'],v['seed'],arm][v['name']];assert cm[arm].tolist()==old['confusion']
   if 'prediction_SHA' in old:assert old['prediction_SHA']==v[arm+'_prediction_SHA']
  joint[v['target'],v['seed'],v['ROI']]+=j
 assert cell_count==173340
 for v in rows('TEACHER_CORRECTNESS_ALPHA1.csv'):
  t,s,g=v['target'],int(v['seed']),v['ROI'];ids=sorted(cases[t,s,'D',0,'source']);j=sum(joint[t,s,z] for z in ids) if g=='__pooled__' else joint[t,s,g];lesswrong=lesscorrect=net=0
  for r in range(3):
   for teacher in range(3):
    for d in range(3):
     for h in range(3):n=int(j[r,teacher,d,h]);net+=n*((h!=r)-(d!=r));lesswrong+=n*((d!=r)-(h!=r))*(teacher==r);lesscorrect+=n*((d==r)-(h==r))*(teacher!=r)
  assert net==lesscorrect-lesswrong and int(v['teacher_correct_H_fewer_errors'])==lesswrong and int(v['teacher_wrong_H_fewer_corrections'])==lesscorrect and int(v['H_minus_D_errors'])==net
 for v in rows('DIRECT_D_TO_H_ALPHA1.csv'):
  t,s,g=v['target'],int(v['seed']),v['ROI'];ids=sorted(cases[t,s,'D',0,'source']);j=sum(joint[t,s,z] for z in ids) if g=='__pooled__' else joint[t,s,g];r=CS.index(v['reference_class']);z=collections.Counter()
  for teacher in range(3):
   if v['teacher_correctness']=='correct' and teacher!=r or v['teacher_correctness']=='wrong' and teacher==r:continue
   for d in range(3):
    for h in range(3):n=int(j[r,teacher,d,h]);z['pixels']+=n;z['D_wrong_H_right']+=n*(d!=r and h==r);z['D_right_H_wrong']+=n*(d==r and h!=r);z['H_minus_D_errors']+=n*((h!=r)-(d!=r))
  assert all(int(v[k])==n for k,n in z.items())
 # Full independent10000 histogram reconstruction, fixed alpha; no mean confusion across seeds.
 cis=rows('PAIRED_CI.csv');proof=js(O/'BOOTSTRAP_RULE.json');first=js(O/'BOOTSTRAP_FIRST50.json');bc=0
 for t in ['SPARCS','L8']:
  sourceids=sorted(cases[t,21701,'D',0,'source']);support={'source':sourceids,'full':sorted(cases[t,21701,'D',0,'test'])}
  if t=='L8':support.update(strata)
  rng=np.random.default_rng(217023);ix={k:rng.integers(0,len(gs),(10000,len(gs))) for k,gs in support.items()};hist={k:np.stack([np.bincount(row,minlength=len(support[k])) for row in vv]) for k,vv in ix.items()};base=np.array([score(cases[t,21701,'D',0,'source'][g]) for g in sourceids])
  for k,v in ix.items():assert hashlib.sha256(v.astype('<i8').tobytes()).hexdigest()==proof['targets'][t]['index_SHA'][k]
  for context in ['endpoint_alpha1','B1','B3','B5']:
   draws={};pts={}
   for s in SS:
    ha,da=(1,1) if context=='endpoint_alpha1' else (selected[t,s,'H',int(context[1:])],selected[t,s,'D',int(context[1:])]);dd={};pp={}
    for k,gs in support.items():
     pool='source' if k=='source' else 'test';h=np.stack([cases[t,s,'H',ha,pool][g] for g in gs]);d=np.stack([cases[t,s,'D',da,pool][g] for g in gs]);hb=(hist[k]@h.reshape(len(gs),9)).reshape(-1,3,3);db=(hist[k]@d.reshape(len(gs),9)).reshape(-1,3,3)
     if k=='source':
      dd['source_pool_loss_diff_pp']=score(db)-score(hb);pp['source_pool_loss_diff_pp']=float(score(d.sum(0))-score(h.sum(0)));hl=base-score(h);dl=base-score(d);dd['T11_diff_pp']=np.sort(np.take(hl,ix[k]),axis=1)[:,-11:].mean(1)-np.sort(np.take(dl,ix[k]),axis=1)[:,-11:].mean(1);pp['T11_diff_pp']=float(np.sort(hl)[-11:].mean()-np.sort(dl)[-11:].mean())
     else:name='test_mIoU_diff_pp' if k=='full' else f'test_{k}_mIoU_diff_pp';dd[name]=score(hb)-score(db);pp[name]=float(score(h.sum(0))-score(d.sum(0)))
    draws[str(s)]=dd;pts[str(s)]=pp
   draws['mean_of_two']={k:(draws[str(SS[0])][k]+draws[str(SS[1])][k])/2 for k in draws[str(SS[0])]};pts['mean_of_two']={k:(pts[str(SS[0])][k]+pts[str(SS[1])][k])/2 for k in pts[str(SS[0])]}
   for scope,ds in draws.items():
    for k,dd in ds.items():
     row=next(v for v in cis if v['target']==t and v['context']==context and v['scope']==scope and v['metric']==k);lo,hi=np.quantile(dd,[.025,.975]);assert abs(float(row['point'])-pts[scope][k])<1e-10 and abs(float(row['CI95_low'])-lo)<1e-10 and abs(float(row['CI95_high'])-hi)<1e-10;assert np.allclose(dd[:50],first[t+'/'+context+'/'+scope+'/'+k],rtol=0,atol=1e-10);bc+=1
 assert len(rows('FINITE_25_PAIRS_ALL.csv'))==100
 for v in rows('FINITE_25_PAIRS_ALL.csv'):
  t,s,ha,da=v['target'],int(v['seed']),float(v['H_alpha']),float(v['D_alpha']);x=float(v['test_mIoU_diff_pp']);y=float(v['source_pool_loss_diff_pp']);z=float(v['T11_diff_pp']);assert abs(x-(score(sum(cases[t,s,'H',ha,'test'].values()))-score(sum(cases[t,s,'D',da,'test'].values()))))<1e-10 and abs(y-(stats[t,s,'H',ha]['pool_loss_pp']-stats[t,s,'D',da]['pool_loss_pp']))<1e-10 and abs(z-(stats[t,s,'H',ha]['T11_pp']-stats[t,s,'D',da]['T11_pp']))<1e-10;assert (v['H_dominates_D']=='True')==(x>=0 and y<=0 and z<=0 and (x>0 or y<0 or z<0))
 result={'status':'passed','raw_files_SHA_checked':len(m),'all_group_and_pool_metric_rows_checked':checks,'all10000_paired_bootstrap_checks':bc,'explicit_four_axis_cells':cell_count,'all16000_training_steps_and_original_SHA':True,'only_lambda_changed':True,'frozen_gate_and_source_rules':True,'all100_finite_pairs_preserved':True,'new_model_calls':0,'seconds':time.monotonic()-clock,'code_SHA':sha(Path(__file__))};(O/'INDEPENDENT_CPU_ACCEPTANCE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()

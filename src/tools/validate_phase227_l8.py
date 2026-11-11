"""Independent scalar-cell, histogram bootstrap, weight-use and table acceptance."""
from pathlib import Path
import json,csv,hashlib,collections,itertools,ast,time
import numpy as np
R=next(p for p in Path(__file__).resolve().parents if (p/'src/tools').is_dir());O=R/'outputs/phase227/phase227_20261025';D=O/'delivery';E=D/'evaluation';SS=[21701,21702];AA=[0,.25,.5,.75,1];ARMS=['C','D','H','W1','W10'];CS=['Surface','Cloud','Shadow']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(v) for v in p.read_text(encoding='utf-8-sig').splitlines()]
def rows(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def score(c):
 c=np.asarray(c,np.float64);tp=np.diagonal(c,axis1=-2,axis2=-1);u=c.sum(-2)+c.sum(-1)-tp;return np.nanmean(np.divide(tp,u,out=np.full(tp.shape,np.nan),where=u>0),axis=-1)*100
def main():
 clock=time.monotonic()
 strata=js(D/'INPUT_PREFLIGHT_ACCEPTED.json')['test_strata'];base={p:js(D/f'historical/evaluation/Source__{p}.json') for p in ['source','val','test']};assert [len(base[p]['per_group']) for p in ['source','val','test']]==[107,8,16];cases={};raw={};st={};selected={}
 for s in SS:
  for a in ARMS:
   for f in AA:
    for p in ['source','val','test']:
     v=base[p] if f==0 else js((E if a in ['W1','W10'] else D/'historical/evaluation')/f'{s}_{a}_a{f}__{p}.json');g={k:np.asarray(c,np.int64) for k,c in v['per_group'].items()};agg=collections.defaultdict(lambda:np.zeros((3,3),np.int64))
     for z in v['per_input']:agg[z.get('scene',z.get('group'))]+=np.asarray(z['confusion'],np.int64)
     assert set(g)==set(base[p]['per_group'])==set(agg) and sum(g.values()).tolist()==v['confusion'] and all(np.array_equal(c,agg[k]) and np.array_equal(c.sum(1),np.asarray(base[p]['per_group'][k]).sum(1)) for k,c in g.items());cases[s,a,f,p]=g;raw[s,a,f,p]=v
    g=cases[s,a,f,'source'];loss=sorted([float(score(base['source']['per_group'][k])-score(c)) for k,c in g.items()],reverse=True);st[s,a,f]={'pool_loss_pp':float(score(base['source']['confusion'])-score(sum(g.values()))),'T11_pp':float(np.mean(loss[:11])),'worst_ROI_loss_pp':loss[0],'ROI_loss_gt5':sum(v>5 for v in loss)}
   for b in [1,3,5]:selected[s,a,b]=max(f for f in AA if st[s,a,f]['pool_loss_pp']<=b and st[s,a,f]['T11_pp']<=5)
 checks=0
 for v in rows('ALL_POOL_METRICS.csv')+rows('PER_GROUP.csv'):
  g=cases[int(v['seed']),v['arm'],float(v['alpha']),v['pool']];c=g[v['group']] if 'group' in v else sum(g[k] for k in (sorted(g) if v['support']=='full' else strata[v['support']]));assert c.tolist()==json.loads(v['confusion']) and abs(score(c)-float(v['mIoU']))<1e-10;u=c.sum(0)+c.sum(1)-np.diag(c);assert int(v['effective_IoU_classes'])==(u>0).sum()
  for i,cl in enumerate(CS):assert int(v[cl+'_reference_pixels'])==c[i].sum() and int(v[cl+'_prediction_pixels'])==c[:,i].sum() and int(v[cl+'_IoU_union_pixels'])==u[i];assert (not v[cl+'_IoU'] and u[i]==0) or abs(float(v[cl+'_IoU'])-100*c[i,i]/u[i])<1e-10
  for k,n,de in [('Shadow_recall',c[2,2],c[2].sum()),('nonShadow_to_Shadow_FPR',c[:2,2].sum(),c[:2].sum())]:assert (not v[k] and de==0) or abs(float(v[k])-n/de)<1e-12
  checks+=1
 for v in rows('TABLE_B_ALL_CANDIDATES.csv'):
  for k,x in st[int(v['seed']),v['arm'],float(v['alpha'])].items():assert abs(float(v[k])-x)<1e-10
 for v in rows('SOURCE_RULE_ALL_ARMS.csv'):assert float(v['alpha'])==selected[int(v['seed']),v['arm'],int(v['budget_pp'])]
 ids=sorted(base['source']['per_group']);supports={'source':ids,'full':sorted(base['test']['per_group']),'yes':strata['yes'],'no':strata['no']};rng=np.random.default_rng(217023);ix={k:rng.integers(0,len(g),(10000,len(g))) for k,g in supports.items()};hist={k:np.stack([np.bincount(v,minlength=len(supports[k])) for v in z]) for k,z in ix.items()};proof=js(O/'BOOTSTRAP_RULE.json');cis=rows('PAIRED_CI.csv');first=js(O/'BOOTSTRAP_FIRST50.json');bm=np.array([score(base['source']['per_group'][g]) for g in ids]);bc=0
 for k,v in ix.items():assert hashlib.sha256(v.astype('<i8').tobytes()).hexdigest()==proof['index_SHA'][k]
 for candidatearm,other in [('W1','D'),('W10','D'),('W1','H'),('W10','H')]:
  for context in ['alpha025','alpha1','B1','B3','B5']:
   draws={};points={}
   for s in SS:
    x,y=(.25,.25) if context=='alpha025' else (1,1) if context=='alpha1' else (selected[s,candidatearm,int(context[1:])],selected[s,other,int(context[1:])]);dd={};pt={}
    for k,groups in supports.items():
     p='source' if k=='source' else 'test';g=np.stack([cases[s,candidatearm,x,p][z] for z in groups]);h=np.stack([cases[s,other,y,p][z] for z in groups]);gb=(hist[k]@g.reshape(len(groups),9)).reshape(-1,3,3);hb=(hist[k]@h.reshape(len(groups),9)).reshape(-1,3,3)
     if k=='source':
      dd['source_pool_loss_diff_pp']=score(hb)-score(gb);pt['source_pool_loss_diff_pp']=float(score(h.sum(0))-score(g.sum(0)));gl=bm-score(g);hl=bm-score(h);dd['T11_diff_pp']=np.sort(np.take(gl,ix[k]),axis=1)[:,-11:].mean(1)-np.sort(np.take(hl,ix[k]),axis=1)[:,-11:].mean(1);pt['T11_diff_pp']=float(np.sort(gl)[-11:].mean()-np.sort(hl)[-11:].mean())
     else:
      key='test_mIoU_diff_pp' if k=='full' else f'test_{k}_mIoU_diff_pp';dd[key]=score(gb)-score(hb);pt[key]=float(score(g.sum(0))-score(h.sum(0)))
    draws[str(s)]=dd;points[str(s)]=pt
   draws['mean_of_two']={k:(draws[str(SS[0])][k]+draws[str(SS[1])][k])/2 for k in draws[str(SS[0])]};points['mean_of_two']={k:(points[str(SS[0])][k]+points[str(SS[1])][k])/2 for k in points[str(SS[0])]}
   for scope,dd in draws.items():
    for k,z in dd.items():
     row=next(v for v in cis if v['comparison']==candidatearm+'-'+other and v['context']==context and v['scope']==scope and v['metric']==k);lo,hi=np.quantile(z,[.025,.975]);assert abs(float(row['point'])-points[scope][k])<1e-10 and abs(float(row['CI95_low'])-lo)<1e-10 and abs(float(row['CI95_high'])-hi)<1e-10 and np.allclose(z[:50],first['/'.join([candidatearm+'-'+other,context,scope,k])],rtol=0,atol=1e-10);bc+=1
 joint=collections.defaultdict(lambda:np.zeros((3,3,3,3,3),np.int64));count=0;by={(s,a):{v['name']:v for v in raw[s,a,1,'source']['per_input']} for s in SS for a in ['D','W1','W10']};teacher={v['name']:v for v in base['source']['per_input']}
 for v in lines(E/'JOINT_COUNTS.jsonl'):
  j=np.asarray(v['J'],np.int64);assert j.shape==(3,3,3,3,3) and (j>=0).all() and j.sum()==v['valid_pixels'];cm={k:np.zeros((3,3),np.int64) for k in ['teacher','D','W1','W10']}
  for r,t,d,a,q in itertools.product(range(3),repeat=5):
   n=int(j[r,t,d,a,q]);cm['teacher'][r,t]+=n;cm['D'][r,d]+=n;cm['W1'][r,a]+=n;cm['W10'][r,q]+=n;count+=1
  assert cm['teacher'].tolist()==teacher[v['name']]['confusion']
  for a in ['D','W1','W10']:
   ref=by[v['seed'],a][v['name']];assert cm[a].tolist()==ref['confusion']
   if 'prediction_SHA' in ref:assert ref['prediction_SHA']==v[a+'_prediction_SHA']
  joint[v['seed'],v['ROI']]+=j
 assert count==260010
 for v in rows('TABLE_C_DIRECT_ALPHA1.csv'):
  s=int(v['seed']);j=sum(joint[s,g] for g in ids) if v['ROI']=='__pooled__' else joint[s,v['ROI']];r=CS.index(v['reference_class']);oi=['D','W1','W10'].index(v['old_arm']);ni=['D','W1','W10'].index(v['new_arm']);c=collections.Counter()
  for t,d,a,q in itertools.product(range(3),repeat=4):
   if v['teacher_correctness']=='correct' and t!=r or v['teacher_correctness']=='wrong' and t==r:continue
   pp=[d,a,q];n=int(j[r,t,d,a,q]);c['pixels']+=n;c['old_wrong_new_right']+=n*(pp[oi]!=r and pp[ni]==r);c['old_right_new_wrong']+=n*(pp[oi]==r and pp[ni]!=r);c['new_minus_old_errors']+=n*((pp[ni]!=r)-(pp[oi]!=r))
  assert all(int(v[k])==x for k,x in c.items())
 direct_rows=rows('TABLE_C_DIRECT_ALPHA1.csv')
 for v in rows('TEACHER_CORRECTNESS_ALPHA1.csv'):
  rr=[z for z in direct_rows if all(z[k]==v[k] for k in ['seed','ROI','old_arm','new_arm'])];tc=sum(int(z['new_minus_old_errors']) for z in rr if z['teacher_correctness']=='correct');tw=sum(int(z['new_minus_old_errors']) for z in rr if z['teacher_correctness']=='wrong');assert -tc==int(v['teacher_correct_new_fewer_errors']) and tw==int(v['teacher_wrong_new_fewer_corrections']) and tc+tw==int(v['new_minus_old_errors'])
 finite=rows('FINITE_ALL200_PAIRS.csv');assert len(finite)==200
 for v in finite:
  s,a,f,b,h=int(v['seed']),v['new_arm'],float(v['new_alpha']),v['old_arm'],float(v['old_alpha']);x=float(score(sum(cases[s,a,f,'test'].values()))-score(sum(cases[s,b,h,'test'].values())));y=st[s,a,f]['pool_loss_pp']-st[s,b,h]['pool_loss_pp'];z=st[s,a,f]['T11_pp']-st[s,b,h]['T11_pp'];assert abs(x-float(v['test_diff_pp']))<1e-10 and abs(y-float(v['source_loss_diff_pp']))<1e-10 and abs(z-float(v['T11_diff_pp']))<1e-10;assert (v['new_dominates_old']=='True')==(x>=0 and y<=0 and z<=0 and (x>0 or y<0 or z<0)) and (v['old_weakly_covers_new']=='True')==(x<=0 and y>=0 and z>=0)
 result={'status':'passed','raw_manifest_checked_in_parent':True,'metric_rows':checks,'full10000_bootstrap_checks':bc,'explicit_five_axis_cells':count,'finite_pairs':len(finite),'new_model_calls':0,'seconds':time.monotonic()-clock,'code_SHA':sha(Path(__file__))};(O/'INDEPENDENT_CPU_ACCEPTANCE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()

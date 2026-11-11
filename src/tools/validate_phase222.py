"""Independent integer margins, source rules, all paired intervals and raw SHA audit."""
from pathlib import Path
import csv,json,hashlib,collections,time
import numpy as np
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase222/phase222_20261024';D=O/'delivery';E=D/'evaluation';SS=[21701,21702];AA=[0,.25,.5,.75,1];CS=['Surface','Cloud','Shadow']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def score(c):
 c=np.asarray(c,np.float64);d=np.diagonal(c,axis1=-2,axis2=-1);u=c.sum(-2)+c.sum(-1)-d;v=np.divide(d,u,out=np.full(d.shape,np.nan),where=u>0);return np.nanmean(v,axis=-1)*100
def main():
 clock=time.monotonic();m=js(D/'DELIVERY_MANIFEST.json');assert all((D/n).stat().st_size==v['bytes'] and sha(D/n)==v['SHA'] for n,v in m.items());complete=js(D/'COMPLETE.json');assert complete['units']==99190 and complete['GPU_seconds']<21600;remote=js(D/'REMOTE_CPU_ACCEPTANCE.json');assert remote['status']=='passed' and remote['total_units']==99190
 pre=js(D/'INPUT_PREFLIGHT_ACCEPTED.json');assert [pre['support'][p]['inputs'] for p in ['train','val','test']]==[1560,963,2167];assert pre['mapping_L8']==[0,2,1,1];strata=pre['test_strata'];assert list(map(len,[strata['yes'],strata['no']]))==[6,10]
 baseline={p:js(E/f'Source__{p}.json') for p in ['source','val','test']};cases={};raw={};scalars=0
 for s in SS:
  for arm in ['C','D']:
   for a in AA:
    for p in ['source','val','test']:
     v=baseline[p] if a==0 else js(E/f'{s}_{arm}_a{a}__{p}.json');g={k:np.asarray(c,np.int64) for k,c in v['per_group'].items()};assert set(g)==set(baseline[p]['per_group']);assert len(v['per_input'])=={'source':535,'val':963,'test':2167}[p]
     aggregate=collections.defaultdict(lambda:np.zeros((3,3),np.int64))
     for inp in v['per_input']:
      c=np.asarray(inp['confusion'],np.int64);assert (c>=0).all();aggregate[inp.get('scene',inp.get('group'))]+=c
     assert set(aggregate)==set(g) and all(np.array_equal(aggregate[k],c) for k,c in g.items());assert sum(g.values()).tolist()==v['confusion']
     assert all(np.array_equal(c.sum(1),np.asarray(baseline[p]['per_group'][k]).sum(1)) for k,c in g.items())
     if p!='source':assert all(c.sum(1).tolist()==pre['support'][p]['class_reference_support'][k] for k,c in g.items())
     cases[s,arm,a,p]=g;raw[s,arm,a,p]=v
 for v in rows('ALL_POOL_METRICS.csv')+rows('PER_GROUP.csv'):
  s,a,f,p=int(v['seed']),v['arm'],float(v['alpha']),v['pool'];g=cases[s,a,f,p]
  if 'group' in v:c=g[v['group']]
  else:c=sum(g[k] for k in (sorted(g) if v['support']=='full' else strata[v['support']]))
  assert c.tolist()==json.loads(v['confusion']);tp=np.diag(c);u=c.sum(0)+c.sum(1)-tp;assert abs(float(v['mIoU'])-score(c))<1e-10
  for i,cl in enumerate(CS):
   key=cl+'_IoU';assert (not v[key] and u[i]==0) or abs(float(v[key])-100*tp[i]/u[i])<1e-10;scalars+=1
  for key,n,d in [('Shadow_recall',c[2,2],c[2].sum()),('nonShadow_to_Shadow_FPR',c[:2,2].sum(),c[:2].sum())]:assert (not v[key] and d==0) or abs(float(v[key])-n/d)<1e-12
  assert int(v['valid_pixels'])==c.sum()
 lock=js(E/'SOURCE_SELECTION_LOCK.json');assert lock['target_metrics_opened_before_lock'] is False;stats={};sourceids=sorted(baseline['source']['per_group']);b=np.stack([baseline['source']['per_group'][g] for g in sourceids]);bm=score(b)
 for s in SS:
  for arm in ['C','D']:
   for a in AA:
    cc=np.stack([cases[s,arm,a,'source'][g] for g in sourceids]);loss=bm-score(cc);st={'pool_loss_pp':float(score(b.sum(0))-score(cc.sum(0))),'T11_pp':float(np.sort(loss)[-11:].mean()),'worst_ROI_loss_pp':float(loss.max()),'ROI_loss_gt5':int((loss>5).sum())};stats[s,arm,a]=st
    for k,v in st.items():assert abs(v-lock['source_metrics'][f'{s}/{arm}/{a}'][k])<1e-10
   for budget in [1,3,5]:assert max(a for a in AA if stats[s,arm,a]['pool_loss_pp']<=budget and stats[s,arm,a]['T11_pp']<=5)==next(v['alpha'] for v in lock['rows'] if v['seed']==s and v['arm']==arm and v['budget_pp']==budget)
 # Check losses/LR against immutable217; endpoints are head-only SHA-verified.
 for s in SS:
  schedule=js(D/f'seed{s}/TARGET_SOURCE_SCHEDULE.json');assert len(schedule)==4000
  newcfg=js(D/f'seed{s}/TRAINING_CONFIG_LOCK.json');oldcfg=js(R/f'outputs/phase217/phase217_20261023/delivery/seed{s}/TRAINING_CONFIG_LOCK.json')
  assert newcfg['optim_wrapper']==oldcfg['optim_wrapper'] and newcfg['param_scheduler']==oldcfg['param_scheduler']
  assert js(D/f'seed{s}/RUNTIME_CODE_LOCK.json')['target_config_SHA']==js(R/f'outputs/phase217/phase217_20261023/delivery/seed{s}/RUNTIME_CODE_LOCK.json')['target_config_SHA']
  for arm in ['C','D']:
   h=D/f'seed{s}/heads/{arm}_iter4000.pth';assert sha(h)==js(D/f'seed{s}/{arm}_TRAIN_COMPLETE.json')['head_SHA']
   losses=[json.loads(v) for v in (D/f'seed{s}/{arm}_LOSSES.jsonl').read_text().splitlines()];old=[json.loads(v) for v in (R/f'outputs/phase217/phase217_20261023/delivery/seed{s}/{arm}_LOSSES.jsonl').read_text().splitlines()];assert len(losses)==len(old)==4000 and [v['lr'] for v in losses]==[v['lr'] for v in old]
   for i,v in enumerate(losses):assert v['step']==i+1 and v['target_input_SHA']==schedule[i]['image_SHA'] and np.isfinite(v['total_loss']) and v['source_KL']>=-1e-6
 # Explicit81 cells/image; verify all reference/teacher/C/D margins and direct counts.
 agg={};jointcount=0
 by={}
 for s in SS:
  for arm in ['C','D']:by[s,arm]={v['name']:v for v in raw[s,arm,1,'source']['per_input']}
 teacher={v['name']:v for v in baseline['source']['per_input']}
 for line in (E/'JOINT_COUNTS.jsonl').open():
  v=json.loads(line);s=v['seed'];j=np.asarray(v['reference_teacher_C_D'],np.int64);assert j.shape==(3,3,3,3) and j.sum()==v['valid_pixels'] and (j>=0).all();cm={k:np.zeros((3,3),np.int64) for k in ['teacher','C','D']}
  for r in range(3):
   for t in range(3):
    for c in range(3):
     for d in range(3):n=int(j[r,t,c,d]);cm['teacher'][r,t]+=n;cm['C'][r,c]+=n;cm['D'][r,d]+=n;jointcount+=1
  assert cm['teacher'].tolist()==teacher[v['name']]['confusion']
  for arm in ['C','D']:assert cm[arm].tolist()==by[s,arm][v['name']]['confusion'] and v[arm+'_prediction_SHA']==by[s,arm][v['name']]['prediction_SHA']
  k=s,v['ROI'];agg[k]=agg.get(k,np.zeros_like(j))+j
 for v in rows('SOURCE_DIRECT_CORRECTNESS.csv'):
  s=int(v['seed']);j=sum(agg[s,g] for g in sourceids) if v['ROI']=='__pooled__' else agg[s,v['ROI']];r=CS.index(v['reference_class']);count=collections.Counter()
  for t in range(3):
   if v['teacher_correctness']=='correct' and t!=r or v['teacher_correctness']=='wrong' and t==r:continue
   for c in range(3):
    for d in range(3):
     n=int(j[r,t,c,d]);count['support']+=n;count['C_wrong_D_right']+=n*(c!=r and d==r);count['C_right_D_wrong']+=n*(c==r and d!=r);count['C_errors']+=n*(c!=r);count['D_errors']+=n*(d!=r)
  count['D_minus_C_errors']=count['D_errors']-count['C_errors'];assert all(int(v[k])==n for k,n in count.items())
 for v in rows('SOURCE_DIRECT_ERROR_FLOWS.csv'):
  s=int(v['seed']);j=sum(agg[s,g] for g in sourceids) if v['ROI']=='__pooled__' else agg[s,v['ROI']];r=CS.index(v['reference_class']);dest=CS.index(v['flow'].split('->')[1]);counts=collections.Counter()
  for t in range(3):
   if v['teacher_correctness']=='correct' and t!=r or v['teacher_correctness']=='wrong' and t==r:continue
   for c in range(3):
    for d in range(3):
     n=int(j[r,t,c,d]);counts['C_pixels']+=n*(c==dest);counts['D_pixels']+=n*(d==dest);counts['removed_by_D']+=n*(c==dest and d!=dest);counts['added_by_D']+=n*(c!=dest and d==dest)
  assert all(int(v[k])==n for k,n in counts.items())
 for v in rows('ERROR_FLOWS.csv'):
  s,a,f,p=int(v['seed']),v['arm'],float(v['alpha']),v['pool'];gs=cases[s,a,f,p];c=sum(gs.values()) if v['group']=='__pooled__' else gs[v['group']];bg={g:np.asarray(x,np.int64) for g,x in baseline[p]['per_group'].items()};b=sum(bg.values()) if v['group']=='__pooled__' else bg[v['group']];i,j=[CS.index(x) for x in v['flow'].split('_to_')];assert int(v['pixels'])==c[i,j] and int(v['net_vs_Source_pixels'])==c[i,j]-b[i,j]
 # Rebuild intervals via histogram matrix multiplication, not analyzer indexed confusion sums.
 rng=np.random.default_rng(217023);keys={'source':sourceids,'full':sorted(baseline['test']['per_group']),'yes':strata['yes'],'no':strata['no']};idx={k:rng.integers(0,len(gs),(10000,len(gs))) for k,gs in keys.items()};hist={k:np.stack([np.bincount(row,minlength=len(keys[k])) for row in ix]) for k,ix in idx.items()};rule=js(O/'BOOTSTRAP_RULE.json');first=js(O/'BOOTSTRAP_FIRST50.json');cis=rows('PAIRED_CI.csv');checks=0
 for k,ix in idx.items():assert hashlib.sha256(ix.astype('<i8').tobytes()).hexdigest()==rule['supports'][k]['index_SHA']
 for context in ['endpoint_alpha1','B1','B3','B5']:
  ds={};ps={}
  for s in SS:
   ca,da=(1,1) if context=='endpoint_alpha1' else tuple(next(v['alpha'] for v in lock['rows'] if v['seed']==s and v['arm']==a and v['budget_pp']==int(context[1:])) for a in ['C','D']);out={};point={}
   for k,gs in keys.items():
    p='source' if k=='source' else 'test';c=np.stack([cases[s,'C',ca,p][g] for g in gs]);d=np.stack([cases[s,'D',da,p][g] for g in gs]);cm=(hist[k]@c.reshape(len(gs),9)).reshape(-1,3,3);dm=(hist[k]@d.reshape(len(gs),9)).reshape(-1,3,3)
    if k=='source':
     out['source_pool_loss_diff_pp']=score(cm)-score(dm);point['source_pool_loss_diff_pp']=float(score(c.sum(0))-score(d.sum(0)));cl=bm-score(c);dl=bm-score(d);out['T11_diff_pp']=np.sort(np.take(dl,idx[k]),axis=1)[:,-11:].mean(1)-np.sort(np.take(cl,idx[k]),axis=1)[:,-11:].mean(1);point['T11_diff_pp']=float(np.sort(dl)[-11:].mean()-np.sort(cl)[-11:].mean())
    else:name='test_'+k+'_mIoU_diff';out[name]=score(dm)-score(cm);point[name]=float(score(d.sum(0))-score(c.sum(0)))
   ds[str(s)]=out;ps[str(s)]=point
  ds['mean_of_two']={k:(ds[str(SS[0])][k]+ds[str(SS[1])][k])/2 for k in ds[str(SS[0])]};ps['mean_of_two']={k:(ps[str(SS[0])][k]+ps[str(SS[1])][k])/2 for k in ps[str(SS[0])]}
  for scope,dd in ds.items():
   for k,v in dd.items():
    row=next(r for r in cis if r['context']==context and r['scope']==scope and r['metric']==k);lo,hi=np.quantile(v,[.025,.975]);assert abs(float(row['point'])-ps[scope][k])<1e-10 and abs(float(row['CI95_low'])-lo)<1e-10 and abs(float(row['CI95_high'])-hi)<1e-10;assert np.allclose(v[:50],first[context+'/'+scope+'/'+k],rtol=0,atol=1e-10);checks+=1
 receipt={'status':'passed','raw_files_SHA_checked':len(m),'scalar_class_IoU_checks':scalars,'explicit_joint_cells':jointcount,'paired_bootstrap_full10000_checks':checks,'Source_rules_recomputed':True,'L8_full16_yes6_no10_kept':True,'official_source_test_reads':0,'new_model_calls':0,'seconds':time.monotonic()-clock,'code_SHA':sha(Path(__file__))};(O/'INDEPENDENT_CPU_ACCEPTANCE.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()

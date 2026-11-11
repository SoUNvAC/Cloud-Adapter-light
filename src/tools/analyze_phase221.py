"""CPU-only219 four-arm tables, fixed paired bootstrap and teacher correctness counts."""
from pathlib import Path
import json,csv,hashlib,time,collections,functools
import numpy as np
from analyze_phase216 import metrics,miou,peak
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase221/phase221_20261024';RAW=OUT/'delivery';OLD=ROOT/'outputs/phase217/phase217_20261023/delivery';H219=ROOT/'outputs/phase219/phase219_20261023/delivery';E218=ROOT/'outputs/phase218/phase218_20261023/delivery';SEEDS=[21701,21702];ARMS=['C','D','G','S','N','P'];AS=[0,.25,.5,.75,1]
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(v) for v in p.read_text(encoding='utf-8-sig').splitlines()]
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def write(n,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r)) or ['no_rows']
 with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
@functools.lru_cache(maxsize=None)
def data(s,a,f,p):return js((OLD if a in ['C','D'] else H219 if a in ['G','S'] else RAW)/'evaluation'/(f'Source__{p}.json' if f==0 else f'{s}_{a}_a{f}__{p}.json'))
def groups(r):return {k:np.array(v,np.int64) for k,v in r['per_group'].items()}
def state(t):
 # t axes reference, frozen teacher, candidate. No D-to-candidate joint pixels inferred.
 out={}
 for tc in [True,False]:
  for sc in [True,False]:out[f'T{int(tc)}S{int(sc)}']=sum(int(t[i,j,k]) for i in range(3) for j in range(3) for k in range(3) if (i==j)==tc and (i==k)==sc)
 return out
def countdiff(t,b):
 c=state(t);d=state(b);benefit=d['T1S0']-c['T1S0'];cost=d['T0S1']-c['T0S1'];err=c['T1S0']+c['T0S0']-d['T1S0']-d['T0S0'];assert err==cost-benefit
 return {'teacher_correct_net_benefit_pixels':benefit,'teacher_correction_net_cost_pixels':cost,'total_error_delta_pixels':err,'candidate_teacher_correct_student_wrong':c['T1S0'],'baseline_teacher_correct_student_wrong':d['T1S0'],'candidate_teacher_wrong_student_correct':c['T0S1'],'baseline_teacher_wrong_student_correct':d['T0S1']}
def main():
 start=time.monotonic();assert js(RAW/'COMPLETE.json')['units']==75018 and js(RAW/'REMOTE_CPU_ACCEPTANCE.json')['status']=='passed'
 locks=[js(x/'evaluation/SOURCE_SELECTION_LOCK.json') for x in [OLD,H219,RAW]];assert all(v['target_metrics_opened_before_lock'] is False for v in locks)
 chosen={(v['seed'],v['arm'],v['budget_pp']):v['alpha'] for l in locks for v in l['rows']}
 base={p:groups(data(21701,'C',0,p)) for p in ['source','val','test']};assert [len(base[p]) for p in ['source','val','test']]==[107,10,10]
 cases={};summary={};stat={};poolrows=[];per=[];flows=[];tail=[];cand=[];selected=[]
 for s in SEEDS:
  for a in ARMS:
   lock=locks[0 if a in ['C','D'] else 1 if a in ['G','S'] else 2]
   for f in AS:
    for p in ['source','val','test']:
     raw=data(s,a,f,p);g=groups(raw);cases[s,a,f,p]=g;assert set(g)==set(base[p]) and all(np.array_equal(c.sum(1),base[p][k].sum(1)) for k,c in g.items());total=sum(g.values());assert total.tolist()==raw['confusion']
     m=metrics(total);gm=[metrics(c) for c in g.values()];row={'seed':s,'arm':a,'alpha':f,'pool':p,'historical_reuse':a in ['C','D','G','S'],'groups':len(g),'group_mean_mIoU':float(np.mean([v['mIoU'] for v in gm])),'max_group_FPR':max(v['nonShadow_to_Shadow_FPR'] for v in gm if v['nonShadow_to_Shadow_FPR'] is not None),'zero_shadow_recall_groups':sum(v['shadow_pixels']>0 and v['Shadow_recall']==0 for v in gm),'groups_without_shadow':sum(v['shadow_pixels']==0 for v in gm),**m}
     for key in ['Surface_IoU','Cloud_IoU','Shadow_IoU','Shadow_recall','nonShadow_to_Shadow_FPR']:
      vals=[v[key] for v in gm if v[key] is not None];row['group_mean_'+key]=float(np.mean(vals)) if vals else None
     poolrows.append(row);summary[s,a,f,p]=row
     for k,c in sorted(g.items()):per.append({'seed':s,'arm':a,'alpha':f,'pool':p,'group':k,'confusion':json.dumps(c.tolist()),'delta_mIoU_vs_Source_pp':float(miou(c)-miou(base[p][k])),**metrics(c)})
     for k,c in list(sorted(g.items()))+[('__pooled__',total)]:
      for i,src in enumerate(['Surface','Cloud','Shadow']):
       for j,dst in enumerate(['Surface','Cloud','Shadow']):
        if i==j:continue
        b=sum(base[p].values()) if k=='__pooled__' else base[p][k];den=int(c[i].sum());flows.append({'seed':s,'arm':a,'alpha':f,'pool':p,'group':k,'flow':src+'->'+dst,'pixels':int(c[i,j]),'net_pixels_vs_Source':int(c[i,j]-b[i,j]),'rate':int(c[i,j])/den if den else None})
    g=cases[s,a,f,'source'];loss={k:float(miou(base['source'][k])-miou(v)) for k,v in g.items()};ordered=sorted(loss,key=lambda k:(-loss[k],k));st={'pool_loss_pp':float(miou(sum(base['source'].values()))-miou(sum(g.values()))),'T11_pp':float(np.mean([loss[k] for k in ordered[:11]])),'worst_ROI_loss_pp':loss[ordered[0]],'worst_ROI':ordered[0],'ROI_loss_gt5':sum(v>5 for v in loss.values())};stat[s,a,f]=st
    for key in ['pool_loss_pp','T11_pp','worst_ROI_loss_pp','ROI_loss_gt5']:assert abs(st[key]-lock['source_metrics'][f'{s}/{a}/{f}'][key])<1e-10
    for rank,k in enumerate(ordered,1):tail.append({'seed':s,'arm':a,'alpha':f,'rank':rank,'ROI':k,'loss_pp':loss[k],'in_T11':rank<=11})
    cand.append({'seed':s,'arm':a,'alpha':f,'historical_reuse':a in ['C','D','G','S'],**st,'source_mIoU':summary[s,a,f,'source']['mIoU'],'val_mIoU':summary[s,a,f,'val']['mIoU'],'test_mIoU':summary[s,a,f,'test']['mIoU'],'test_Shadow_IoU':summary[s,a,f,'test']['Shadow_IoU'],'test_Shadow_recall':summary[s,a,f,'test']['Shadow_recall'],'test_FPR':summary[s,a,f,'test']['nonShadow_to_Shadow_FPR']})
   for b in [1,3,5]:
    f=max(f for f in AS if stat[s,a,f]['pool_loss_pp']<=b and stat[s,a,f]['T11_pp']<=5);assert f==chosen[s,a,b];selected.append({'seed':s,'arm':a,'budget_pp':b,'alpha':f,**stat[s,a,f],'test_mIoU':summary[s,a,f,'test']['mIoU']})
 write('ALL_CANDIDATES.csv',cand);write('ALL_POOL_METRICS.csv',poolrows);write('PER_GROUP.csv',per);write('ERROR_FLOWS.csv',flows);write('SOURCE_TAIL_GROUPS.csv',tail);write('JOINT_RULE.csv',selected);write('TABLE_A_ENDPOINTS.csv',[r for r in cand if r['alpha']==1])
 tensors={};perimage={};correctrows=[]
 oldrows=[v for v in lines(E218/'PER_IMAGE.jsonl') if v['pool']=='Source_val535']+lines(H219/'evaluation/ENDPOINT_CORRECTNESS.jsonl');newrows=[]
 for v in lines(RAW/'evaluation/JOINT_COUNTS.jsonl'):
  j=np.array(v['J'],np.int64)
  for arm,axes in [('N',(0,3,5)),('P',(0,3,4))]:
   tt=j.sum(axis=axes);ss=state(tt);newrows.append({'seed':v['seed'],'arm':arm,'ROI':v['ROI'],'name':v['name'],'teacher_prediction_SHA':v['teacher_prediction_SHA'],'reference_teacher_student_counts':tt.tolist(),'confusion':tt.sum(1).tolist(),'correctness':[{'teacher_correct':bool(tc),'student_correct':bool(sc),'pixels':ss[f'T{tc}S{sc}']} for tc in [0,1] for sc in [0,1]]})
 assert len(oldrows)==4280 and len(newrows)==2140
 for v in oldrows+newrows:
  key=v['seed'],v['arm'],v['ROI'];t=np.array(v['reference_teacher_student_counts'],np.int64);assert t.shape==(3,3,3) and (t>=0).all();tensors[key]=tensors.get(key,np.zeros((3,3,3),np.int64))+t
  st=state(t);assert all(st[f"T{int(x['teacher_correct'])}S{int(x['student_correct'])}"]==x['pixels'] for x in v['correctness'])
  expected=next(q for q in data(v['seed'],v['arm'],1,'source')['per_input'] if q['name']==v['name']);assert t.sum(1).tolist()==expected['confusion'];perimage[v['seed'],v['arm'],v['name']]=(t.sum(2),v['teacher_prediction_SHA'])
 for s in SEEDS:
  for a in ARMS:
   for k in sorted(base['source']):
    t=tensors[s,a,k];assert np.array_equal(t.sum(1),cases[s,a,1,'source'][k]) and np.array_equal(t.sum(2),base['source'][k]);correctrows.append({'seed':s,'arm':a,'ROI':k,'valid_pixels':int(t.sum()),**state(t),'reference_teacher_student_counts':json.dumps(t.tolist())})
   for name in [v['name'] for v in data(s,a,1,'source')['per_input']]:
    tc,ph=perimage[s,a,name];bc,bh=perimage[s,'D',name];assert ph==bh and np.array_equal(tc,bc)
 write('ENDPOINT_TEACHER_CORRECTNESS.csv',correctrows)
 comparisons=[];perdiff=[];foc=[];table=[]
 def compare(s,a,b,af,bf,context):
  r={'seed':s,'comparison':a+'-'+b,'context':context,'candidate_alpha':af,'baseline_alpha':bf,'test_mIoU_diff_pp':summary[s,a,af,'test']['mIoU']-summary[s,b,bf,'test']['mIoU'],'source_pool_loss_diff_pp':stat[s,a,af]['pool_loss_pp']-stat[s,b,bf]['pool_loss_pp'],'T11_diff_pp':stat[s,a,af]['T11_pp']-stat[s,b,bf]['T11_pp']}
  if context=='endpoint_alpha1':r.update(countdiff(sum(tensors[s,a,k] for k in base['source']),sum(tensors[s,b,k] for k in base['source'])))
  if context=='endpoint_alpha1':
   ta=sum(tensors[s,a,k] for k in base['source']);tb=sum(tensors[s,b,k] for k in base['source']);r['Shadow_error_delta_pixels']=int(ta[2].sum()-ta[2,:,2].sum()-tb[2].sum()+tb[2,:,2].sum())
  return r
 for a,b in [('N','G'),('N','D'),('N','P')]:
  for context in ['endpoint_alpha1','B1','B3','B5']:
   for s in SEEDS:
    af,bf=(1,1) if context=='endpoint_alpha1' else (chosen[s,a,int(context[1:])],chosen[s,b,int(context[1:])]);r=compare(s,a,b,af,bf,context);comparisons.append(r)
    if context=='endpoint_alpha1':
     table.append(r)
     for k in sorted(base['source']):
      rr={'seed':s,'comparison':a+'-'+b,'ROI':k,'source_mIoU_diff_pp':float(miou(cases[s,a,1,'source'][k])-miou(cases[s,b,1,'source'][k])),**countdiff(tensors[s,a,k],tensors[s,b,k])};perdiff.append(rr)
      if k in ['ROI_0050','ROI_1013','ROI_1404','ROI_4813']:foc.append(rr)
 write('TABLE_B_DIFFERENCES.csv',table);write('ALL_COMPARISONS.csv',comparisons);write('ALL_ROI_CORRECTNESS_DIFFERENCES.csv',perdiff);write('FOUR_FOCUS_ROIS.csv',foc)
 confidence=lines(H219/'evaluation/TEACHER_CONFIDENCE_REFERENCE.jsonl');assert len(confidence)==1070;bands=[];by=collections.defaultdict(list)
 for v in confidence:
  assert v['geometry_pixels']==131072;cm=np.array(v['confusion'],np.int64);assert int(cm.sum())==v['valid_pixels'] and int(cm.sum()-np.trace(cm))==v['teacher_errors'];by[v['ROI'],v['band']].append(v);by['__pooled__',v['band']].append(v)
 for (k,band),vs in sorted(by.items()):
  cm=sum(np.array(v['confusion'],np.int64) for v in vs);r={'ROI':k,'band':band,'geometry_pixels':sum(v['geometry_pixels'] for v in vs),'valid_pixels':int(cm.sum()),'teacher_errors':int(cm.sum()-np.trace(cm)),'reference_error_rate':float((cm.sum()-np.trace(cm))/cm.sum()) if cm.sum() else None,'mean_teacher_confidence':float(np.mean([v['mean_teacher_confidence'] for v in vs])),'confusion':json.dumps(cm.tolist())}
  for i,n in enumerate(['Surface','Cloud','Shadow']):r[n+'_support']=int(cm[i].sum());r[n+'_errors']=int(cm[i].sum()-cm[i,i]);r[n+'_error_rate']=float((cm[i].sum()-cm[i,i])/cm[i].sum()) if cm[i].sum() else None
  bands.append(r)
 for k in base['source']:assert sum(np.array(json.loads(v['confusion']),np.int64) for v in bands if v['ROI']==k).tolist()==base['source'][k].tolist()
 write('TEACHER_CONFIDENCE_DIAGNOSTIC.csv',bands)
 # Exactly the217 group indices; seeds remain paired and nonlinear metrics remain within seed.
 it=np.random.default_rng(217023).integers(0,10,(10000,10));ir=np.random.default_rng(217023).integers(0,107,(10000,107));sid=sorted(base['source']);tid=sorted(base['test']);bm=np.array([miou(base['source'][k]) for k in sid]);ci=[];proof={}
 for a,b in [('N','G'),('N','D'),('N','P')]:
  for ctx in ['endpoint_alpha1','B1','B3','B5']:
   draws={};pts={}
   for s in SEEDS:
    af,bf=(1,1) if ctx=='endpoint_alpha1' else (chosen[s,a,int(ctx[1:])],chosen[s,b,int(ctx[1:])]);aa=np.stack([cases[s,a,af,'test'][k] for k in tid]);bb=np.stack([cases[s,b,bf,'test'][k] for k in tid]);td=miou(aa[it].sum(1))-miou(bb[it].sum(1));aa=np.stack([cases[s,a,af,'source'][k] for k in sid]);bb=np.stack([cases[s,b,bf,'source'][k] for k in sid]);sd=miou(bb[ir].sum(1))-miou(aa[ir].sum(1));al=bm-miou(aa);bl=bm-miou(bb);tl=np.sort(al[ir],axis=1)[:,-11:].mean(1)-np.sort(bl[ir],axis=1)[:,-11:].mean(1);draws[str(s)]={'test_mIoU_diff_pp':td,'source_pool_loss_diff_pp':sd,'T11_diff_pp':tl};pts[str(s)]=compare(s,a,b,af,bf,ctx)
   draws['mean_of_two']={k:(draws[str(SEEDS[0])][k]+draws[str(SEEDS[1])][k])/2 for k in draws[str(SEEDS[0])]};pts['mean_of_two']={k:(pts[str(SEEDS[0])][k]+pts[str(SEEDS[1])][k])/2 for k in draws['mean_of_two']}
   for scope,ds in draws.items():
    for k,v in ds.items():
     lo,hi=np.quantile(v,[.025,.975]);ci.append({'comparison':a+'-'+b,'context':ctx,'scope':scope,'metric':k,'point':pts[scope][k],'CI95_low':float(lo),'CI95_high':float(hi),'groups':10 if k.startswith('test') else 107,'draws':10000,'bootstrap_seed':217023,'conditional':'fixed trained models, Source/replay pool and selected alphas; no re-selection or training-seed-population uncertainty'});proof[a+'-'+b+'/'+ctx+'/'+scope+'/'+k]={'first50':v[:50].tolist(),'draws_SHA':hashlib.sha256(v.tobytes()).hexdigest()}
 write('PAIRED_CI.csv',ci);save('BOOTSTRAP_PROOF.json',proof)
 damage=[]
 for a in ARMS:
  for context in ['alpha'+str(f) for f in AS]+['B1','B3','B5']:
   af=[float(context[5:]) if context.startswith('alpha') else chosen[s,a,int(context[1:])] for s in SEEDS]
   for k in sid:
    loss=[float(miou(base['source'][k])-miou(cases[s,a,f,'source'][k])) for s,f in zip(SEEDS,af)];damage.append({'arm':a,'context':context,'ROI':k,'seed21701_alpha':af[0],'seed21702_alpha':af[1],'seed21701_loss_pp':loss[0],'seed21702_loss_pp':loss[1],'both_loss_gt5':all(v>5 for v in loss)})
 write('ALL_ROI_DAMAGE_MEMBERSHIP.csv',damage);write('COMMON_DAMAGE_ROIS.csv',[v for v in damage if v['both_loss_gt5']])
 save('ANALYSIS_RECEIPT.json',{'status':'passed','seconds':time.monotonic()-start,'CPU_peak_bytes':peak(),'new_model_calls':0,'60_candidates':len(cand),'all_pool_rows':len(poolrows),'per_group_rows':len(per),'all_correctness_ROI_rows':len(correctrows),'shared_scene_ROI_draws':True,'no_pixel_transition_D_to_candidate_inferred':True,'new_source_selection_SHA':sha(RAW/'evaluation/SOURCE_SELECTION_LOCK.json'),'old_source_selection_SHA':sha(OLD/'evaluation/SOURCE_SELECTION_LOCK.json'),'code_SHA':sha(Path(__file__))});assert peak()<16*1024**3
 print(json.dumps(table,indent=2))
if __name__=='__main__':main()

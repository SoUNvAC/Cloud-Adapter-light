"""Frozen SPARCS image weighting: all points, paired groups, and five-axis counts."""
from pathlib import Path
import json,csv,collections,hashlib,functools,time
import numpy as np
from analyze_phase216 import metrics,miou,peak
from analyze_phase224 import denominators
R=next(p for p in Path(__file__).resolve().parents if (p/'src/tools').is_dir());O=R/'outputs/phase226/phase226_20261025';D=O/'delivery';E=D/'evaluation';SS=[21701,21702];AA=[0,.25,.5,.75,1];ARMS=['C','D','H','R','Q'];CS=['Surface','Cloud','Shadow']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(v) for v in p.read_text(encoding='utf-8-sig').splitlines()]
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def write(n,rr):
 with (O/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for v in rr for k in v)) or ['no_rows']);w.writeheader();w.writerows(rr)
@functools.lru_cache(None)
def raw(s,a,f,p):return js((E if a in ['R','Q'] and f else D/'historical/evaluation')/(f'{s}_{a}_a{f}__{p}.json' if f else f'Source__{p}.json'))
def main():
 clock=time.monotonic();assert js(D/'COMPLETE.json')['units']==74508 and js(D/'REMOTE_CPU_ACCEPTANCE.json')['status']=='passed'
 base={p:raw(21701,'C',0,p)['per_group'] for p in ['source','val','test']};cases={};stats={};pm={};per=[];pool=[];allc=[];tails=[];flows=[];choice={};locks=[js(E/'SOURCE_SELECTION_LOCK.json'),js(D/'historical/evaluation/SOURCE_SELECTION_LOCK.json'),js(D/'historical/H_SOURCE_SELECTION_LOCK.json')];assert locks[0]['target_metrics_opened_before_lock'] is False
 for s in SS:
  for a in ARMS:
   for f in AA:
    for p in ['source','val','test']:
     q=raw(s,a,f,p);g={k:np.asarray(v,np.int64) for k,v in q['per_group'].items()};cases[s,a,f,p]=g;assert set(g)==set(base[p]) and sum(g.values()).tolist()==q['confusion'] and all(np.array_equal(v.sum(1),np.asarray(base[p][k]).sum(1)) for k,v in g.items())
     for support in ['full']:
      keys=sorted(g) if support=='full' else strata[support];c=sum(g[k] for k in keys);row={'seed':s,'arm':a,'alpha':f,'pool':p,'support':support,'groups':len(keys),'confusion':json.dumps(c.tolist()),**metrics(c),**denominators(c),'FPR_qualification':'provided SPARCS dense reference'};pool.append(row);pm[s,a,f,p,support]=row
     for k,c in sorted(g.items()):
      b=np.asarray(base[p][k]);per.append({'seed':s,'arm':a,'alpha':f,'pool':p,'group':k,'loss_pp':float(miou(b)-miou(c)),'confusion':json.dumps(c.tolist()),**metrics(c),**denominators(c)})
      for i in range(3):
       for j in range(3):
        if i!=j:flows.append({'seed':s,'arm':a,'alpha':f,'pool':p,'group':k,'flow':CS[i]+'->'+CS[j],'pixels':int(c[i,j]),'net_vs_Source_pixels':int(c[i,j]-b[i,j])})
    g=cases[s,a,f,'source'];loss={k:float(miou(base['source'][k])-miou(v)) for k,v in g.items()};order=sorted(g,key=lambda k:(-loss[k],k));st={'pool_loss_pp':float(miou(raw(s,a,0,'source')['confusion'])-miou(sum(g.values()))),'T11_pp':float(np.mean([loss[k] for k in order[:11]])),'worst_ROI':order[0],'worst_ROI_loss_pp':loss[order[0]],'ROI_loss_gt5':sum(v>5 for v in loss.values())};stats[s,a,f]=st
    for rank,k in enumerate(order,1):tails.append({'seed':s,'arm':a,'alpha':f,'ROI':k,'rank':rank,'loss_pp':loss[k],'in_T11':rank<=11})
    allc.append({'seed':s,'arm':a,'alpha':f,**st,**{f'test{suffix}_mIoU':pm[s,a,f,'test',support]['mIoU'] for suffix,support in [('', 'full')]},'val_mIoU':pm[s,a,f,'val','full']['mIoU']})
   for b in [1,3,5]:
    chosen=max(f for f in AA if stats[s,a,f]['pool_loss_pp']<=b and stats[s,a,f]['T11_pp']<=5);choice[s,a,b]=chosen;lock=locks[0 if a in ['R','Q'] else 2 if a=='H' else 1];assert chosen==next(v['alpha'] for v in lock['rows'] if v['seed']==s and v['arm']==a and v['budget_pp']==b and (a!='H' or v['target']=='SPARCS'))
 write('ALL_POOL_METRICS.csv',pool);write('PER_GROUP.csv',per);write('ERROR_FLOW_NET_CHANGES.csv',flows);write('SOURCE_TAIL_GROUPS.csv',tails);write('TABLE_B_ALL_CANDIDATES.csv',allc);write('FOCUS_ROI0124_ALPHA025_AND1.csv',[v for v in per if v['pool']=='source' and v['group']=='ROI_0124' and v['alpha'] in [.25,1]])
 tablea=[]
 for f in [.25,1]:
  for a in ARMS:
   rr=[v for v in allc if v['arm']==a and v['alpha']==f];tablea.extend(rr);mean={'seed':'mean_of_two','arm':a,'alpha':f,'worst_ROI':'see individual seeds'}
   mean.update({k:float(np.mean([v[k] for v in rr])) for k in rr[0] if k not in mean});tablea.append(mean)
 write('TABLE_A_FIXED_POINTS.csv',tablea);write('SOURCE_RULE_ALL_ARMS.csv',[{'seed':s,'arm':a,'budget_pp':b,'alpha':choice[s,a,b]} for s in SS for a in ARMS for b in [1,3,5]])
 ids=sorted(base['source']);bm=np.array([miou(base['source'][g]) for g in ids]);supports={'source':ids,'full':sorted(base['test'])};rng=np.random.default_rng(217023);ix={k:rng.integers(0,len(v),(10000,len(v))) for k,v in supports.items()};proof={'seed':217023,'draws':10000,'stream_order':list(supports),'index_SHA':{k:hashlib.sha256(v.astype('<i8').tobytes()).hexdigest() for k,v in ix.items()},'groups':{k:len(v) for k,v in supports.items()},'fixed_alpha_not_reselected':True,'historical_weight_generation_uncertainty_not_included':True};comparison=[];ci=[];first={}
 for other in ['Q','D','H']:
  for context in ['alpha025','alpha1','B1','B3','B5']:
   pts={};draws={};individual=[]
   for s in SS:
    x,y=(.25,.25) if context=='alpha025' else (1,1) if context=='alpha1' else (choice[s,'R',int(context[1:])],choice[s,other,int(context[1:])]);g=np.stack([cases[s,'R',x,'source'][k] for k in ids]);h=np.stack([cases[s,other,y,'source'][k] for k in ids]);gl=bm-miou(g);hl=bm-miou(h);v={'source_pool_loss_diff_pp':stats[s,'R',x]['pool_loss_pp']-stats[s,other,y]['pool_loss_pp'],'T11_diff_pp':stats[s,'R',x]['T11_pp']-stats[s,other,y]['T11_pp']};dd={'source_pool_loss_diff_pp':miou(h[ix['source']].sum(1))-miou(g[ix['source']].sum(1)),'T11_diff_pp':np.sort(gl[ix['source']],axis=1)[:,-11:].mean(1)-np.sort(hl[ix['source']],axis=1)[:,-11:].mean(1)}
    for k in ['full']:
     gg=np.stack([cases[s,'R',x,'test'][n] for n in supports[k]]);hh=np.stack([cases[s,other,y,'test'][n] for n in supports[k]]);key='test_mIoU_diff_pp' if k=='full' else f'test_{k}_mIoU_diff_pp';v[key]=float(miou(gg.sum(0))-miou(hh.sum(0)));dd[key]=miou(gg[ix[k]].sum(1))-miou(hh[ix[k]].sum(1))
    pts[str(s)]=v;draws[str(s)]=dd;individual.append({'comparison':'R-'+other,'context':context,'seed':s,'R_alpha':x,'other_alpha':y,**v})
   pts['mean_of_two']={k:float(np.mean([v[k] for v in pts.values()])) for k in pts[str(SS[0])]};draws['mean_of_two']={k:(draws[str(SS[0])][k]+draws[str(SS[1])][k])/2 for k in draws[str(SS[0])]};comparison.extend(individual+[{'comparison':'R-'+other,'context':context,'seed':'mean_of_two','R_alpha':'/'.join(str(v['R_alpha']) for v in individual),'other_alpha':'/'.join(str(v['other_alpha']) for v in individual),**pts['mean_of_two']}])
   for scope,ds in draws.items():
    for k,v in ds.items():
     lo,hi=np.quantile(v,[.025,.975]);ci.append({'comparison':'R-'+other,'context':context,'scope':scope,'metric':k,'point':pts[scope][k],'CI95_low':float(lo),'CI95_high':float(hi),'draws':10000,'seed':217023});first['/'.join(['R-'+other,context,scope,k])]=v[:50].tolist()
 write('FIXED_COMPARISONS.csv',comparison);write('PAIRED_CI.csv',ci);save('BOOTSTRAP_RULE.json',proof);save('BOOTSTRAP_FIRST50.json',first)
 finite=[];coverage=[]
 for s in SS:
  for a in ['R','Q']:
   for f in AA:
    covers=[];wins=[]
    for oldarm in ['D','H']:
     for oldf in AA:
      z={'seed':s,'new_arm':a,'new_alpha':f,'old_arm':oldarm,'old_alpha':oldf,'test_diff_pp':pm[s,a,f,'test','full']['mIoU']-pm[s,oldarm,oldf,'test','full']['mIoU'],'source_loss_diff_pp':stats[s,a,f]['pool_loss_pp']-stats[s,oldarm,oldf]['pool_loss_pp'],'T11_diff_pp':stats[s,a,f]['T11_pp']-stats[s,oldarm,oldf]['T11_pp']};v=np.array([z['test_diff_pp'],-z['source_loss_diff_pp'],-z['T11_diff_pp']]);z['new_dominates_old']=bool((v>=0).all() and (v>0).any());z['old_weakly_covers_new']=bool((v<=0).all());z['old_dominates_new']=bool((v<=0).all() and (v<0).any());finite.append(z);covers.append(z['old_weakly_covers_new']);wins.append(z['new_dominates_old'])
    coverage.append({'seed':s,'arm':a,'alpha':f,'weakly_covered_by_any_DH_point':any(covers),'dominates_any_DH_point':any(wins),'old_points':10,'descriptive_not_continuous_frontier':True})
 write('FINITE_ALL200_PAIRS.csv',finite);write('FINITE_DH_COVERAGE.csv',coverage)
 agg=collections.defaultdict(lambda:np.zeros((3,3,3,3,3),np.int64))
 for v in lines(E/'JOINT_COUNTS.jsonl'):agg[v['seed'],v['ROI']]+=np.asarray(v['J'],np.int64)
 direct=[];correctness=[];index=np.indices((3,3,3))
 for s in SS:
  for group in ids+['__pooled__']:
   j=sum(agg[s,k] for k in ids) if group=='__pooled__' else agg[s,group]
   for olda,newa,oi,ni in [('D','R',0,1),('D','Q',0,2),('R','Q',1,2)]:
    lesswrong=lesscorrect=net=0
    for r in range(3):
     for t in range(3):
      x=j[r,t];diff=int(x[index[ni]!=r].sum()-x[index[oi]!=r].sum());net+=diff
      if r==t:lesswrong-=diff
      else:lesscorrect+=int(x[index[oi]==r].sum()-x[index[ni]==r].sum())
     for tier,keep in [('all',[0,1,2]),('correct',[r]),('wrong',[t for t in range(3) if t!=r])]:
      x=j[r,keep].sum(0);direct.append({'seed':s,'ROI':group,'old_arm':olda,'new_arm':newa,'reference_class':CS[r],'teacher_correctness':tier,'pixels':int(x.sum()),'old_wrong_new_right':int(x[(index[oi]!=r)&(index[ni]==r)].sum()),'old_right_new_wrong':int(x[(index[oi]==r)&(index[ni]!=r)].sum()),'new_minus_old_errors':int(x[index[ni]!=r].sum()-x[index[oi]!=r].sum())})
    assert net==lesscorrect-lesswrong;correctness.append({'seed':s,'ROI':group,'old_arm':olda,'new_arm':newa,'teacher_correct_new_fewer_errors':lesswrong,'teacher_wrong_new_fewer_corrections':lesscorrect,'new_minus_old_errors':net})
 write('TABLE_C_DIRECT_ALPHA1.csv',direct);write('TEACHER_CORRECTNESS_ALPHA1.csv',correctness)
 losses=[]
 for s in SS:
  for a in ['R','Q']:
   rr=lines(D/f'seed{s}/{a}_LOSSES.jsonl')
   for name,z in [('first100',rr[:100]),('last100',rr[-100:]),('all4000',rr)]:losses.append({'seed':s,'arm':a,'segment':name,**{k:float(np.mean([v[k] for v in z])) for k in ['target_loss','source_KL','weight_used','weighted_KL','total_loss']}})
 write('LOSS_SUMMARY.csv',losses);save('ANALYSIS_RECEIPT.json',{'new_model_calls':0,'seconds':time.monotonic()-clock,'CPU_peak_bytes':peak(),'pool_rows':len(pool),'per_group_rows':len(per),'candidates':len(allc),'finite_pairs':len(finite)});assert peak()<16*1024**3
if __name__=='__main__':main()

"""Independent CPU pair AUC, transitions, arithmetic and order verification."""
from phase242_common import *
import math,traceback
def exact_auc(pos_path,neg_path,cpu):
 p=np.memmap(pos_path,mode='r',dtype='<f8');n=np.memmap(neg_path,mode='r',dtype='<f8')
 try:
  if not len(p) or not len(n):return None
  asc=n[::-1];wins=np.longdouble(0)
  for start in range(0,len(p),131072):
   q=p[start:start+131072];left=np.searchsorted(asc,q,'left');right=np.searchsorted(asc,q,'right');wins+=np.sum(left.astype(np.longdouble)+.5*(right-left),dtype=np.longdouble);cpu.check()
  return float(wins/(np.longdouble(len(p))*len(n)))
 finally:p._mmap.close();n._mmap.close()
def main():
 global cpu
 cpu=CPU('validate');checks=[]
 def ok(name,value):assert bool(value),name;checks.append(name)
 seal=js(R/'VAL_SEAL.json');ok('val before seal before test',js(R/'VAL_COMPLETE.json')['epoch']<=seal['epoch']<js(R/'TEST_ACCESS_STARTED.json')['epoch'])
 for n,h in seal['files'].items():ok('seal '+n,sha(R/n)==h)
 for split in ('val','test'):
  for domain in ('L8','SPARCS'):
   r=js(R/(split.upper()+'_'+domain+'_RESULTS.json'));scenes=r['scenes'];source=r['matrices']['Source:raw'];ss={(c['positive_scene'],c['negative_scene']):c for c in source['cells']};ps={c['positive_scene']:c['P'] for c in source['cells']};ns={c['negative_scene']:c['N'] for c in source['cells']};den=float(sum(ps.values()))*sum(ns.values());diagden=sum(ps[s]*float(ns[s]) for s in scenes)
   for modelstate,mat in r['matrices'].items():
    cells=mat['cells'];ok('matrix NxN '+modelstate,len(cells)==len(scenes)**2);pool=math.fsum((c['AUC'] or 0.)*(ps[c['positive_scene']]*float(ns[c['negative_scene']])/den) for c in cells);ok('independent weighted pooled '+modelstate,abs(pool-mat['summary']['pooled_AUC'])<1e-10)
    diagonal=[c['AUC'] for c in cells if c['positive_scene']==c['negative_scene'] and c['AUC'] is not None];ok('macro diagonal '+modelstate,abs(sum(diagonal)/len(diagonal)-mat['summary']['scene_mean_AUC'])<1e-10)
    within=math.fsum((c['AUC'] or 0.)*ps[c['positive_scene']]*ns[c['negative_scene']] for c in cells if c['positive_scene']==c['negative_scene'])/diagden;ok('weighted diagonal '+modelstate,abs(within-mat['summary']['within_weighted_AUC'])<1e-10)
    for c in cells:ok('fixed P,N,weight '+modelstate,c['P']==ps[c['positive_scene']] and c['N']==ns[c['negative_scene']] and abs(c['weight']-ps[c['positive_scene']]*float(ns[c['negative_scene']])/den)<1e-15)
   for model in ('Source','C21701','C21702'):
    for scene in scenes:
     rec=js(R/'scene_receipts'/split/domain/(scene+'.json'));cache=rec['Source']['cache'] if model=='Source' else rec['C'][model]['cache']
     for label,item in cache.items():ok('sorted cache SHA '+str((split,domain,model,scene,label)),sha(R/item['path'])==item['SHA'])
     if model=='Source':continue
     c=rec['C'][model];T=np.asarray(c['transition_reference_Source_C']);source_cm=rec['Source']['decision']['argmax']['confusion'];raw=next(v for v in r['decisions'] if v['scene']==scene and v['model']==model and v['state']=='raw' and v['decision']=='argmax');ok('full Source/C transition marginals',T.sum(2).tolist()==source_cm and T.sum(1).tolist()==raw['confusion']);ok('transition valid support',T.sum()==rec['valid'])
     stats=c['statistics'];ok('unlabelled median from all valid',stats[0]['delta']['median']==c['m'])
     for stat in stats:
      if stat['delta']['n']:ok('paired mean identity',abs(stat['delta']['mean']-stat['s_C']['mean']+stat['s_Source']['mean'])<1e-10)
     for k,change in enumerate(c['changes']):ok('correct-wrong transition',change['Source_correct_C_wrong']==int(T[k,k,:].sum()-T[k,k,k]) and change['Source_wrong_C_correct']==int(T[k,:,k].sum()-T[k,k,k]))
    if model=='Source':continue
    a={(c['positive_scene'],c['negative_scene']):c for c in r['matrices'][model+':raw']['cells']};b={(c['positive_scene'],c['negative_scene']):c for c in r['matrices'][model+':J1']['cells']}
    for scene in scenes:
     for key in ('AUC','AP'):
      x=a[scene,scene][key];y=b[scene,scene][key];ok('J1 diagonal invariance',x is None and y is None or x is not None and y is not None and abs(x-y)<1e-10)
    for cell in r['comparisons'][model]:
     pair=(cell['positive_scene'],cell['negative_scene']);source_auc=ss[pair]['AUC'];c_auc=a[pair]['AUC']
     if source_auc is None or c_auc is None:ok('NA pair preserved',cell['delta_AUC'] is None and cell['weighted_delta'] is None)
     else:ok('weighted signed pair damage',abs(cell['delta_AUC']-(c_auc-source_auc))<1e-14 and abs(cell['weighted_delta']-cell['weight']*(c_auc-source_auc))<1e-14)
    for l in [v for v in r['leave_one'] if v['model']==model]:
     remaining=[s for s in scenes if s!=l['excluded']];denom=float(sum(ps[s] for s in remaining))*sum(ns[s] for s in remaining)
     def reduced(cells):
      vals=[v for v in cells if v['positive_scene'] in remaining and v['negative_scene'] in remaining];pooled=math.fsum((v['AUC'] or 0.)*ps[v['positive_scene']]*ns[v['negative_scene']] for v in vals)/denom;diagonal=[v['AUC'] for v in vals if v['positive_scene']==v['negative_scene'] and v['AUC'] is not None];return pooled,sum(diagonal)/len(diagonal)
     sp,sm=reduced(source['cells']);cp,cm=reduced(r['matrices'][model+':raw']['cells']);ok('independent leave-one ranking',abs(cp-sp-l['pooled_delta'])<1e-10 and abs(cm-sm-l['scene_mean_delta'])<1e-10)
    for c in r['matrices'][model+':J0']['cells']:
     raw=a[c['positive_scene'],c['negative_scene']]
     for key in ('AP','AUC'):ok('global J0 pair identity',raw[key]==c[key])
    signal=r['signals'][model];rows=signal['scatter'];x=np.array([v['T'] for v in rows]);y=np.array([v['delta_AUC'] for v in rows]);nonzero=(x!=0)&(y!=0)
    def ranks(a):
     vals,inverse,counts=np.unique(a,return_inverse=True,return_counts=True);ends=np.cumsum(counts);return (ends-.5*(counts-1))[inverse]
    rho=float(np.corrcoef(ranks(x),ranks(y))[0,1]) if len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0 else None;ok('independent Spearman rank correlation',rho is None and signal['Spearman'] is None or rho is not None and signal['Spearman'] is not None and abs(rho-signal['Spearman'])<1e-12);ok('fixed positive signal direction',signal['non_tie_pairs']==int(nonzero.sum()) and (signal['direction_agreement'] is None if not nonzero.any() else abs(signal['direction_agreement']-float((x[nonzero]*y[nonzero]>0).mean()))<1e-15))
    for lo in signal['leave_one_scene']:
     keep=np.array([lo['excluded'] not in (v['positive_scene'],v['negative_scene']) for v in rows]);xx=x[keep];yy=y[keep];rho=float(np.corrcoef(ranks(xx),ranks(yy))[0,1]) if len(xx)>1 and np.ptp(xx)>0 and np.ptp(yy)>0 else None;ok('leave-one signal rho',len(xx)==lo['evaluable_pairs'] and (rho is None if lo['Spearman'] is None else rho is not None and abs(rho-lo['Spearman'])<1e-12))
   for a,b,c in zip(r['comparisons']['C21701'],r['comparisons']['C21702'],r['comparisons']['two_seed_mean']):
    if a['weighted_delta'] is not None:ok('arithmetic two-seed pair mean',abs((a['weighted_delta']+b['weighted_delta'])/2-c['weighted_delta'])<1e-14)
   for contribution in r['contributions']:
    cells=r['comparisons'][contribution['model']];row=math.fsum(v['weighted_delta'] or 0. for v in cells if v['positive_scene']==contribution['scene']);col=math.fsum(v['weighted_delta'] or 0. for v in cells if v['negative_scene']==contribution['scene']);ok('row-column contribution sums',abs(row-contribution['Shadow_row_contribution'])<1e-12 and abs(col-contribution['Surface_column_contribution'])<1e-12)
   if split=='test':
    avg=r['comparisons']['two_seed_mean'];cross=[v for v in avg if v['positive_scene']!=v['negative_scene'] and v['weighted_delta'] is not None];selected=[('damage',v) for v in sorted([v for v in cross if v['weighted_delta']<0],key=lambda v:(v['weighted_delta'],v['positive_scene'],v['negative_scene']))[:3]]+[('benefit',v) for v in sorted([v for v in cross if v['weighted_delta']>0],key=lambda v:(-v['weighted_delta'],v['positive_scene'],v['negative_scene']))[:3]];ok('fixed top-six selection',[(k,v['positive_scene'],v['negative_scene']) for k,v in selected]==[(v['kind'],v['positive_scene'],v['negative_scene']) for v in r['top_pairs']])
    for d in r['decompositions']:
     i=d['positive_scene'];j=d['negative_scene'];model=d['model'];values={}
     for name,pm,nm in [('SS','Source','Source'),('CS',model,'Source'),('SC','Source',model),('CC',model,model)]:values[name]=exact_auc(cachepath(split,domain,pm,i,1),cachepath(split,domain,nm,j,0),cpu);ok('independent searchsorted '+name,abs(values[name]-d[name])<1e-10)
     sp=((values['CS']-values['SS'])+(values['CC']-values['SC']))/2;sn=((values['SC']-values['SS'])+(values['CC']-values['CS']))/2;ok('independent side decomposition',abs(sp-d['Shadow_component'])<1e-10 and abs(sn-d['Surface_component'])<1e-10 and abs(sp+sn-(values['CC']-values['SS']))<1e-10)
   cpu.check()
 ok('no model calls or updates',js(R/'LOCK.json')['network_calls']==0 and js(R/'LOCK.json')['gradient_updates']==0);ok('tiny exactness tests',js(R/'KERNEL_ACCEPTANCE.json')['passed']);save('INDEPENDENT_ACCEPTANCE.json',{'passed':True,'checks':len(checks),'details':checks,'network_calls':0,'gradient_updates':0,'new_thresholds':0,'new_bootstrap':0});cpu.finish('passed')
if __name__=='__main__':
 try:main()
 except BaseException:
  save('VALIDATION_FAILURE.json',{'traceback':traceback.format_exc()})
  if 'cpu' in globals():cpu.finish('failed')
  raise

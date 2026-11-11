"""Direct D/N/P count projections and class-normalized weight diagnostics; CPU only."""
from pathlib import Path
import numpy as np,json,csv,collections,statistics,hashlib
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/phase221/phase221_20261024';RAW=O/'delivery';CLASSES=['Surface','Cloud','Shadow'];BANDS=['high','low']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(v) for v in p.read_text(encoding='utf-8-sig').splitlines()]
def write(n,rows):
 with (O/n).open('w',encoding='utf-8-sig',newline='') as out:
  w=csv.DictWriter(out,fieldnames=list(dict.fromkeys(k for r in rows for k in r)) or ['no_rows']);w.writeheader();w.writerows(rows)
def table(j,seed,roi):
 ds,gs,ss=np.indices((3,3,3));out=[];flows=[]
 for bi,band in enumerate(BANDS):
  for c,cl in enumerate(CLASSES):
   for tier,keep in [('all',[0,1,2]),('correct',[c]),('wrong',[t for t in range(3) if t!=c])]:
    x=j[bi,c,keep].sum(0);n=int(x.sum());de=int(x[ds!=c].sum());ge=int(x[gs!=c].sum());se=int(x[ss!=c].sum());fixed=int(x[(ds!=c)&(gs==c)].sum());broken=int(x[(ds==c)&(gs!=c)].sum());assert ge-de==broken-fixed
    r={'seed':seed,'ROI':roi,'band':band,'reference_class':cl,'teacher_reference_correctness':tier,'valid_pixels':n,'D_wrong_N_right':fixed,'D_right_N_wrong':broken,'N_minus_D_errors':ge-de,'D_errors':de,'N_errors':ge,'P_errors':se,'D_error_rate':de/n if n else None,'N_error_rate':ge/n if n else None,'P_error_rate':se/n if n else None,'P_minus_D_errors':se-de};out.append(r)
    dests=[0,1] if c==2 else [2]
    for dest in dests:
     removed=int(x[(ds==dest)&(gs!=dest)].sum());added=int(x[(ds!=dest)&(gs==dest)].sum());flow={'seed':seed,'ROI':roi,'band':band,'reference_class':cl,'teacher_reference_correctness':tier,'flow':cl+'->'+CLASSES[dest],'valid_reference_pixels':n,'D_flow_pixels':int(x[ds==dest].sum()),'N_flow_pixels':int(x[gs==dest].sum()),'P_flow_pixels':int(x[ss==dest].sum()),'D_flow_removed_by_N':removed,'N_flow_added_from_other_D_predictions':added,'N_minus_D_flow':added-removed};assert flow['N_minus_D_flow']==flow['N_flow_pixels']-flow['D_flow_pixels'];flows.append(flow)
 return out,flows
def main():
 aggregate={};raw=lines(RAW/'evaluation/JOINT_COUNTS.jsonl');assert len(raw)==1070
 for v in raw:
  key=v['seed'],v['ROI'];j=np.array(v['J'],np.int64);assert j.shape==(2,3,3,3,3,3) and j.sum()==v['valid_pixels'];aggregate[key]=aggregate.get(key,np.zeros(j.shape,np.int64))+j
 rows=[];flows=[];npdirect=[];focus=js(RAW/'FOCUS_LOCK.json')
 for s in [21701,21702]:
  ids=sorted(k for seed,k in aggregate if seed==s);assert len(ids)==107
  for k in ids+['__pooled__']:
   j=sum(aggregate[s,z] for z in ids) if k=='__pooled__' else aggregate[s,k];rr,ff=table(j,s,k);rows.extend(rr);flows.extend(ff);nn,_=table(j.transpose(0,1,2,5,4,3),s,k)
   npdirect.extend({key.replace('D','_BASE_').replace('P','D').replace('_BASE_','P'):value for key,value in r.items()} for r in nn)
 for r in rows:
  r['teacher_original_correct_net_errors']=next(v['N_minus_D_errors'] for v in rows if v['seed']==r['seed'] and v['ROI']==r['ROI'] and v['band']==r['band'] and v['reference_class']==r['reference_class'] and v['teacher_reference_correctness']=='correct')
 write('TABLE_C_POOLED.csv',[v for v in rows if v['ROI']=='__pooled__' and v['teacher_reference_correctness']=='all']);write('TABLE_C_ALL_ROIS.csv',[v for v in rows if v['ROI']!='__pooled__']);write('DIRECT_N_MINUS_P.csv',npdirect);write('DIRECT_ERROR_FLOWS.csv',flows);write('FOUR_FOCUS_ROIS.csv',[v for v in rows if v['ROI'] in focus['historical_four']]);write('THIRTEEN_POSTHOC_DAMAGE_ROIS.csv',[v for v in rows if v['ROI'] in focus['both_seeds_G_minus_D_mIoU_below_minus1pp_13']])
 # Jw[reference,teacher,G_high,P_high] proves reference-class weights without reusing train masks.
 exposure=collections.defaultdict(collections.Counter)
 for v in lines(RAW/'evaluation/VAL_WEIGHT_DIAGNOSTIC.jsonl'):
  z=np.array(v['weight_counts'],np.int64);assert z.shape==(3,3,2,2);proof=v['weight_proof'];assert z.sum()==262144
  for ref in range(3):
   for teacher in range(3):
    q=proof['classes'][teacher]
    for gh in range(2):
     for ph in range(2):
      n=int(z[ref,teacher,gh,ph])
      if not n:continue
      g=.5+gh;nw=float(np.float32(g/q['mean_g']));pw=float(np.float32((.5+ph)/q['mean_g']))
      for basis,c in [('reference',ref),('teacher_prediction',teacher)]:
       a=exposure['source_val535','unique535',basis,c];a['support']+=n;a['G_sum']+=n*g;a['N_sum']+=n*nw;a['P_sum']+=n*pw;a['G_high']+=n*gh
 for v in js(RAW/'weight_preparation/WEIGHT_MANIFEST.json'):
  for q in v['classes']:
   if not q['support']:continue
   for method,m in [('unique256',1),('original4000_sequence',v['repeat_count'])]:
    a=exposure['replay256',method,'teacher_prediction',q['class']];a['support']+=q['support']*m;a['G_sum']+=(.5*q['support']+q['g_high_pixels'])*m;a['N_sum']+=q['N_sum']*m;a['P_sum']+=q['P_sum']*m;a['G_high']+=q['g_high_pixels']*m
 weightrows=[]
 for (pool,rep,basis,cl),v in sorted(exposure.items()):
  n=v['support'];weightrows.append({'pool':pool,'repetition':rep,'basis':basis,'class':CLASSES[cl],**dict(v),'G_high_fraction':v['G_high']/n,'G_mean_weight':v['G_sum']/n,'N_mean_weight':v['N_sum']/n,'P_mean_weight':v['P_sum']/n,'meaning':'Source-val diagnostic only, not actual gradients' if pool=='source_val535' else 'teacher prediction, no replay references'})
 write('CLASS_WEIGHT_DISTRIBUTIONS.csv',weightrows)
 contrib=[]
 for s in [21701,21702]:
  for arm in ['N','P']:
   loss=lines(RAW/f'seed{s}/{arm}_LOSSES.jsonl');assert len(loss)==4000
   for c in range(3):
    a=[v['teacher_class_KL'][c] for v in loss];means=[v['weight_mean'] for v in a if v['support']];contrib.append({'seed':s,'arm':arm,'teacher_prediction_class':CLASSES[c],'steps_present':len(means),'empty_class_steps':4000-len(means),'support_with_repeated_uses':sum(v['support'] for v in a),'mean_recorded_weight_when_present':statistics.mean(means) if means else None,'min_recorded_weight_mean':min(means) if means else None,'max_recorded_weight_mean':max(means) if means else None,'mean_weighted_KL_contribution_all4000':statistics.mean(v['weighted_KL_contribution'] for v in a),'first100_mean_contribution':statistics.mean(v['weighted_KL_contribution'] for v in a[:100]),'last100_mean_contribution':statistics.mean(v['weighted_KL_contribution'] for v in a[-100:]),'meaning':'weighted loss component divided by whole image support; not gradient causal attribution'})
 write('TRAINING_CLASS_KL_CONTRIBUTIONS.csv',contrib)
 # Preserve per-seed rows and the mean of metrics, never mean confusion first.
 comparisons=[v for v in csv.DictReader((O/'TABLE_B_DIFFERENCES.csv').open(encoding='utf-8-sig')) if v['seed']!='mean_of_two'];means=[]
 for comp in ['N-G','N-D','N-P']:
  rr=[v for v in comparisons if v['comparison']==comp];assert len(rr)==2;r={'seed':'mean_of_two','comparison':comp,'context':'endpoint_alpha1','candidate_alpha':1,'baseline_alpha':1}
  for k in rr[0]:
   if k not in r:r[k]=statistics.mean(float(v[k]) for v in rr)
  means.append(r)
 write('TABLE_B_DIFFERENCES.csv',comparisons+means)
 (O/'JOINT_AXES.json').write_text(json.dumps({'axes':['high_low','reference','teacher','D','N','P'],'shape':[2,3,3,3,3,3],'bands':BANDS,'classes':CLASSES,'direct_transitions':'D/N/P same pixel available; N/G only net confusion differences, not direct G->N transitions','weight_diagnostic_axes':['reference','teacher','G_high','P_high']},indent=2))
if __name__=='__main__':main()

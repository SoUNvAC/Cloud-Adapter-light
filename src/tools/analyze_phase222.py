"""L8 fixed full/stratified C/D report; paired scenes and ROIs, not patch inference."""
from pathlib import Path
import csv,json,hashlib,time,collections
import numpy as np
from analyze_phase216 import metrics,miou,peak
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase222/phase222_20261024';RAW=O/'delivery';EV=RAW/'evaluation';SS=[21701,21702];AA=[0,.25,.5,.75,1];CS=['Surface','Cloud','Shadow']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,rows):
 with (O/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for v in rows for k in v)) or ['no_rows']);w.writeheader();w.writerows(rows)
def save(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def main():
 clock=time.monotonic();assert js(RAW/'COMPLETE.json')['units']==99190
 lock=js(EV/'SOURCE_SELECTION_LOCK.json');assert lock['target_metrics_opened_before_lock'] is False;pre=js(RAW/'INPUT_PREFLIGHT_ACCEPTED.json');ids=pre['test_strata'];assert list(map(len,[ids['yes'],ids['no']]))==[6,10]
 base={p:{g:np.asarray(c,np.int64) for g,c in js(EV/f'Source__{p}.json')['per_group'].items()} for p in ['source','val','test']};assert [len(base[p]) for p in ['source','val','test']]==[107,8,16]
 cases={};poolrows=[];per=[];flows=[];candidate=[];tail=[];stats={};pooled={}
 for s in SS:
  for a in ['C','D']:
   for f in AA:
    st=lock['source_metrics'][f'{s}/{a}/{f}'];stats[s,a,f]=st
    for pool in ['source','val','test']:
     x=js(EV/(f'Source__{pool}.json' if f==0 else f'{s}_{a}_a{f}__{pool}.json'));gg={g:np.asarray(c,np.int64) for g,c in x['per_group'].items()};cases[s,a,f,pool]=gg
     assert set(gg)==set(base[pool]) and all(np.array_equal(v.sum(1),base[pool][g].sum(1)) for g,v in gg.items()) and sum(gg.values()).tolist()==x['confusion']
     for support in (['full','yes','no'] if pool=='test' else ['full']):
      keys=sorted(gg) if support=='full' else ids[support];c=sum(gg[g] for g in keys);m=metrics(c);row={'seed':s,'arm':a,'alpha':f,'pool':pool,'support':support,'groups':len(keys),**m,'confusion':json.dumps(c.tolist()),'qualification':'USGS Shadows=no is not verified absence; FPR is nominal' if support=='no' else 'provided reference'};poolrows.append(row);pooled[s,a,f,pool,support]=row
     for g,c in sorted(gg.items()):
      row={'seed':s,'arm':a,'alpha':f,'pool':pool,'group':g,'usgs_shadows':('yes' if g in ids['yes'] else 'no') if pool=='test' else 'yes' if pool=='val' else 'NA','confusion':json.dumps(c.tolist()),'delta_mIoU_pp':metrics(c)['mIoU']-metrics(base[pool][g])['mIoU'],**metrics(c)};per.append(row)
     for g,c in list(sorted(gg.items()))+[('__pooled__',sum(gg.values()))]:
      m=metrics(c);b=metrics(sum(base[pool].values()) if g=='__pooled__' else base[pool][g])
      for i in CS:
       for j in CS:
        if i!=j:
         k=i+'_to_'+j;flows.append({'seed':s,'arm':a,'alpha':f,'pool':pool,'group':g,'flow':k,'pixels':m[k+'_pixels'],'net_vs_Source_pixels':m[k+'_pixels']-b[k+'_pixels'],'rate':m[k+'_rate']})
    row={'seed':s,'arm':a,'alpha':f,**st,'val_mIoU':pooled[s,a,f,'val','full']['mIoU']}
    for support in ['full','yes','no']:
     for k in ['mIoU','Shadow_IoU','Shadow_recall','nonShadow_to_Shadow_FPR']:row['test_'+support+'_'+k]=pooled[s,a,f,'test',support][k]
    candidate.append(row)
    for rank,g in enumerate(sorted(base['source'],key=lambda g:float(miou(cases[s,a,f,'source'][g])-miou(base['source'][g]))),1):tail.append({'seed':s,'arm':a,'alpha':f,'rank':rank,'ROI':g,'loss_pp':float(miou(base['source'][g])-miou(cases[s,a,f,'source'][g])),'in_T11':rank<=11})
 write('ALL_CANDIDATES.csv',candidate);write('ALL_POOL_METRICS.csv',poolrows);write('PER_GROUP.csv',per);write('ERROR_FLOWS.csv',flows);write('SOURCE_TAIL_GROUPS.csv',tail)
 write('TABLE_A_ENDPOINTS.csv',[v for v in candidate if v['alpha']==1]);selected={(v['seed'],v['arm'],v['budget_pp']):v['alpha'] for v in lock['rows']};contexts=['endpoint_alpha1','B1','B3','B5'];comparisons=[];joint=[]
 def choices(s,context):return (1,1) if context=='endpoint_alpha1' else tuple(selected[s,a,int(context[1:])] for a in ['C','D'])
 def compare(s,ca,da,context):
  c=stats[s,'C',ca];d=stats[s,'D',da];v={'seed':s,'context':context,'C_alpha':ca,'D_alpha':da,'source_pool_loss_diff_pp':d['pool_loss_pp']-c['pool_loss_pp'],'T11_diff_pp':d['T11_pp']-c['T11_pp']}
  for support in ['full','yes','no']:
   for k in ['mIoU','Shadow_recall','nonShadow_to_Shadow_FPR']:
    x=pooled[s,'D',da,'test',support][k];y=pooled[s,'C',ca,'test',support][k];v['test_'+support+'_'+k+'_diff']=None if x is None or y is None else x-y
  return v
 for context in contexts:
  rr=[]
  for s in SS:
   ca,da=choices(s,context);v=compare(s,ca,da,context);rr.append(v)
   if context!='endpoint_alpha1':
    for a,f in zip(['C','D'],[ca,da]):joint.append({'seed':s,'budget_pp':int(context[1:]),'arm':a,'alpha':f,**stats[s,a,f],**{f'test_{support}_mIoU':pooled[s,a,f,'test',support]['mIoU'] for support in ['full','yes','no']}})
  avg={'seed':'mean_of_two','context':context,'C_alpha':'/'.join(str(v['C_alpha']) for v in rr),'D_alpha':'/'.join(str(v['D_alpha']) for v in rr)}
  for k in rr[0]:
   if k not in avg:avg[k]=None if any(v[k] is None for v in rr) else float(np.mean([v[k] for v in rr]))
  comparisons.extend(rr+[avg])
 write('TABLE_B_DIFFERENCES.csv',comparisons);write('JOINT_RULE.csv',joint)
 # Per-support independent draws; same draw indices pair both models and both seeds.
 sourceids=sorted(base['source']);rng=np.random.default_rng(217023);indexes={p:rng.integers(0,len(gs),(10000,len(gs))) for p,gs in [('source',sourceids),('full',sorted(base['test'])),('yes',ids['yes']),('no',ids['no'])]};proof={'seed':217023,'draws':10000,'stream_order':['source107','full16','yes6','no10'],'supports':{k:{'groups':len(v[0]),'index_SHA':hashlib.sha256(v.astype('<i8').tobytes()).hexdigest()} for k,v in indexes.items()},'interval_definition':'within-seed nonlinear confusion metrics; paired index across seeds; mean differences; fixed source-selected alpha'};cis=[];first50={};bm=np.array([miou(base['source'][g]) for g in sourceids])
 for context in contexts:
  dd={};pp={}
  for s in SS:
   ca,da=choices(s,context);c=np.stack([cases[s,'C',ca,'source'][g] for g in sourceids]);d=np.stack([cases[s,'D',da,'source'][g] for g in sourceids]);ix=indexes['source'];cl=bm-miou(c);dl=bm-miou(d)
   ds={'source_pool_loss_diff_pp':miou(c[ix].sum(1))-miou(d[ix].sum(1)),'T11_diff_pp':np.sort(dl[ix],axis=1)[:,-11:].mean(1)-np.sort(cl[ix],axis=1)[:,-11:].mean(1)};point=compare(s,ca,da,context);ps={k:point[k] for k in ds}
   for support in ['full','yes','no']:
    keys=sorted(base['test']) if support=='full' else ids[support];c=np.stack([cases[s,'C',ca,'test'][g] for g in keys]);d=np.stack([cases[s,'D',da,'test'][g] for g in keys]);ix=indexes[support];k='test_'+support+'_mIoU_diff';ds[k]=miou(d[ix].sum(1))-miou(c[ix].sum(1));ps[k]=point[k]
   dd[str(s)]=ds;pp[str(s)]=ps
  dd['mean_of_two']={k:(dd[str(SS[0])][k]+dd[str(SS[1])][k])/2 for k in dd[str(SS[0])]};pp['mean_of_two']={k:(pp[str(SS[0])][k]+pp[str(SS[1])][k])/2 for k in pp[str(SS[0])]}
  for scope,ds in dd.items():
   for k,v in ds.items():
    lo,hi=np.quantile(v,[.025,.975]);cis.append({'context':context,'scope':scope,'metric':k,'point':pp[scope][k],'CI95_low':float(lo),'CI95_high':float(hi),'groups':107 if k in ['source_pool_loss_diff_pp','T11_diff_pp'] else 16 if 'full' in k else 6 if 'yes' in k else 10,'draws':10000,'seed':217023});first50[context+'/'+scope+'/'+k]=v[:50].tolist()
 write('PAIRED_CI.csv',cis);save('BOOTSTRAP_RULE.json',proof);save('BOOTSTRAP_FIRST50.json',first50)
 # Stream raw four-axis counts, report direct C->D changes separately by teacher correctness.
 agg={};rawrows=0
 for line in (EV/'JOINT_COUNTS.jsonl').open():
  v=json.loads(line);j=np.asarray(v['reference_teacher_C_D'],np.int64);assert j.shape==(3,3,3,3) and j.sum()==v['valid_pixels'];key=v['seed'],v['ROI'];agg[key]=agg.get(key,np.zeros_like(j))+j;rawrows+=1
 assert rawrows==1070
 direct=[];directflows=[];cc,dd=np.indices((3,3))
 for s in SS:
  assert len([g for seed,g in agg if seed==s])==107
  for roi in sourceids+['__pooled__']:
   j=sum(agg[s,g] for g in sourceids) if roi=='__pooled__' else agg[s,roi]
   for r,cl in enumerate(CS):
    for tier,keep in [('all',[0,1,2]),('correct',[r]),('wrong',[k for k in range(3) if k!=r])]:
     x=j[r,keep].sum(0);fixed=int(x[(cc!=r)&(dd==r)].sum());broken=int(x[(cc==r)&(dd!=r)].sum());ce=int(x[cc!=r].sum());de=int(x[dd!=r].sum());assert de-ce==broken-fixed
     direct.append({'seed':s,'ROI':roi,'reference_class':cl,'teacher_correctness':tier,'support':int(x.sum()),'C_wrong_D_right':fixed,'C_right_D_wrong':broken,'C_errors':ce,'D_errors':de,'D_minus_C_errors':de-ce})
     for dest in [k for k in range(3) if k!=r]:directflows.append({'seed':s,'ROI':roi,'reference_class':cl,'teacher_correctness':tier,'flow':cl+'->'+CS[dest],'C_pixels':int(x[cc==dest].sum()),'D_pixels':int(x[dd==dest].sum()),'removed_by_D':int(x[(cc==dest)&(dd!=dest)].sum()),'added_by_D':int(x[(cc!=dest)&(dd==dest)].sum())})
 write('SOURCE_DIRECT_CORRECTNESS.csv',direct);write('SOURCE_DIRECT_ERROR_FLOWS.csv',directflows)
 historical=list(csv.DictReader((R/'outputs/phase217/phase217_20261023/ENDPOINTS.csv').open(encoding='utf-8-sig')));write('HISTORICAL_SPARCS_REFERENCE_ONLY.csv',historical)
 mem=peak();assert mem<16*1024**3;save('ANALYSIS_RECEIPT.json',{'seconds':time.monotonic()-clock,'CPU_peak_bytes':mem,'units':0,'candidates':20,'pool_rows':len(poolrows),'group_rows':len(per),'bootstrap_rows':len(cis),'direct_rows':len(direct),'code_SHA':sha(Path(__file__))});print(json.dumps([v for v in comparisons if v['context']=='endpoint_alpha1']))
if __name__=='__main__':main()

"""Seal common no-label support, then join local references once; no calibration."""
import json,csv,collections,hashlib,datetime,shutil
from pathlib import Path
import numpy as np
from phase89_cpu_ap import average_precision_score as independent_ap
R=Path(__file__).resolve().parents[2];O=R/'src/research_results/phase112_20261014';D=R/'data/pixbox_s2_cmix_5036991/phase112_20261014';U=R/'outputs/phase112/phase112_20261014'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def table(p,rows):
 with p.open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ap(rr,name):
 y=np.array([r['reference']=='positive_ground_cloud_shadow' for r in rr]);s=np.array([r[name] for r in rr],np.float32)
 if not y.any() or not (~y).any():return None,None
 order=np.argsort(-s,kind='stable');s=s[order];yy=y[order];ends=np.flatnonzero(np.r_[s[1:]!=s[:-1],True]);tp=np.cumsum(yy,dtype=np.int64)[ends];v=float(np.sum(np.diff(np.r_[0,tp])*tp/(ends+1)/y.sum()))
 cpu=independent_ap(y,np.array([r[name] for r in rr],np.float32));assert abs(v-cpu)<1e-12
 return v,cpu
def operating(rr,name,t):
 y=np.array([r['reference']=='positive_ground_cloud_shadow' for r in rr]);pred=np.array([r[name]>=t for r in rr]);p=int(y.sum());n=len(rr)-p;tp=int((y&pred).sum());fp=int((~y&pred).sum())
 d={'positive':p,'negative':n,'TP':tp,'FP':fp,'recall':tp/p if p else None,'precision':tp/(tp+fp) if tp+fp else None,'negative_false_positive_fraction':fp/n if n else None}
 for kind in ['negative_surface','negative_cloud']:
  m=np.array([r['reference']==kind for r in rr]);nn=int(m.sum());ff=int((m&pred).sum());d.update({kind+'_n':nn,kind+'_FP':ff,kind+'_false_positive_fraction':ff/nn if nn else None})
 return d
def main():
 lock=read(O/'PRE_EXECUTION_LOCK.json')
 for n,h in lock['source_sha256'].items():assert sha(R/n)==h,n
 base=R/'data/pixbox_s2_cmix_5036991/phase105_20261013/point_scores_no_labels.json';spectral=R/'data/pixbox_s2_cmix_5036991/phase106_20261013/point_scores_no_labels.json'
 assert sha(base)==read(R/'src/research_results/phase105_20261013/SCORES_SEALED.json')['score_sha256']
 assert sha(spectral)==read(R/'src/research_results/phase106_20261013/SCORES_SEALED.json')['score_sha256']
 source=read(base)['rows'];spec={r['point_id']:r for r in read(spectral)['rows']};assert len(source)==len(spec)==9500
 receipt=read(U/'SCORES_SEALED.json');omni={};raw_hashes={}
 for pid in [x['product_id'] for x in lock['products']]:
  path=D/'omni'/pid/'UNLABELLED_POINT_SCORES.json';raw_hashes[pid]=sha(path);assert raw_hashes[pid]==receipt['score_sha256'][pid]
  for q in read(path)['points']:
   assert q['point_id'] not in omni;omni[q['point_id']]=q
 assert len(omni)==9500 and set(omni)=={r['point_id'] for r in source}
 rows=[];loss=collections.defaultdict(lambda:{'original_points':0,'original_input_invalid':0,'omni_input_invalid':0,'omni_no_coverage':0,'new_loss_among_original_valid':0,'common_valid':0})
 for b in source:
  q=omni[b['point_id']];z=spec[b['point_id']];assert all(str(q[k])==str(b[k]) for k in ['point_id','product_id','row20','col20']);assert z['A']==b['Source'] and z['input_valid']==b['input_valid']
  for k in ['product_id','group_id','row20','col20']:assert str(z[k])==str(b[k])
  ok=bool(b['input_valid'] and q['valid'] and q['original_valid'] and q['author_valid'] and q['coverage'])
  for n,v in [('G7',b['G7']),('Source',b['Source']),('Source_spectral',z['B']),('Omni',q['shadow_confidence'])]:
   if ok:assert v is not None and np.isfinite(v)
  r={k:b[k] for k in ['point_id','product_id','group_id','row20','col20']};r.update(G7=b['G7'],Source=b['Source'],Source_spectral=z['B'],Omni=q['shadow_confidence'],common_valid=ok);rows.append(r)
  l=loss[b['group_id']];l['original_points']+=1;l['original_input_invalid']+=int(not b['input_valid']);l['omni_input_invalid']+=int(not(q['original_valid'] and q['author_valid']));l['omni_no_coverage']+=int(not q['coverage']);l['new_loss_among_original_valid']+=int(b['input_valid'] and not ok);l['common_valid']+=int(ok)
 # Freeze all IDs and masks without reference semantics or correctness filtering.
 write(D/'COMMON_SUPPORT_PRIVATE.json',{'rows':[{'point_id':r['point_id'],'common_valid':r['common_valid']} for r in rows]})
 write(O/'COMMON_SUPPORT_LOCK.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before_reference_join':True,'original_points':9500,'support_sha256':sha(D/'COMMON_SUPPORT_PRIVATE.json'),'score_sha256':raw_hashes,'base_cache_sha256':sha(base),'spectral_cache_sha256':sha(spectral),'common_valid':sum(r['common_valid'] for r in rows),'no_prediction_correctness_filter':True})
 table(O/'support_losses.csv',[dict(group_id=k,**v) for k,v in sorted(loss.items())])
 refpath=R/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json';assert sha(refpath)==read(R/'src/research_results/phase105_20261013/task_lock.json')['phase103_reference_sha256']
 refs={r['point_ids'][0]:r for r in read(refpath)};assert set(refs)==set(omni)
 groups=collections.defaultdict(list);joined=[]
 for r in rows:
  ref=refs[r['point_id']];assert all(str(r[k])==str(ref[k]) for k in ['product_id','group_id','row20','col20']);r=dict(r,reference=ref['reference']);joined.append(r)
  if r['common_valid'] and ref['reference']!='ignore':groups[r['group_id']].append(r)
 table(D/'joined_reference_scores_private.csv',joined)
 allrr=[r for rr in groups.values() for r in rr];group_table=[];cpu_delta=[]
 for gid,rr in sorted(groups.items()):
  pos=sum(r['reference']=='positive_ground_cloud_shadow' for r in rr);neg=len(rr)-pos;g=dict(group_id=gid,tiles=','.join(sorted({r['product_id'].split('_T')[1][:5] for r in rr})),positive=pos,negative=neg)
  cp={}
  for name in ['G7','Omni','Source','Source_spectral']:g[name+'_AP'],cp[name]=ap(rr,name)
  g['delta_AP_pp']=(g['G7_AP']-g['Omni_AP'])*100 if g['G7_AP'] is not None else None
  g['change']='NA' if g['delta_AP_pp'] is None else 'improved' if g['delta_AP_pp']>1 else 'declined' if g['delta_AP_pp']< -1 else 'stable'
  if cp['G7'] is not None:cpu_delta.append((cp['G7']-cp['Omni'])*100)
  for name in ['G7','Omni']:g.update({name+'_'+k:v for k,v in operating(rr,name,lock['thresholds'][name]).items()})
  g['delta_recall_pp']=(g['G7_recall']-g['Omni_recall'])*100 if pos else None;g['delta_negative_FP_pp']=(g['G7_negative_false_positive_fraction']-g['Omni_negative_false_positive_fraction'])*100 if neg else None
  group_table.append(g)
 table(O/'per_group_results.csv',group_table)
 positive_groups=[g for g in group_table if g['positive']>0];support_ok=len(positive_groups)>=8 and all(g['positive']>=5 and g['negative']>=20 for g in positive_groups) and all(g['G7_AP'] is not None for g in positive_groups)
 primary={'comparison':'G7_153 minus Omni','unit':'related group; products merged within group','support_pass':support_ok,'n_AP_groups':len(positive_groups),'resamples':10000,'seed':112014,'descriptive_interval':True,'rule':'at least8 positive groups, each>=5positive/20negative; CI lower>0 AND width<=20pp','passed':False}
 bootstrap_error=None
 if support_ok:
  delta=np.array([g['delta_AP_pp'] for g in positive_groups]);idx=np.random.default_rng(112014).integers(0,len(delta),(10000,len(delta)));ci=np.quantile(delta[idx].mean(1),[.025,.975]);scalar=sorted(sum(cpu_delta[i] for i in row)/len(delta) for row in idx)
  def quant(q):
   t=q*9999;i=int(t);return scalar[i]+(t-i)*(scalar[min(i+1,9999)]-scalar[i])
  bootstrap_error=float(np.max(np.abs(ci-[quant(.025),quant(.975)])));assert bootstrap_error<1e-12
  primary.update(mean_delta_AP_pp=float(delta.mean()),CI95_pp=ci.tolist(),width_pp=float(ci[1]-ci[0]),passed=bool(ci[0]>0 and ci[1]-ci[0]<=20),improved=sum(g['change']=='improved' for g in positive_groups),stable=sum(g['change']=='stable' for g in positive_groups),declined=sum(g['change']=='declined' for g in positive_groups),worst_group=min(positive_groups,key=lambda g:g['delta_AP_pp'])['tiles'],worst_delta_AP_pp=float(delta.min()))
 primary['conclusion']='本次探索支持G7排序增量' if primary['passed'] else '增益未获本次探索支持' if support_ok else '共同支持不足，不判断主要终点'
 write(O/'primary_endpoint.json',primary)
 summary=[]
 for name in ['G7','Omni','Source','Source_spectral']:
  vals=[g[name+'_AP'] for g in group_table if g[name+'_AP'] is not None];pooled,_=ap(allrr,name);s={'score':name,'macro_sampled_AP_pp':float(np.mean(vals))*100 if vals else None,'pooled_sampled_AP_pp':pooled*100 if pooled is not None else None,'AP_groups':len(vals)};summary.append(s)
 table(O/'four_score_summary.csv',summary)
 operations=[]
 for name in ['G7','Omni']:
  s={'score':name,'frozen_threshold':lock['thresholds'][name],'macro_recall':float(np.mean([g[name+'_recall'] for g in group_table if g[name+'_recall'] is not None])),'macro_precision':float(np.mean([g[name+'_precision'] for g in group_table if g[name+'_precision'] is not None])),'macro_negative_FP_fraction':float(np.mean([g[name+'_negative_false_positive_fraction'] for g in group_table if g[name+'_negative_false_positive_fraction'] is not None])),'max_group_negative_FP_fraction':max(g[name+'_negative_false_positive_fraction'] for g in group_table if g[name+'_negative_false_positive_fraction'] is not None)}
  for kind in ['negative_surface','negative_cloud']:
   vals=[g[name+'_'+kind+'_false_positive_fraction'] for g in group_table if g[name+'_'+kind+'_false_positive_fraction'] is not None];s['macro_'+kind+'_FP_fraction']=float(np.mean(vals)) if vals else None
  s.update({'pooled_'+k:v for k,v in operating(allrr,name,lock['thresholds'][name]).items()});operations.append(s)
 table(O/'two_operating_points.csv',operations)
 write(O/'operating_differences.json',{'G7_minus_Omni':{k:operations[0][k]-operations[1][k] for k in ['macro_recall','macro_precision','macro_negative_FP_fraction','max_group_negative_FP_fraction']},'not_matched_false_positive_budget':True,'Omni_threshold_not_author_argmax':True,'not_full_scene_FPR':True})
 write(O/'CPU_STATISTICS_ACCEPTANCE.json',{'status':'passed','independent_AP_all_four_scores_groups_and_pool':True,'ties_aggregated':True,'scalar_bootstrap_max_abs_error':bootstrap_error,'original_points':9500,'common_valid':sum(r['common_valid'] for r in rows),'ignored_on_common_support':sum(r['common_valid'] and r['reference']=='ignore' for r in joined),'positive':sum(r['reference']=='positive_ground_cloud_shadow' for r in allrr),'negative':sum(r['reference']!='positive_ground_cloud_shadow' for r in allrr),'no_positive_groups':sum(g['positive']==0 for g in group_table),'private_join_sha256':sha(D/'joined_reference_scores_private.csv'),'extra_model_calls':0,'threshold_scans':0})
 print(json.dumps(primary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

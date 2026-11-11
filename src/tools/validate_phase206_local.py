"""Independent Phase206 arithmetic/state/support/cost acceptance; no model imports."""
import ast,csv,hashlib,json,math,sys,time
from pathlib import Path
R=Path('D:/Cloud-Adapter-light');O=R/'outputs/phase206/phase206_20261019';D=O/'delivery'
sys.path.insert(0,str(R/'outputs/phase202/phase202_20261017/software'))
import numpy as np
CHECKS=0
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def rows(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def check(x):
 global CHECKS
 assert x;CHECKS+=1

def main():
 start=time.time();check((O/'LOCAL_EXIT_CODE').read_text().strip()=='0');check(load(O/'CPU_ACCEPTANCE.json')['status']=='passed')
 manifest=load(O/'INPUT_DELIVERY_LOCK.json')['manifest']
 for n,h in manifest['files'].items():check(sha(D/n)==h)
 for stage in ('reconstruction','infer'):check(load(D/stage/'SCORES_SEALED.json')['all_model_parameters_buffers_unchanged'])
 rec=load(D/'RECONSTRUCTION_ACCEPTANCE.json');check(rec['status']=='passed' and len(rec['rows'])==12)
 for r in rec['rows']:check(r['max_score_error']<=1e-5 and r['decisions_support_exact'])
 events=[json.loads(q) for q in (D/'CALL_LEDGER.jsonl').read_text().splitlines()];check(len(events)==6002);check([q['global_call'] for q in events]==list(range(1,6003)));check(sum(q['stage']=='infer' for q in events)==5730);check(all(q['kind']!='training' for q in events))
 white=load(D/'SMP_WHITELIST.json');W=set(white['trainable']);B=set(white['decoder_state'])-W;check(len(W)==32 and len(B)==30 and len({k.rsplit('.',1)[0] for k in B})==10);enc=[]
 for seed in (12201,12301):
  for t in ('10','01'):
   s=load(D/'infer'/f'STATE_{seed}_{t}.json');enc.append(s['encoder_sha256']);check(len([k for k in s['tensor_sources'] if k.startswith('encoder.')])==382)
   for k,origin in s['tensor_sources'].items():check(origin==('seed'+str(seed)+'_endpoint' if ((k in W and t[0]=='1') or (k in B and t[1]=='1')) else 'official_R4'))
 check(len(set(enc))==1)
 for r in load(D/'ACTUAL_WEIGHT_ACCEPTANCE.json')['rows']:check(r['initial_equals_official'] and r['changed_parameters']==32 and r['changed_buffers']==30)
 for r in load(D/'infer/SCENE_RECEIPTS.json'):check(r['raw_ensemble_mean_error']<=1e-5 and r['official_first_member_confidence_error']<=1e-5 and r['all_postprocessing_error']<=1e-5 and r['coverage_gradient_exact'])
 allcomparisons=[];cohort_counts=[]
 for pool in ('Catalogue','PixBox'):
  per=rows(pool+'_PER_GROUP.csv');table=rows(pool+'_TABLE2.csv');summary=load(O/(pool+'_ALL_METRICS.json'));check(len(summary)==14 and len(table)==(128 if pool=='Catalogue' else 40));lookup={(r['strategy'],r['group_id']):r for r in per}
  for row in per:
   m=np.asarray(ast.literal_eval(row['confusion']),np.int64);check(m.shape==(3,3) and (m>=0).all());p=int(row['positive']);n=int(row['negative']);tp=int(row['TP']);fp=int(row['FP']);check(p==m[2].sum() and n==m[:2].sum() and tp==m[2,2] and fp==m[:2,2].sum());check(int(row['FN'])==p-tp and int(row['TN'])==n-fp);check(row['AP']=='' if p==0 else 0<=float(row['AP'])<=1)
   check(row['precision']=='' if tp+fp==0 else abs(float(row['precision'])-tp/(tp+fp))<1e-12)
  for s in summary:
   pr=[r for r in per if r['strategy']==s['strategy']];mat=np.sum([ast.literal_eval(r['confusion']) for r in pr],axis=0);check(mat.tolist()==s['confusion']);check(abs(np.mean([float(r['AP']) for r in pr if r['AP']!=''])*100-s['macro_AP_pp'])<1e-10);check(s['positive']==(655961 if pool=='Catalogue' else 379) and s['negative']==(32767527 if pool=='Catalogue' else 8190));check(abs(max(float(r['negative_shadow_fraction']) for r in pr)*100-s['max_group_negative_shadow_fraction_pp'])<1e-10)
   if pool=='Catalogue':
    union=mat.sum(0)+mat.sum(1)-np.diag(mat);iou=[mat[i,i]/union[i] for i in range(3) if union[i]>0];check(abs(float(np.mean(iou))*100-s['Catalogue_pooled_mIoU_pp'])<1e-10)
   else:check(s['Catalogue_pooled_mIoU_pp'] is None)
  def state_names(seed,fam):return {'00':'R4' if fam=='member' else 'E4','11':('C4-' if fam=='member' else 'E4-')+str(seed),'10':('M_' if fam=='member' else 'E_')+str(seed)+'_10','01':('M_' if fam=='member' else 'E_')+str(seed)+'_01'}
  for r in table:
   ns=state_names(int(r['seed']),r['family']);q={t:lookup[n,r['group_id']] for t,n in ns.items()}
   if not r['AP00_pp']:check(all(r[k+'_pp']=='' for k in ('10-11','01-00','10-00','I')));continue
   a={t:float(q[t]['AP'])*100 for t in ns}
   for t in ns:check(abs(a[t]-float(r['AP'+t+'_pp']))<1e-10)
   expected={'10-11':a['10']-a['11'],'01-00':a['01']-a['00'],'10-00':a['10']-a['00'],'I':a['11']+a['00']-a['10']-a['01']}
   for k,v in expected.items():check(abs(v-float(r[k+'_pp']))<1e-10)
  for rr in rows(pool+'_REPEATED_CHANGES.csv'):
   if rr['seed12201_delta_AP_pp']=='':check(rr['both_damaged']=='');continue
   dv=[float(rr['seed'+str(seed)+'_delta_AP_pp']) for seed in (12201,12301)];check((rr['both_damaged']=='True')==all(v< -1 for v in dv));check((rr['both_improved']=='True')==all(v>1 for v in dv));check((rr['both_stable']=='True')==all(abs(v)<=1 for v in dv))
  for row in load(O/(pool+'_INTERVALS.json'))['rows']:
   if row['family']=='member_vs_E4':
    group=rows(pool+'_MEMBER_'+row['contrast'][:2]+'_VS_E4_GROUPS.csv');values=[float(q['seed'+str(row['seed'])+'_delta_AP_pp']) for q in group if q['seed'+str(row['seed'])+'_delta_AP_pp']!='']
   else:values=[float(q[row['contrast']+'_pp']) for q in table if q['family']==row['family'] and int(q['seed'])==row['seed'] and q[row['contrast']+'_pp']!='']
   a=np.asarray(values,np.float64);indices=np.random.default_rng(206019).integers(len(a),size=(10000,len(a)));means=np.asarray([math.fsum(float(a[i]) for i in q)/len(q) for q in indices]);check(abs(math.fsum(values)/len(values)-row['mean_delta_AP_pp'])<1e-10);check(np.max(abs(np.quantile(means,[.025,.975])-row['CI95_pp']))<1e-10);check(row['n_groups']==(7 if pool=='Catalogue' else 8));check(row['bootstrap_seed']==206019 and row['bootstrap_draws']==10000);allcomparisons.append(row)
  for family in ('member','ensemble'):
   for con in ('10-11','01-00','10-00','11-00'):
    rr=[r for r in rows(pool+'_REPEATED_CHANGES.csv') if r['family']==family and r['contrast']==con];cohort_counts.append(dict(pool=pool,family=family,contrast=con,AP_groups=sum(r['positive']!='0' for r in rr),both_improved=sum(r['both_improved']=='True' for r in rr),both_damaged=sum(r['both_damaged']=='True' for r in rr),both_stable=sum(r['both_stable']=='True' for r in rr)))
 local=load(O/'LOCAL_STATUS.json');remote=load(D/'CONTROLLER_STATUS.json');receipt=load(O/'DELIVERY_RECEIPT.json');size=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());summarydir=R/'src/research_results/phase206_20261019';summary_size=sum(p.stat().st_size for p in summarydir.rglob('*') if p.is_file()) if summarydir.exists() else 0;combined=receipt['remote_new_bytes']+size+summary_size;check(combined<6000000000 and local['peak_bytes']<=16*1024**3)
 cost=dict(network_calls=6002,training=0,fits=0,reconstruction_calls=272,main_calls=5730,GPU_stage_seconds=remote['stage_seconds_total'],CPU_seconds_cumulative=local['cumulative_CPU_seconds'],local_peak_bytes=local['peak_bytes'],remote_peak_CPU_bytes=max(load(D/stage/'RUN_STATUS.json')['rss_peak_bytes'] for stage in ('reconstruction','infer')),remote_peak_GPU_reserved_bytes=max(load(D/stage/'RUN_STATUS.json')['gpu_peak_reserved'] for stage in ('reconstruction','infer')),minimum_host_free_bytes=min(load(D/stage/'RUN_STATUS.json')['host_min_free_bytes'] for stage in ('reconstruction','infer')),remote_new_bytes=receipt['remote_new_bytes'],local_output_bytes=size,combined_before_final_reports=combined,combined_final_report_allowance_bytes=10000000,combined_conservative_upper_bytes=combined+10000000,ledger_sha256=sha(D/'CALL_LEDGER.jsonl'),prior_CPU_failure='NA precision serialized asNaN; original266.36s retained; Catalogue tables reused unchanged; PixBox-only CPU rerun;0newforwards',local_first_pass_CPU_peak_unknown=False,old205failed_gate_peak='unknown; not imputed to206')
 check(cost['combined_conservative_upper_bytes']<6000000000 and cost['GPU_stage_seconds']<14400 and cost['minimum_host_free_bytes']>=4000000000)
 save('COST_RECEIPT.json',cost);save('COHORT_REPEATED_COUNTS.json',cohort_counts);save('INDEPENDENT_ACCEPTANCE.json',dict(status='passed',checks=CHECKS,seconds=time.time()-start,code_sha256=sha(Path(__file__)),bootstrap_independent_fsum=True,reference_only_local=True,no_neural_calls=True,no_fit=True,undefined_precision_retained_NA=True))
 print(json.dumps(dict(status='passed',checks=CHECKS,cost=cost)),flush=True)
if __name__=='__main__':main()

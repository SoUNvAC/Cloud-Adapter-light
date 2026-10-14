"""Fixed within-Omni-bin pair ranking diagnostic; no fit/forward/download."""
import argparse,bisect,collections,csv,datetime,hashlib,json,time,shutil
from pathlib import Path
import numpy as np
from phase89_cpu_ap import average_precision_score as independent_ap
R=Path(__file__).resolve().parents[2];O=R/'src/research_results/phase113_20261014';U=R/'outputs/phase113/phase113_20261014';D=R/'data/pixbox_s2_cmix_5036991/phase112_20261014';P=R/'src/research_results/phase112_20261014'
NAMES=['Source_spectral','Omni','Source'];EDGES=[0.,.2,.4,.6,.8,1.]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvread(p):
 with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
 with p.open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def bin_id(s):
 assert 0<=s<=1 and np.isfinite(s)
 return min(bisect.bisect_right(EDGES,s)-1,4)
def pair_numerator(positive,negative):
 # Twice pair credit, exact integer: wins=2, ties=1, losses=0.
 n=np.sort(np.array(negative,dtype=np.float32));p=np.array(positive,dtype=np.float32)
 lower=np.searchsorted(n,p,side='left');upper=np.searchsorted(n,p,side='right')
 return int((lower+upper).sum())
def scalar_pairs(positive,negative):return sum(2 if p>n else 1 if p==n else 0 for p in positive for n in negative)
def test():
 assert [bin_id(x) for x in [0,.199999,.2,.4,.6,.8,1.]]==[0,0,1,2,3,4,4]
 assert pair_numerator([.5,.9],[.5,.1])==7
 for seed in range(10):
  g=np.random.default_rng(seed);a=g.integers(0,5,11).astype(np.float32);b=g.integers(0,5,13).astype(np.float32);assert pair_numerator(a,b)==scalar_pairs(a,b)
 return {'fixed_bin_boundaries':True,'hand_pair_tie_case':True,'random_exact_integer_pair_replays':10,'model_calls':0}
def freeze():
 assert not O.exists(),'Preserve any previous attempt'
 files=[R/'src/research_plans/PHASE113_STRONG_BASE_COMPLEMENT_DIAGNOSTIC_20261014.md',Path(__file__),R/'src/tools/phase89_cpu_ap.py',P/'COMMON_SUPPORT_LOCK.json',P/'CPU_STATISTICS_ACCEPTANCE.json',P/'four_score_summary.csv',P/'per_group_results.csv',R/'src/research_results/phase106_20261013/model_lock.json',R/'src/research_results/phase106_20261013/input_score_lock.json']
 control=R/'outputs/phase67/phase67_20261010/delivery/control_lock.json';model=read(R/'src/research_results/phase106_20261013/model_lock.json')
 assert sha(control)==model['control_sha256'];assert read(control)['models']['Source_spectral']==model['model']
 source=R/'data/pixbox_s2_cmix_5036991/phase105_20261013/point_scores_no_labels.json';spectral=R/'data/pixbox_s2_cmix_5036991/phase106_20261013/point_scores_no_labels.json';joined=D/'joined_reference_scores_private.csv';support=D/'COMMON_SUPPORT_PRIVATE.json';prior=read(P/'COMMON_SUPPORT_LOCK.json')
 assert sha(source)==prior['base_cache_sha256'] and sha(spectral)==prior['spectral_cache_sha256'] and sha(support)==prior['support_sha256']
 assert sha(joined)==read(P/'CPU_STATISTICS_ACCEPTANCE.json')['private_join_sha256']
 for pid,h in prior['score_sha256'].items():assert sha(D/'omni'/pid/'UNLABELLED_POINT_SCORES.json')==h
 O.mkdir(parents=True);U.mkdir(parents=True,exist_ok=True);(O/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n')
 lock={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':{str(p.relative_to(R)).replace('\\','/'):sha(p) for p in files},'private_inputs_sha256':{str(p.relative_to(R)).replace('\\','/'):sha(p) for p in [source,spectral,joined,support,control]},'omni_score_sha256':prior['score_sha256'],'bin_edges':EDGES,'intervals':'[0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1]','pair_rule':'only same group/same fixed bin; win2/tie1/loss0; denominator twice summed bin pair counts','score_names':NAMES,'primary':'Source_spectral minus Omni','participating_positive_min':5,'participating_negative_min':20,'minimum_groups':4,'bootstrap_seed':113014,'resamples':10000,'minimum_mean_delta_pp':2,'CI_lower_must_exceed':0,'thresholds':{'Omni':.5,'Source_spectral':model['model']['threshold']},'post_hoc_old_pool':True,'new_fits':0,'new_forwards':0,'downloads':0,'threshold_scans':0}
 write(O/'INPUT_BIN_LOCK.json',lock);write(O/'SYNTHETIC_CPU_TESTS.json',test());print('Phase113 locked; no real diagnostics computed')
def evaluate():
 started=time.monotonic();lock=read(O/'INPUT_BIN_LOCK.json')
 for n,h in dict(lock['source_sha256'],**lock['private_inputs_sha256']).items():assert sha(R/n)==h,n
 for pid,h in lock['omni_score_sha256'].items():assert sha(D/'omni'/pid/'UNLABELLED_POINT_SCORES.json')==h
 # Reuse exactly the old sealed scores; CSV round-trips must match raw JSON values.
 base={r['point_id']:r for r in read(R/'data/pixbox_s2_cmix_5036991/phase105_20261013/point_scores_no_labels.json')['rows']};spec={r['point_id']:r for r in read(R/'data/pixbox_s2_cmix_5036991/phase106_20261013/point_scores_no_labels.json')['rows']};omni={}
 for pid in lock['omni_score_sha256']:
  for r in read(D/'omni'/pid/'UNLABELLED_POINT_SCORES.json')['points']:assert r['point_id'] not in omni;omni[r['point_id']]=r
 common={r['point_id']:r['common_valid'] for r in read(D/'COMMON_SUPPORT_PRIVATE.json')['rows']};allrows=csvread(D/'joined_reference_scores_private.csv');assert len(allrows)==9500
 groups=collections.defaultdict(list)
 for row in allrows:
  k=row['point_id'];assert (row['common_valid']=='True')==common[k]
  if not common[k] or row['reference']=='ignore':continue
  row=dict(row)
  for name,value in [('Source',base[k]['Source']),('Source_spectral',spec[k]['B']),('Omni',omni[k]['shadow_confidence'])]:
   assert float(row[name])==value and float(np.float32(value))==value;row[name]=value
  row['positive']=row['reference']=='positive_ground_cloud_shadow';row['bin']=bin_id(row['Omni']);groups[row['group_id']].append(row)
 assert len(groups)==10 and sum(r['positive'] for rr in groups.values() for r in rr)==379 and sum(not r['positive'] for rr in groups.values() for r in rr)==8188
 # First reproduce the previous AP, including NA groups and exact score ties.
 old={r['group_id']:r for r in csvread(P/'per_group_results.csv')};oldsummary={r['score']:r for r in csvread(P/'four_score_summary.csv')};replays=[]
 for gid,rr in sorted(groups.items()):
  y=np.array([r['positive'] for r in rr])
  for name in NAMES:
   value=independent_ap(y,np.array([r[name] for r in rr],np.float32)) if y.any() and (~y).any() else None;expected=old[gid][name+'_AP'];assert (value is None)==(expected=='')
   if value is not None:assert abs(value-float(expected))<1e-12
   replays.append({'group_id':gid,'score':name,'AP':value,'prior_AP_equal':True})
 flat=[r for rr in groups.values() for r in rr]
 for name in NAMES:
  value=independent_ap([r['positive'] for r in flat],[r[name] for r in flat]);assert abs(value*100-float(oldsummary[name]['pooled_sampled_AP_pp']))<1e-10
  macro=np.mean([r['AP'] for r in replays if r['score']==name and r['AP'] is not None])*100;assert abs(macro-float(oldsummary[name]['macro_sampled_AP_pp']))<1e-10
 write(O/'PRIOR_AP_REPLAY.json',{'status':'passed','per_group':replays,'pooled_and_macro_three_scores_equal':True})
 perbin=[];pergroup=[];cross=[];cpu_deltas=[];pair_total=0
 for gid,rr in sorted(groups.items()):
  pos=sum(r['positive'] for r in rr);neg=len(rr)-pos;tile=old[gid]['tiles'];participating_p=participating_n=pairs=0;numerators={n:0 for n in NAMES};independents={n:0 for n in NAMES}
  for bi in range(5):
   subset=[r for r in rr if r['bin']==bi];positive=[r for r in subset if r['positive']];negative=[r for r in subset if not r['positive']];den=len(positive)*len(negative);pairs+=den
   if den:participating_p+=len(positive);participating_n+=len(negative)
   br={'group_id':gid,'tile':tile,'bin':bi,'lower':EDGES[bi],'upper':EDGES[bi+1],'upper_inclusive':bi==4,'positive':len(positive),'negative':len(negative),'pair_count':den,'participating_positive':len(positive) if den else 0,'participating_negative':len(negative) if den else 0,'participating_points':len(subset) if den else 0}
   for name in NAMES:
    a=[r[name] for r in positive];b=[r[name] for r in negative];num=pair_numerator(a,b);cpu=scalar_pairs(a,b);assert num==cpu;numerators[name]+=num;independents[name]+=cpu;br[name+'_twice_pair_credit']=num;br[name+'_conditional_AUC']=num/(2*den) if den else None
   perbin.append(br)
  eligible=pos>0 and participating_p>=5 and participating_n>=20 and pairs>0
  gr={'group_id':gid,'tile':tile,'positive':pos,'negative':neg,'participating_positive':participating_p,'participating_negative':participating_n,'participating_points':participating_p+participating_n,'pair_count':pairs,'eligible':eligible,'NA_reason':None if eligible else 'no_reference_shadow' if pos==0 else 'insufficient_participating_support'}
  for name in NAMES:gr[name+'_conditional_AUC']=numerators[name]/(2*pairs) if eligible else None;gr[name+'_twice_pair_credit']=numerators[name]
  gr['delta_conditional_AUC_pp']=(gr['Source_spectral_conditional_AUC']-gr['Omni_conditional_AUC'])*100 if eligible else None
  if eligible:cpu_deltas.append((independents['Source_spectral']-independents['Omni'])/(2*pairs)*100)
  pergroup.append(gr);pair_total+=pairs
  for kind in ['positive_ground_cloud_shadow','negative_surface','negative_cloud']:
   cell={'group_id':gid,'tile':tile,'reference_stratum':kind,'n':0,'Omni0_spectral0':0,'Omni0_spectral1':0,'Omni1_spectral0':0,'Omni1_spectral1':0}
   for row in rr:
    if row['reference']!=kind:continue
    a=int(row['Omni']>=lock['thresholds']['Omni']);b=int(row['Source_spectral']>=lock['thresholds']['Source_spectral']);cell['n']+=1;cell[f'Omni{a}_spectral{b}']+=1
   assert sum(cell[k] for k in ['Omni0_spectral0','Omni0_spectral1','Omni1_spectral0','Omni1_spectral1'])==cell['n'];cross.append(cell)
 csvwrite(O/'per_bin_support_and_AUC.csv',perbin);csvwrite(O/'per_group_conditional_AUC.csv',pergroup);csvwrite(O/'fixed_error_flow_per_group.csv',cross)
 eligible=[g for g in pergroup if g['eligible']];primary={'comparison':'Source_spectral minus Omni within fixed Omni bins','n_groups':len(eligible),'minimum_groups':4,'support_pass':len(eligible)>=4,'passed':False,'seed':113014,'resamples':10000,'post_hoc_descriptive':True,'rule':'at least4 eligible groups AND mean>=2pp AND 95% CI lower>0','total_comparable_pairs_all_groups':pair_total};bootstrap_error=None
 if len(eligible)>=4:
  delta=np.array([g['delta_conditional_AUC_pp'] for g in eligible]);idx=np.random.default_rng(113014).integers(0,len(delta),(10000,len(delta)));ci=np.quantile(delta[idx].mean(1),[.025,.975]);scalar=sorted(sum(cpu_deltas[i] for i in row)/len(delta) for row in idx)
  def quant(q):
   t=q*9999;i=int(t);return scalar[i]+(t-i)*(scalar[min(i+1,9999)]-scalar[i])
  bootstrap_error=float(np.max(np.abs(ci-[quant(.025),quant(.975)])));assert bootstrap_error<1e-12
  primary.update(mean_delta_conditional_AUC_pp=float(delta.mean()),CI95_pp=ci.tolist(),width_pp=float(ci[1]-ci[0]),passed=bool(delta.mean()>=2 and ci[0]>0),macro_conditional_AUC={n:float(np.mean([g[n+'_conditional_AUC'] for g in eligible])) for n in NAMES})
 primary['conclusion']='存在值得另立拟合试验的补充线索' if primary['passed'] else '支持不足，关闭本组补充评分假说' if len(eligible)<4 else '未满足固定补充线索判据，关闭本组补充评分假说'
 write(O/'PRIMARY_DIAGNOSTIC.json',primary)
 pooled=[]
 for kind in ['positive_ground_cloud_shadow','negative_surface','negative_cloud']:
  row={'reference_stratum':kind}
  for key in ['n','Omni0_spectral0','Omni0_spectral1','Omni1_spectral0','Omni1_spectral1']:row[key]=sum(x[key] for x in cross if x['reference_stratum']==kind)
  pooled.append(row)
 csvwrite(O/'fixed_error_flow_pooled.csv',pooled)
 write(O/'CPU_ACCEPTANCE.json',{'status':'passed','pair_credit_exact_integer_comparison':True,'independent_brute_force_pairs_all_scores_all50_bins':True,'total_pairs_all_groups':pair_total,'NA_no_shadow_groups':sum(g['positive']==0 for g in pergroup),'bootstrap_scalar_max_absolute_error':bootstrap_error,'old_AP_three_scores_reproduced':True,'synthetic_tests':test(),'same_pairs_same_groups_all_scores':True})
 write(O/'COST.json',{'seconds':time.monotonic()-started,'new_forwards':0,'new_fits':0,'new_download_bytes':0,'threshold_scans':0,'bins_searched':0,'new_candidates':0,'groups':10,'points':8567,'protocol_unchanged':True,'labels_or_scores_in_git':False})
 print(json.dumps(primary,ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','evaluate','test']);args=p.parse_args()
 if args.mode=='test':print(test())
 elif args.mode=='freeze':freeze()
 else:evaluate()

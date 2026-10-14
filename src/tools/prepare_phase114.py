"""Fixed sampling and anonymization. Never print or publish the private reveal."""
import csv,json,hashlib,collections,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'src/research_results/phase114_20261014';U=R/'outputs/phase114/phase114_20261014';D=R/'data/pixbox_s2_cmix_5036991/phase114_20261014'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def key(r):return hashlib.sha256(('PHASE114_AUDIT|'+r['point_id']).encode()).hexdigest()
def main():
 assert not O.exists() and not D.exists(),'Preserve previous attempt'
 prior=R/'src/research_results/phase112_20261014';data=R/'data/pixbox_s2_cmix_5036991/phase112_20261014';joined=data/'joined_reference_scores_private.csv';support=data/'COMMON_SUPPORT_PRIVATE.json';coord=R/'data/pixbox_s2_cmix_5036991/phase103_20261013/coordinate_metadata.csv'
 assert sha(joined)==read(prior/'CPU_STATISTICS_ACCEPTANCE.json')['private_join_sha256'];assert sha(support)==read(prior/'COMMON_SUPPORT_LOCK.json')['support_sha256'];assert sha(coord)=='c2f029a349096738935d9239585c67ca6dfc5a4c71017f5e8634af982b86ce68'
 valid=[r for r in rows(joined) if r['common_valid']=='True' and r['reference']!='ignore'];assert len(valid)==8567
 strata=collections.defaultdict(lambda:collections.defaultdict(list))
 for r in valid:
  positive=r['reference']=='positive_ground_cloud_shadow';pred=float(r['Omni'])>=.5
  layer='TP' if positive and pred else 'FN' if positive else 'FP' if pred else 'TN';strata[r['group_id']][layer].append(r)
 assert len(strata)==10 and sum(len(g['FN']) for g in strata.values())==75 and sum(len(g['FP']) for g in strata.values())==93
 samples=[];counts=[]
 for gid,g in sorted(strata.items()):
  count={'group_id':gid}
  for layer,cap in [('FN',2),('FP',2),('TP',1),('TN',1)]:
   chosen=sorted(g[layer],key=key)[:cap];count[layer+'_available']=len(g[layer]);count[layer+'_selected']=len(chosen)
   assert chosen==sorted(g[layer],key=key)[:min(cap,len(g[layer]))]
   for r in chosen:samples.append(dict(r,sampling_layer=layer))
  assert sum(count[n+'_selected'] for n in ['FN','FP','TP','TN'])<=6;counts.append(count)
 assert len(samples)<=60 and len({r['point_id'] for r in samples})==len(samples)
 samples.sort(key=lambda r:hashlib.sha256(('PHASE114_BLIND|'+r['point_id']).encode()).hexdigest())
 coords={r['point_id']:r for r in rows(coord)};jobs=[];private=[]
 for i,r in enumerate(samples,1):
  aid=f'A{i:03d}';c=coords[r['point_id']];assert all(str(r[k])==str(c[k]) for k in ['product_id','group_id','row20','col20'])
  private.append(dict(r,anonymous_id=aid))
  jobs.append({'anonymous_id':aid,'product_id':c['product_id'],'native_x_m':float(c['native_x_m']),'native_y_m':float(c['native_y_m']),'row10':int(c['row10']),'col10':int(c['col10']),'row20':int(c['row20']),'col20':int(c['col20'])})
 O.mkdir(parents=True);U.mkdir(parents=True);D.mkdir(parents=True);(O/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n')
 write(D/'SAMPLE_REVEAL_PRIVATE.json',{'samples':private,'do_not_open_before_opinion_seal':True});write(D/'ANONYMOUS_RENDER_INPUTS.json',{'rows':jobs,'no_labels_scores_layers_or_group_ids':True})
 files=[Path(__file__),R/'src/tools/render_phase114.py',R/'src/research_plans/PHASE114_OMNI_RESIDUAL_BLIND_AUDIT_20261014.md',prior/'COMMON_SUPPORT_LOCK.json',prior/'CPU_STATISTICS_ACCEPTANCE.json',R/'src/research_results/phase104_20261013/object_manifest.json']
 write(O/'SAMPLING_BLIND_LOCK.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':{str(p.relative_to(R)).replace('\\','/'):sha(p) for p in files},'private_reveal_sha256':sha(D/'SAMPLE_REVEAL_PRIVATE.json'),'anonymous_inputs_sha256':sha(D/'ANONYMOUS_RENDER_INPUTS.json'),'joined_input_sha256':sha(joined),'coordinate_sha256':sha(coord),'counts':counts,'n_samples':len(samples),'n_groups':10,'sampling_prefix':'PHASE114_AUDIT|','anonymization_prefix':'PHASE114_BLIND|','threshold':.5,'extents_m':[640,10240],'display_TOA_min_max':[0,.3],'display_interpolation':'nearest only; no added detail','extent_alignment':'20m grid aligned, original point cell at central vicinity','review_status':'not_executed','reveal_status':'not_executed','no_sample_replacement':True})
 write(O/'CPU_SAMPLING_ACCEPTANCE.json',{'status':'passed','original_FN':75,'original_FP':93,'all10_groups_used':True,'hash_order_quota_replayed':True,'no_duplicate_points':True,'max_group_samples':max(sum(c[k+'_selected'] for k in ['FN','FP','TP','TN']) for c in counts),'point20m_binding_matches_frozen_coordinates':True,'anonymous_payload_no_label_score_group_or_sampling_layer':all(not({'group_id','sampling_layer','reference','Omni','Source','G7','Source_spectral'}&set(x)) for x in jobs)})
 write(U/'blind_review_template.json',{'reviewer_id':'','reviewer_experience':'','review_mode':'human','records':[{'anonymous_id':j['anonymous_id'],'status':'not_reviewed','point_candidate_class':None,'cell_20m_state':None,'visible_evidence':'','confidence':None,'larger_context_needed':None,'human_arbitration_needed':None} for j in jobs]})
 print('Frozen anonymous samples:',len(samples),'groups:10; reference FN75/FP93 replay passed; no private reveal printed')
if __name__=='__main__':main()

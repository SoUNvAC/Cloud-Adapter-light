"""Summarize only already-sealed, completed opinions; never unblind draft records."""
import json,csv,hashlib,datetime,collections,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'src/research_results/phase114_20261014';U=R/'outputs/phase114/phase114_20261014';D=R/'data/pixbox_s2_cmix_5036991/phase114_20261014'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def csvwrite(p,rows):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 seal=read(O/'OPINION_SHA_SEAL.json');assert sha(D/'SEALED_BLIND_OPINIONS.json')==seal['opinion_sha256'];assert sha(O/'SAMPLING_BLIND_LOCK.json')==seal['sampling_lock_sha256']
 opinion=read(D/'SEALED_BLIND_OPINIONS.json');revealed=read(D/'REVEALED_COMPARISON_PRIVATE.json')['rows'];done=[r for r in opinion['records'] if r['status']=='reviewed']
 assert {r['anonymous_id'] for r in done}=={r['opinion']['anonymous_id'] for r in revealed}
 assert len(revealed)==seal['reviewed'];assert all(r['opinion']==next(x for x in done if x['anonymous_id']==r['opinion']['anonymous_id']) for r in revealed)
 # Preserve the preparation-only reports. Raw opinions/reveal never enter Git.
 snap=O/'material_preparation_snapshot';snap.mkdir(exist_ok=False)
 for name in ['README.md','EXECUTION_STATUS.json','NEXT_HYPOTHESIS.md','group_review_status.csv','review_category_status.csv','DELIVERY_SHA256.json']:shutil.copyfile(O/name,snap/name)
 locked=read(O/'SAMPLING_BLIND_LOCK.json');totals={k:sum(g[k+'_selected'] for g in locked['counts']) for k in ['FN','FP','TP','TN']};bylayer=collections.defaultdict(list);bygroup=collections.defaultdict(list);private=[]
 target={'positive_ground_cloud_shadow':'Shadow','negative_surface':'Surface','negative_cloud':'Cloud'}
 for r in revealed:
  op=r['opinion'];ref=r['reference_and_fixed_decision'];kind=ref['sampling_layer'];assert kind in totals
  concrete=op['point_candidate_class'] in ['Cloud','Shadow','Surface'];mixed=op['point_candidate_class']=='混合' or op['cell_20m_state']=='混合';unknown=op['point_candidate_class']=='无法判断' or op['cell_20m_state']=='无法判断'
  clear=concrete and op['cell_20m_state']=='同质' and op['confidence']=='高';conflict=concrete and op['point_candidate_class']!=target[ref['reference']]
  q=dict(r,categories={'clearly_interpretable_candidate':clear,'mixed_or_scale_question':mixed,'unable_to_determine':unknown,'possible_reference_disagreement':conflict},reviewer_expertise_not_documented=not opinion.get('reviewer_experience'))
  private.append(q);bylayer[kind].append(q);bygroup[ref['group_id']].append(q)
 write(D/'REVIEWED_POINT_EVIDENCE_PRIVATE.json',{'rows':private,'candidate_interpretation_not_independent_adjudication':True,'no_label_changes':True,'no_post_exclusion_performance':True})
 categories=[]
 for k,n in totals.items():
  rr=bylayer[k];categories.append({'stratum':k,'sampled':n,'completed_opinions':len(rr),'pending_draft_or_not_completed':n-len(rr),'clearly_interpretable_candidate':sum(q['categories']['clearly_interpretable_candidate'] for q in rr) if rr else None,'mixed_or_scale_question':sum(q['categories']['mixed_or_scale_question'] for q in rr) if rr else None,'unable_to_determine':sum(q['categories']['unable_to_determine'] for q in rr) if rr else None,'possible_reference_disagreement':sum(q['categories']['possible_reference_disagreement'] for q in rr) if rr else None,'categories_overlap':True})
 csvwrite(O/'review_category_status.csv',categories)
 groups=[]
 for g in locked['counts']:
  gid=g['group_id'];q={'group_id':gid};q.update({k:g[k+'_selected'] for k in totals});q['total']=sum(q[k] for k in totals);q['completed_opinions']=len(bygroup[gid]);q['pending']=q['total']-q['completed_opinions'];q['interpretation_status']='partial' if bygroup[gid] else 'not_executed';groups.append(q)
 csvwrite(O/'group_review_status.csv',groups)
 missing={k:sum(not r.get(k) for r in opinion['records'] if r['status']!='reviewed') for k in ['point_candidate_class','cell_20m_state','visible_evidence','confidence','larger_context_needed','human_arbitration_needed']}
 write(O/'SUBMISSION_ACCEPTANCE.json',{'status':'partial_completed_opinions_sealed','original_file_SHA':seal['opinion_sha256'],'records':len(opinion['records']),'completed':len(done),'draft':sum(r['status']=='draft' for r in opinion['records']),'all_candidate_classes_provided':all(r.get('point_candidate_class') for r in opinion['records']),'uncompleted_missing_fields':missing,'review_mode_declared':opinion.get('review_mode'),'reviewer_id_provided':bool(opinion.get('reviewer_id')),'reviewer_experience_provided':bool(opinion.get('reviewer_experience')),'human_expertise_not_assumed':True,'only_completed_records_revealed':True,'draft_opinions_preserved_without_reference_join':True})
 write(O/'REVIEW_CPU_ACCEPTANCE.json',{'status':'passed','sealed_submission_SHA_unchanged':True,'completed_IDs_equal_revealed_IDs':True,'all47_anonymous_records_preserved':True,'strata_sample_total':sum(totals.values()),'reviewed_plus_pending':sum(q['completed_opinions']+q['pending'] for q in groups),'public_summary_has_no_point_labels_or_scores':True,'no_forward_fit_download_threshold_search_label_edit_or_performance_recompute':True})
 write(O/'EXECUTION_STATUS.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'one_completed_opinion_revealed_46_drafts_preserved_blind','completed':len(done),'draft_or_pending':len(opinion['records'])-len(done),'original_submission_sealed':True,'reveal_scope':'completed records only','new_hypothesis_proposed':False,'cross_group_failure_claim':False,'labels_modified':False,'performance_recomputed':False,'new_forwards':0,'fits':0,'downloads':0,'threshold_search':0,'work_dirs_synced':False})
 print('Completed opinions processed:',len(done),'pending retained:',len(opinion['records'])-len(done),'groups with completed opinion:',len([g for g in groups if g['completed_opinions']]))
if __name__=='__main__':main()

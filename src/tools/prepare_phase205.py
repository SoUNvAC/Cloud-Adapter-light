"""Freeze Phase205 original fit/crop/evaluation identities without new sampling."""
import csv,hashlib,json,shutil,sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase205/phase205_20261019'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'INPUT_LOCK.json').exists();bindings=[]
 for phase,seed in [(122,12201),(123,12301)]:
  d=R/f'outputs/phase{phase}/phase{phase}_202610{15 if phase==122 else 16}/delivery'
  fit=load(d/'FIT_INPUT_LOCK.json');s=load(d/'PAIRED_CROP_SCHEDULE.json');assert len(fit['bindings'])==100 and s['seed']==seed and len(s['rows'])==1000 and s['fit_order']==[x['product'] for x in fit['bindings']]
  bindings.append(fit['bindings'])
  for n in ('FIT_INPUT_LOCK.json','PAIRED_CROP_SCHEDULE.json'):
   target=f'{seed}_{n}';shutil.copyfile(d/n,O/target)
 assert bindings[0]==bindings[1]
 plan=load(R/'outputs/phase203/phase203_20261018/delivery/EVALUATION_INPUT_PLAN.json');rows=plan['rows'];assert len(rows)==48 and sum(r['pool']=='Catalogue' for r in rows)==32
 assert sum(len(r['actual_patch_coordinates']) for r in rows)==1146
 save('EVALUATION_INPUT_PLAN.json',plan)
 import analyze_phase203_local as prior
 import numpy as np
 support=[]
 for r in rows:
  pid=r['product_id']
  if r['pool']=='Catalogue':
   arrays=[prior.cat_old(a,pid) for a in ('R','C122','C123')];indices=arrays[0]['flat_indices'];assert all(np.array_equal(v['flat_indices'],indices) for v in arrays);common=np.logical_and.reduce([a['valid'] for a in arrays]);support.append(dict(pool='Catalogue',product_id=pid,count=int(common.sum()),flat_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),common_valid_sha256=hashlib.sha256(common.tobytes()).hexdigest()))
  else:
   arrays=[prior.pix_old(a,pid) for a in ('R','C122','C123')];maps=[{q['point_id']:q for q in a} for a in arrays];ids=[q['point_id'] for q in arrays[0] if all(d[q['point_id']]['valid'] and d[q['point_id']].get('coverage',True) and d[q['point_id']].get('author_valid',True) for d in maps)];support.append(dict(pool='PixBox',product_id=pid,count=len(ids),common_image_valid_point_ids=ids,semantic_labels_not_read=True))
 save('EXPECTED_SUPPORT.json',dict(rows=support,PixBox_image_valid_total=sum(r['count'] for r in support if r['pool']=='PixBox'),reference_ignored_not_removed=True,only_score_validity_and_locations_read=True))
 lock200=load(R/'outputs/phase200/phase200_20261017/input_model_lock.json');w=lock200['execution_plan']['weights_V4'];assert len(w)==2
 save('V4_WEIGHT_LOCK.json',dict(version=4,library='smp',HF_revision='c9a4fb88188709127aa25cfe51ae7fd41b0132f8',weights=w,phase200_lock_sha256=sha(R/'outputs/phase200/phase200_20261017/input_model_lock.json')))
 meta=load(R/'outputs/phase65/source_metadata/provisional_overlap.json');by={q['product']:q for q in meta['candidates']};coverage=[]
 for r in rows:
  q=by.get(r['product_id']);coverage.append(dict(pool=r['pool'],product_id=r['product_id'],group_id=r['group']['group_id'] if r['pool']=='Catalogue' else r['product']['group_id'] if 'group_id' in r['product'] else 'see fixed coordinate manifest',project_CloudSEN12_source_product_overlap=q['exact_product_overlap'] if q else 'unknown',project_CloudSEN12_source_tile_time_overlap=q['tile_acquisition_overlap'] if q else 'unknown',project_source_tile_any_date=q['source_tile_overlap_any_date'] if q else 'unknown',existing_product_record='outputs/phase65/source_metadata/provisional_overlap.json' if q else 'src/research_results/phase110_20261014/sources/'+r['product_id']+'_B8A_metadata.json',existing_footprint_rules='src/research_results/phase109_20261013/OVERLAP_RULES.json',V4_High_PC_SR_Scribble_2k_KappaSet_full_author_manifest='unknown',OCM_hard_negatives_scribble='not publicly complete; unknown',independent_evaluation=False))
 with (O/'TRAINING_COVERAGE.csv').open('x',encoding='utf-8-sig',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(coverage[0]));writer.writeheader();writer.writerows(coverage)
 save('TRAINING_COVERAGE_SOURCES.json',dict(changelog='https://github.com/DPIRD-DMA/OmniCloudMask/blob/main/MODEL_CHANGELOG.md',method='https://omnicloudmask.readthedocs.io/en/latest/how-it-works.html',checked_date_Beijing='2026-10-19',V4_training_sources=['CloudSEN12 High','High Planetary Computer','High SR','Scribble','2k','KappaSet','OCM hard negatives','OCM Scribble'],limits='Project source records are not the complete author V4 train/validation manifest. Unknown product or footprint overlap remains unknown. OCM additions not publicly complete.',project_source_audit_sha256=sha(R/'outputs/phase65/source_metadata/provisional_overlap.json'),geometry_rule_sha256=sha(R/'src/research_results/phase109_20261013/OVERLAP_RULES.json')))
 inputs={p.name:sha(p) for p in O.iterdir() if p.is_file()};sources={p.relative_to(R).as_posix():sha(p) for p in [R/'src/tools/phase119_common.py',R/'src/tools/phase118_common.py',R/'src/tools/phase120_common.py',R/'src/tools/phase200_common.py',R/'src/tools/install_phase111_runtime.py']}
 save('INPUT_LOCK.json',dict(protocol_sha256=sha(R/'src/research_plans/PHASE205_V4_ADAPTATION_DAMAGE_CHECK_20261019.md'),files=inputs,source_sha256=sources,planned_training_calls=2000,planned_evaluation_calls=4584,planned_gate_and_constructor_allowance=160,planned_total_allowance=6744,approved_total_cap=8000,GPU_stage_seconds_cap=14400,CPU_bytes_cap=16*1024**3,GPU_bytes_cap=20*1024**3,new_persistent_bytes_cap=6000000000,host_free_bytes_floor=4000000000,no_calibration_inputs=True,cutoff_Beijing='2026-10-20T00:00:00+08:00'))
 print('local input/crop identities locked; estimated6744 calls within8000 cap')
if __name__=='__main__':main()

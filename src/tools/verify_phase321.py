"""Remote acceptance of frozen baselines, geometry thresholds and unchanged inputs."""
from audit_phase321 import R,O,sha
import json,csv,sys,math
from pathlib import Path
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):return list(csv.DictReader(Path(p).read_text(encoding='utf-8-sig').splitlines()))
def main():
 assert sys.platform=='linux'
 p=O/'recovery01';v=js(p/'RESULT.json');assert v['status']=='completed'
 pin=js(p/'INPUT_FINAL_SHA.json');assert pin['all_unchanged'] and all(sha(n)==x['sha256'] for n,x in pin['files'].items())
 lock=js(p/'INPUT_LOCK.json');assert all(sha(R/'src/tools'/n)==h for n,h in lock['code'].items())
 original=next(r for r in rows(R/'src/research_results/phase317_20261026/METRICS.csv') if r['pool']=='CMC' and r['strategy']=='Source')
 for r in v['sensitivity']:
  if r['policy']!='full':continue
  if r['dataset']=='Catalogue':
   assert r['retained_groups']==32
   for current,old in [('macro_AP_pct','macro_Shadow_AP_pct'),('macro_mIoU_pct','group_mean_mIoU_pct'),('pooled_mIoU_pct','pooled_mIoU_pct'),('macro_FPR_pct','macro_Shadow_FPR_pct')]:assert abs(r[current]-float(original[old]))<=1e-6,(current,r[current],original[old])
   assert r['confusion']==json.loads(original['confusion'])
  else:
   b=next(x for x in rows(R/'src/research_results/phase112_20261014/four_score_summary.csv') if x['score']=='Source')
   assert r['retained_groups']==10 and abs(r['macro_AP_pct']-float(b['macro_sampled_AP_pp']))<=1e-6
 for r in rows(p/'MATCH_PAIRS.csv'):
  area=float(r['intersection_m2']) if r['intersection_m2'] else None
  assert (r['spatial_overlap']=='True')==(area is not None and area>1.0)
  for k in ['source_fraction','target_fraction']:
   if r[k]:assert math.isfinite(float(r[k])) and -1e-8<=float(r[k])<=1.000001
 assert v['training']==v['forwards']==v['model_loads']==v['pixel_reads']==v['downloads']==0
 target=rows(p/'TARGET_OVERLAP.csv');combined=[]
 for ds in ['Catalogue','ALCD','PixBox']:
  rr=[r for r in target if r['dataset']==ds];keys={r['key'] for r in rr}
  union={flag:len({r['key'] for r in rr if r[flag]=='True'}) for flag in ['same_scene_candidate','same_tile_different_day','spatial_overlap','unknown']}
  excluded={r['key'] for r in rr if r['strict_clean']=='False'}
  evaluated={r['key'] for r in rr if r['evaluated']=='True'}
  combined.append({'dataset':ds,'source_roles':'train+val','products':len(keys),**union,'strict_clean':len(keys-excluded),'evaluated_products':len(evaluated),'evaluated_excluded':len(evaluated&excluded)})
 import rasterio,shapely
 out={'status':'passed','baseline_full_source_reproduced':True,'metric_tolerance_pp':1e-6,'input_and_execution_code_SHA':True,'area_and_fraction_acceptance':True,'combined_counts':combined,'versions':{'rasterio':rasterio.__version__,'GDAL':rasterio.__gdal_version__,'shapely':shapely.__version__},'V4_overlap_certified':False}
 (O/'ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()

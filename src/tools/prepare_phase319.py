"""Pre-register every scientific rule before any fit geometry quality access."""
from phase319_common import *
from datetime import datetime,timezone,timedelta
import subprocess
def main():
 assert not ROOT.exists();assert datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')=='20261027'
 guard();ROOT.mkdir();(ROOT/'code').mkdir()
 plan=R/'src/research_plans/phase319.md';code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase319*.py')}
 for p in (R/'src/tools').glob('*phase319*.py'):shutil.copyfile(p,ROOT/'code'/p.name)
 frozen={}
 for pool in POOLS:
  rows=inputs(pool);assert len(rows)=={'SPARCS':60,'L8':100}[pool]
  assert len({r['scene'] for r in rows})=={'SPARCS':60,'L8':18}[pool]
  if pool=='L8':assert all(r['usgs_shadows']=='yes' for r in rows)
  frozen[pool]=dict(manifest=pin(PRIOR/pool/'IMAGES.json','original fixed manifest'),pseudo_lock=pin(PRIOR/pool/'PSEUDO_LOCK.json','frozen teacher lock'),fit=rows,expected_A_Sh=EXPECTED[pool])
 dependencies=[pin(R/'src/tools/phase317_common.py','inherited byte-only helpers'),pin(PRIOR/'REFERENCE_LOCK.json','original converter axis identity'),pin(R/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth','unchanged Source bytes; never loaded')]
 assert dependencies[-1]['sha256']==SOURCE
 window=PRIOR/'DECISION_WINDOW.json';shutil.copyfile(window,ROOT/'DECISION_WINDOW.json')
 prereg=dict(status='preregistered',epoch=time.time(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),plan_sha256=sha(plan),code=code,inputs=frozen,Source_sha256=SOURCE,decision_window_sha256=sha(window),historical_exposure='Phase317/318 fit quality previously exposed; fixed-rule follow-up exploration, not blind',
  geometry=dict(height_m=[200,12000],angle_halfwidth_degrees=5,search='Shadow toward Sun; reverse placebo toward Sun+180',radial_tolerance='one output pixel half diagonal',grid='actual CRS/affine composed with original-to512 resize; geographic forward/inverse checks at center and corners; record maximum direction approximation error; reject if greater than0.1 degree or relative scale deviation greater than0.1 percent',cloud='imagevalid & sum(scores)>1e-12 & argmax(scores)==1; no confidence cutoff',shadow='imagevalid & sum(scores)>1e-12 & argmax(scores)==2 & q_Shadow>=0.9',conf='per qualified image same k as G_sun; descending q_Shadow then original row-major index; no reference filter',unknown='kernel outside valid field is unknown; full theoretical kernel, not just clipped offsets',selection_barrier='all masks and kernels sealed before independent quality process; coordinate mask identity only allowed earlier'),
  criteria=dict(qualified_A_Sh_fraction_min=.9,reference_coverage_min=.95,pooled_precision_min=.9,gain_over_original_pp_min=5,full_A_Sh_retention_min=.3,scene_min=5,scene_valid_selected_min=100,scene_median_gain_strictly_positive=True,scene_majority_gain_strictly_positive=True,gain_over_conf_pp_min=2,gain_over_reverse_pp_min=2,both_pools_must_pass=True,reverse_zero_support='NA, insufficient direction evidence',missing_input='not_testable',tested_failure='no_go',success='go_to_method_design'),
  reference=dict(L8='exact original ordinal raw-toPNG identity, rawfill0 ignored; no shift/rotation search',SPARCS='native7 classes parent [2,2,0,0,0,1,0], nearest512',comparison='same qualified products/common reference definitions; original full A_Sh denominator for retention',strata='native labels, biome only descriptive',statistics='scene aggregation, no pixel bootstrap'),
  limits=dict(neural_forwards=0,training=0,fitting=0,downloads=0,author_contact=0,threads=4,memory_bytes=16*1024**3,wall_seconds=10800,new_two_host_bytes=2000000000,host_free_bytes=4*1024**3),resources=guard())
 prereg['dependencies']=dependencies
 write(ROOT/'PREREGISTRATION.json',prereg);write(ROOT/'STATUS.json',dict(status='preregistered',epoch=time.time()))
 print('Phase319 preregistered; zero target quality reads')
if __name__=='__main__':main()

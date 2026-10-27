"""Lock exact existing members, prior exposed results and both analysis definitions."""
from phase320_common import *
from datetime import datetime,timezone,timedelta
import subprocess
def main():
 assert not ROOT.exists();assert datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')=='20261027';guard();assert shutil.disk_usage(ROOT.parent).free>=8*1024**3
 ROOT.mkdir();(ROOT/'tmp').mkdir();(ROOT/'code').mkdir();write(ROOT/'STARTED.json',dict(epoch=time.time(),pid=os.getpid(),date='20261027',normal_CPU_wall_cap_seconds=7200,repair_total_cap_seconds=10800))
 pins={};gaps=[]
 def bind(p,expected=None,kind='input'):
  p=bound(p);key=str(p)
  if key in pins:
   e=pins[key];assert expected is None or expected==e['sha256'];return e
  if not p.is_file():gaps.append(dict(path=key,kind=kind,reason='missing; no new inference'));return None
  h=sha(p)
  if expected is not None and h!=expected:gaps.append(dict(path=key,kind=kind,reason='SHA identity mismatch',actual=h,expected=expected));return None
  e=dict(path=key,sha256=h,bytes=p.stat().st_size,kind=kind);pins[key]=e;return e
 source=bind(R/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth',SOURCE,'Source identity only; never load')
 for n in ['phase317_common.py','worker_phase317.py','reference_phase317.py','analyze_phase317.py']:bind(R/'src/tools'/n,kind='historical definitions; source text only')
 m=bind(P317/'METRICS.csv');pm=bind(P317/'PER_SCENE_METRICS.csv');refs_entry=bind(P317/'REFERENCE_LOCK.json');old319=[bind(P319/n) for n in ['NATIVE_STRATA.csv','PRECISION_COVERAGE.csv','SELECTION_LOCK.json','QUALITY_LOCK.json','GEOMETRY_INPUTS.json','COORDINATE_ACCEPTANCE.json']]
 metrics=read(P317/'METRICS.csv') if m else [];refs=load(P317/'REFERENCE_LOCK.json')['rows'] if refs_entry else []
 threshold={}
 for pool in POOLS:
  rr=[r for r in refs if r['pool']==pool];ready=bool(m and pm and refs_entry and source);entries=[];locks={}
  for arm in ARMS:
   lp=OLD/'EVALUATION_LOCK.json' if pool=='CMC' and arm=='PL' else P317/pool/arm/'EVALUATION_LOCK.json';ep=bind(lp,kind='original evaluation lock');ready&=ep is not None
   locks[arm]=load(lp)['rows'] if ep else []
  for r in rr:
   ref=bind(r['file'],r['sha256'],'original317 validation reference');caches={};ready&=ref is not None
   for arm in ARMS:
    if pool=='CMC' and arm=='PL':
     item=next((v for v in locks[arm] if v['product']==r['name'] and v['arm']=='LoRA'),None);path=OLD/'evaluation'/('LoRA_'+r['name']+'.npz')
    else:item=next((v for v in locks[arm] if v['name']==r['name']),None);path=Path(item['file']) if item else None
    cache=bind(path,item['sha256'],'softmax(s)[2] cache, not fit q') if item else None;ready&=cache is not None;caches[arm]=cache
   entries.append(dict(name=r['name'],scene=r['scene'],reference=ref,cache=caches,expected_support=r['support']))
  assert len(rr)=={'CMC':32,'SPARCS':10,'L8':963}[pool] and len({r['scene'] for r in rr})=={'CMC':32,'SPARCS':10,'L8':8}[pool]
  cb=next((r for r in metrics if r['pool']==pool and r['strategy']=='CB'),None)
  threshold[pool]=dict(status='ready' if ready else 'not_testable',rows=entries,CB_original_macro_recall_pct=float(cb['macro_Shadow_recall_pct']) if cb else None,CB_original_macro_FPR_pct=float(cb['macro_Shadow_FPR_pct']) if cb else None)
 fit={};selection=load(P319/'SELECTION_LOCK.json')['rows'] if all(old319) else [];quality=load(P319/'QUALITY_LOCK.json')['references'] if all(old319) else []
 for pool in ['SPARCS','L8']:
  image=bind(P317/pool/'IMAGES.json');teacher=bind(P317/pool/'PSEUDO_LOCK.json');ready=all(e is not None for e in old319) and image is not None and teacher is not None and source is not None;rows=[]
  manifest=load(P317/pool/'IMAGES.json')['fit'] if image else [];teacherrows=load(P317/pool/'PSEUDO_LOCK.json')['rows'] if teacher else []
  for r in manifest:
   if pool=='L8':assert r['usgs_shadows']=='yes'
   tr=next(t for t in teacherrows if t['name']==r['name']);se=next((v for v in selection if v['pool']==pool and v['name']==r['name']),None);qe=next((v for v in quality if v['pool']==pool and v['name']==r['name']),None)
   te=bind(tr['file'],tr['sha256'],'original317 fit full scores/pred/pseudo/valid');ref=bind(qe['path'],qe['sha256'],'original319 fit native reference') if qe else None;geom=bind(se['file'],se['sha256'],'original319 geometry masks, never recompute') if se else None;ready&=te is not None and ref is not None
   rows.append(dict(name=r['name'],scene=r['scene'],biome=r.get('biome','NA'),teacher=te,reference=ref,geometry=geom,geometry_qualified=se['status']=='qualified' if se else None))
  assert len(rows)=={'SPARCS':60,'L8':100}[pool];fit[pool]=dict(status='ready' if ready else 'not_testable',rows=rows)
 code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase320*.py')}
 for p in (R/'src/tools').glob('*phase320*.py'):shutil.copyfile(p,ROOT/'code'/p.name)
 shutil.copyfile(P317/'DECISION_WINDOW.json',ROOT/'DECISION_WINDOW.json')
 definitions=dict(exposure='post-hoc diagnostic on already exposed317-319 development data; not blind',fit_shadow='imagevalid & sum(s)>1e-12 & argmax(s)==2 & qShadow>=0.90; q=s/sum(s)',fit_expected_A_Sh=EXPECTED,fit_native={'SPARCS':{'0':'shadow','1':'shadow_over_water','2':'water','3':'snow','4':'land','5':'cloud','6':'flooded'},'L8':{'0':'invalid_fill','64':'shadow','128':'clear','192':'thin_cloud','255':'cloud'}},unknown_semantics='L8 clear and SPARCS land do not identify water/dark surface/terrain shadow or annotation errors; biome is not a pixel class',boundary='both valid 4-neighbor binary disagreements marked on BOTH sides; Chebyshev2 dilation; within2 of outermost image pixel or any invalid reference is context unknown; cloud/noncloud and shadow/nonshadow',geometry='reuse319 status/masks; unqualified products separate from unmatched_unknown; no projection repair',threshold='predict_Shadow = cached softmax(s)[2] >= global t; exact distinct finite float32 blocks, all/none; no tie splitting',selection_FPR='max macro recall under CB original macro FPR; ties smaller FPR then higher threshold',selection_recall='min macro FPR reaching CB original macro recall; ties larger recall then higher threshold',neighbors='adjacent actual staircase points straddling target METRIC, lower_neighbor lower metric side, upper_neighbor upper metric side; not interpolation',macro='recall includes scenes with positives; FPR includes scenes with negatives; group denominators fixed; AP excludes pure-positive and pure-negative scenes per317',AP='exact noninterpolated grouped-tie AP, sum(delta positives * cumulative precision)/positives, matching317; no binning',mIoU='threshold three-class mIoU NA; original argmax baseline separate',precision='no predictions NA; recall0 if positives exist; unknown class cause remains unknown',scene_fields='known_Surface_errors=all reference Surface errors; unresolved_Surface_errors=SPARCS land or L8 clear subset; water subset; overlapping descriptors MUST NOT sum',original_tolerance_pp=1e-6,integer_confusions='exact',oracle_diagnostic=True)
 write(ROOT/'ANALYSIS_LOCK.json',dict(status='locked',epoch=time.time(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),plan_sha256=sha(R/'src/research_plans/phase320.md'),code=code,inputs=pins,fit=fit,threshold=threshold,known_results=dict(phase317='CB recall gain accompanied by FPR cost in all pools; original metrics locked',phase318='Flip did not establish consistent two-pool unique reliability gain; unchanged',phase319='SPARCS not_testable due local-grid admitted coverage66.31%, L8 no_go; original geometry unchanged'),definitions=definitions,window_sha256=sha(ROOT/'DECISION_WINDOW.json'),limits=dict(neural_calls=0,trainings=0,fits=0,downloads=0,CPU_seconds=7200,total_with_repairs_seconds=10800,memory_bytes=16*1024**3,threads=4,new_two_host_persistent_bytes=1024**3,scratch_bytes=6*1024**3,start_free_disk_bytes=8*1024**3),missing=gaps,resources=guard()))
 write(ROOT/'INPUT_SHA256.json',pins);write(ROOT/'STATUS.json',dict(status='locked',epoch=time.time()));print('Phase320 locked existing input identities and post-hoc rules',flush=True)
if __name__=='__main__':main()

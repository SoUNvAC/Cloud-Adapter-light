"""Synthetic end-to-end CPU check, no experimental data or model calls."""
from pathlib import Path
import sys,types,json,hashlib,shutil,os
try:import resource
except ImportError:sys.modules['resource']=types.ModuleType('resource')
import numpy as np
import phase241_common as common,phase241_cpu_common as core,phase241_ranking as rank,calibrate_phase241 as cal,analyze_phase241 as analysis
R=Path(os.environ.get('PHASE241_TEST_ROOT',str(Path(__file__).resolve().parents[2]/'outputs/phase241/phase241_20261030/CPU_SYNTHETIC_TEST')))
def main():
 assert not R.exists();(R/'code').mkdir(parents=True);name='rank_merge.exe' if os.name=='nt' else 'rank_merge';exe=R.parent/'code'/name;shutil.copy2(exe,R/'code'/name)
 for m in (common,core,rank,cal,analysis):m.R=R
 class CPU:
  def __init__(self,*args):pass
  def check(self):pass
  def finish(self,*args):pass
 cal.CPU=analysis.CPU=CPU
 rng=np.random.default_rng(76);source={}
 for split in ('val','test'):
  index=[]
  for domain in ('L8','SPARCS'):
   for scene_id in range(3):
    scene=f'{domain}_{split}_{scene_id}';y=rng.integers(0,3,(8,8),dtype=np.uint8)
    if domain=='SPARCS' and scene_id==2:y[:]=1
    y[0,0]=255;lp=R/f'{scene}_label.npy';np.save(lp,y);source[scene]=y
    for model in ('Source','C21701','C21702'):
     a=rng.uniform(.01,1,(3,8,8)).astype(np.float32);a/=a.sum(0);fp=R/f'{scene}_{model}.npy';np.save(fp,a);valid=y!=255;pr=a.argmax(0);cm=np.bincount(3*y[valid].astype(np.int64)+pr[valid],minlength=9).reshape(3,3)
     index.append({'domain':domain,'split':split,'model':model,'scene':scene,'name':scene+'.png','usgs_shadows':'yes' if scene_id<2 else 'no','label_file':lp.name,'label_SHA':common.sha(lp),'score_file':fp.name,'score_SHA':common.sha(fp),'confusion':cm.tolist()})
  (R/(split.upper()+'_SCORE_INDEX.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in index))
 common.save('VAL_COMPLETE.json',{'passed':True});cal.main();common.save('TEST_STARTED.json',{'epoch':__import__('time').time()});common.save('TEST_COMPLETE.json',{'passed':True});analysis.main()
 m=common.js(R/'METRICS.json');assert not m['missing'];assert len(common.js(R/'CALIBRATION_LOCK.json')['fits'])==6;assert len(common.js(R/'THRESHOLD_LOCK.json')['thresholds'])==12
 for v in m['pooled']:
  if v['split']=='val' and v['decision']=='frozen_threshold':assert v['scene_mean_nonShadow_to_Shadow_FPR']<=.01
 assert any(v['AUC'] is None for v in m['ranking'] if v['scope']=='scene' and v['domain']=='SPARCS')
 b=common.js(R/'BOOTSTRAP.json');assert len(b)==210
 common.save('TEST_ACCEPTED.json',{'passed':True,'six_bias_fits':True,'twelve_exact_thresholds':True,'full_ranking_and_decision_pipeline':True,'NA_preserved':True,'bootstrap_rows':len(b),'real_model_calls':0});print((R/'TEST_ACCEPTED.json').read_text())
if __name__=='__main__':main()

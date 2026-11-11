"""End-to-end synthetic val/lock/test/independent-validation before real cache access."""
from pathlib import Path
import json,os,shutil,sys,time
import numpy as np
import phase242_common as common,analyze_phase242 as analysis,validate_phase242 as validator

def main():
 actual=common.R;root=actual/os.environ.get('PHASE242_SMOKE_FOLDER','CPU_SYNTHETIC_TEST');prior=root/'prior';assert not root.exists();(root/'code').mkdir(parents=True);name='rank242.exe' if os.name=='nt' else 'rank242';shutil.copy2(actual/'code'/name,root/'code'/name)
 for module in (common,analysis,validator):module.R=root;module.P=prior
 class CPU:
  def __init__(self,*args):pass
  def check(self):pass
  def finish(self,*args):pass
  def remaining(self):return 120
 analysis.CPU=validator.CPU=CPU
 rng=np.random.default_rng(24201);indexes={};old={'per_scene':[],'ranking':[]};thresholds={d:{m:.65 for m in ('Source','C21701','C21702')} for d in ('L8','SPARCS')}
 def brute(pos,neg):
  if not len(pos) or not len(neg):return None,None
  s=np.r_[pos,neg];truth=np.r_[np.ones(len(pos)),np.zeros(len(neg))];ap=sum(np.sum((s==v)&(truth==1))*truth[s>=v].mean() for v in np.unique(s))/len(pos);auc=sum(float(a>b)+.5*float(a==b) for a in pos for b in neg)/(len(pos)*len(neg));return float(ap),float(auc)
 for split in ('val','test'):
  indexes[split]=[]
  for domain in ('L8','SPARCS'):
   ys={};scores={m:{} for m in ('Source','C21701','C21702')}
   for i in range(3):
    scene=f'{domain}_{split}_{i}';y=np.resize(np.array([0,1,2],np.uint8),16).reshape(4,4);y[0,0]=255
    if domain=='SPARCS' and split=='val' and i==2:y[:]=1
    ys[scene]=y;lp=prior/'labels'/(scene+'.npy');lp.parent.mkdir(exist_ok=True,parents=True);np.save(lp,y)
    for model in scores:
     p=rng.uniform(.01,1,(3,4,4)).astype(np.float32);p/=p.sum(0);fp=prior/'scores'/model/(scene+'.npy');fp.parent.mkdir(parents=True,exist_ok=True);np.save(fp,p);s=common.logscore(p);valid=y!=255;scores[model][scene]=(s[valid],y[valid]);indexes[split].append({'domain':domain,'model':model,'scene':scene,'split':split,'name':scene,'score_file':str(fp.relative_to(prior)),'label_file':str(lp.relative_to(prior)),'score_SHA':common.sha(fp),'label_SHA':common.sha(lp)})
     decisions=analysis.decision(p[:,valid],y[valid],0,thresholds[domain][model],True)
     for dec,v in decisions.items():old['per_scene'].append({'domain':domain,'split':split,'model':model,'state':'raw','scene':scene,'decision':dec,**v})
   for model,scenes in scores.items():
    raw=[]
    for s,y in scenes.values():raw.append(brute(s[y==2],s[y==0]))
    ss=np.concatenate([v[0] for v in scenes.values()]);yy=np.concatenate([v[1] for v in scenes.values()]);ap,auc=brute(ss[yy==2],ss[yy==0]);old['ranking'].append({'domain':domain,'split':split,'model':model,'state':'raw','support':'main','task':'Shadow_Surface','scope':'pooled','AP':ap,'AUC':auc});old['ranking'].append({'domain':domain,'split':split,'model':model,'state':'raw','support':'main','task':'Shadow_Surface','scope':'scene_mean','AP':analysis.avg([v[0] for v in raw]),'AUC':analysis.avg([v[1] for v in raw])})
 common.save('INDEX_REFERENCES.json',indexes);common.save('PRIOR_METRICS.json',old);common.save('RAW_THRESHOLDS.json',thresholds);common.save('MODEL_IDENTITIES.json',{'synthetic':True});common.save('LOCK.json',{'network_calls':0,'gradient_updates':0});common.save('KERNEL_ACCEPTANCE.json',{'passed':True,'synthetic':'full pipeline independent of experimental pixels'})
 sys.argv=['analyze','val'];analysis.main();sys.argv=['analyze','test'];analysis.main();validator.main();assert common.js(root/'INDEPENDENT_ACCEPTANCE.json')['passed']
 common.R=actual;common.save('CPU_SMOKE_ACCEPTED.json',{'passed':True,'checks':common.js(root/'INDEPENDENT_ACCEPTANCE.json')['checks'],'full_val_lock_test_pipeline':True,'network_calls':0});print(json.dumps(common.js(actual/'CPU_SMOKE_ACCEPTED.json')))
if __name__=='__main__':main()

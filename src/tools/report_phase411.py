"""CPU summaries from all scheduled target evaluations; no model calls."""
from pathlib import Path
import sys,json,hashlib,csv
import numpy as np
R=Path('/home/scv/shared/phase411_20261101')
sys.path.insert(0,'/home/scv/shared/phase400_20261031/code')
from report_phase400 import describe,ap_exact,writecsv
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ref=Path('/home/scv/shared/phase410_20261101')
 def read(name):
  with (ref/name).open() as f:return [x for x in csv.DictReader(f) if x['mode']=='baseline']
 summaries=read('TARGET_SUMMARY.csv');groups=read('TARGET_GROUPS.csv')
 assert len(summaries)==8 and len(groups)==256
 for mode in ['layerwise']:
  sr=R/mode
  for step in range(500,4001,500):
   report=js(sr/'evaluations/M'/f'step_{step}/METRICS.json');cm=np.zeros((3,3),dtype=np.int64);aps=[];scores=[];labels=[]
   assert len(report['per_group'])==32
   for row in report['per_group']:
    file=sr/row['score_file'];assert sha(file)==row['score_SHA']
    with np.load(file) as z:
     valid=z['truth']!=255;s=z['shadow'][valid].astype(np.float64);y=z['truth'][valid]==2
    ap=ap_exact(s,y);aps.append(ap);scores.append(s);labels.append(y);cm+=np.asarray(row['confusion'],dtype=np.int64)
    groups.append({'mode':mode,'step':step,'name':row['product'],'group_id':row['group_id'],**describe(row['confusion']),'AP':None if ap is None else ap*100,'prediction_SHA':row['prediction_SHA']})
   assert cm.tolist()==report['confusion']
   d=describe(cm);assert abs(d['mIoU']-report['mIoU'])<1e-8
   assert sum(a is not None for a in aps)==7
   summaries.append({'mode':mode,'step':step,**d,'macro_AP':np.mean([a for a in aps if a is not None])*100,'pooled_AP':ap_exact(np.concatenate(scores),np.concatenate(labels))*100})
 writecsv(R/'TARGET_SUMMARY.csv',summaries);writecsv(R/'TARGET_GROUPS.csv',groups)
if __name__=='__main__':main()

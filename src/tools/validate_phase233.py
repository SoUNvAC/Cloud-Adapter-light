"""Independent scalar counts, ratios, scene deltas and same-draw bootstrap."""
from analyze_phase233 import R,D,OLD,NAMES,js,save,metrics,START
from analyze_phase230 import cpu_peak
import math,time
import numpy as np
def scalar(c):
 c=[[int(x) for x in row] for row in c];ds=[c[i][i] for i in range(3)];u=[sum(c[i])+sum(c[j][i] for j in range(3))-ds[i] for i in range(3)];ious=[100*ds[i]/u[i] if u[i] else None for i in range(3)]
 ratio=lambda n,d:100*n/d if d else None
 return {'mIoU':math.fsum(v for v in ious if v is not None)/sum(v is not None for v in ious),'class_IoU':ious,'valid_classes':sum(x>0 for x in u),'shadow_recall':ratio(ds[2],sum(c[2])),'nonshadow_FPR':ratio(c[0][2]+c[1][2],sum(c[0])+sum(c[1])),'valid_pixels':sum(map(sum,c))}
def eq(a,b):assert a is None and b is None or a is not None and b is not None and abs(a-b)<1e-10,(a,b)
def main():
 assert js(R/'CPU_ANALYSIS_COMPLETE.json')['status']=='passed';t1=js(R/'TABLE1.json');t2=js(R/'TABLE2.json');per=js(R/'PER_SCENE.json');data={};count=0
 for model in NAMES:
  source=js(D/f'evaluation/{model}.json');native={v['name']:v for v in js(R/'L8_NATIVE_REFERENCES.json')};old={v['name']:v for v in js(OLD/f'delivery/evaluation/L8_{model}.json')['per_input']}
  for mode in ['OLD','VALID']:
   grouped={}
   for v in source['per_input']:
    assert v['prediction_SHA']==old[v['name']]['prediction_SHA'] and v['OLD']==old[v['name']]['confusion'];assert v['reference_SHA']==native[v['name']]['sha256'];assert sum(v['Fill_prediction_histogram'])==native[v['name']]['invalid'];g=v['scene'];grouped.setdefault(g,[[0]*3 for i in range(3)])
    for i in range(3):
     for j in range(3):
      d=v['OLD'][i][j]-v['VALID'][i][j];assert d==(v['Fill_prediction_histogram'][j] if i==0 else 0);grouped[g][i][j]+=v[mode][i][j]
    count+=1
   data[model,mode]=grouped;total=[[sum(c[i][j] for c in grouped.values()) for j in range(3)] for i in range(3)];row=next(v for v in t1 if v['model']==model and v['support']==mode);m=scalar(total)
   for k in ['mIoU','shadow_recall','nonshadow_FPR','valid_pixels','valid_classes']:eq(m[k],row[k])
   for a,b in zip(m['class_IoU'],row['class_IoU']):eq(a,b)
   for g,c in grouped.items():
    stored=next(v for v in per if v['model']==model and v['support']==mode and v['scene']==g);assert c==stored['confusion']
    for k in ['mIoU','shadow_recall','nonshadow_FPR']:eq(scalar(c)[k],stored[k])
   for k in ['mIoU','shadow_recall','nonshadow_FPR']:
    vals=[scalar(c)[k] for c in grouped.values() if scalar(c)[k] is not None];eq(math.fsum(vals)/len(vals) if vals else None,row['scene_mean_'+k]);assert len(vals)==row[k+'_defined_scenes']
 gs=sorted(data['Source','OLD']);idx=np.load(R/'BOOTSTRAP_FIXED_INDICES.npz')['scene_indices'];assert np.array_equal(idx,np.random.default_rng(22901).integers(0,8,(10000,8)));bootchecks=0
 for label,ad,bd in [('SPARCS_minus_L8','SPARCS','L8'),('SPARCS_minus_Source','SPARCS','Source'),('L8_minus_Source','L8','Source')]:
  dist=[]
  for draw in idx:
   diffs=[]
   for seed in [21701,21702]:
    am=f'{ad}_{seed}';bm='Source' if bd=='Source' else f'{bd}_{seed}'
    def pool(m):return [[sum(data[m,'VALID'][gs[k]][i][j] for k in draw) for j in range(3)] for i in range(3)]
    diffs.append(scalar(pool(am))['mIoU']-scalar(pool(bm))['mIoU'])
   dist.append(math.fsum(diffs)/2);bootchecks+=1
  mean=next(v for v in t2 if v['comparison']==label and v['seed']=='two_seed_metric_difference_mean');ci=np.quantile(dist,[.025,.975]);assert np.max(np.abs(ci-mean['VALID_paired_scene_bootstrap_95_descriptive_pp']))<1e-10
  for seed in [21701,21702]:
   row=next(v for v in t2 if v['comparison']==label and v['seed']==seed);am=f'{ad}_{seed}';bm='Source' if bd=='Source' else f'{bd}_{seed}'
   for mode in ['OLD','VALID']:
    a=np.array(list(data[am,mode].values())).sum(0);b=np.array(list(data[bm,mode].values())).sum(0);eq(scalar(a)['mIoU']-scalar(b)['mIoU'],row[mode+'_delta_mIoU_pp'])
   eq(row['VALID_delta_mIoU_pp']-row['OLD_delta_mIoU_pp'],row['delta_change_VALID_minus_OLD_pp'])
 assert count==9630 and bootchecks==30000 and cpu_peak()<16*1024**3;save('INDEPENDENT_ACCEPTANCE.json',{'status':'passed','per_input_support_rows':count,'scene_rows':80,'summary_rows':10,'bootstrap_scalar_rechecks':bootchecks,'bootstrap_new_random_draws':False,'same_index_file_replayed':True,'only_reference_Surface_changes':True,'all4815_OLD_SHA_confusions_exact':True,'CPU_seconds':time.monotonic()-START,'CPU_peak_bytes':cpu_peak()});print('independent CPU acceptance passed')
if __name__=='__main__':main()

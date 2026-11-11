"""Zero-call L8 identity, spatial split and reference support audit before any fit."""
from pathlib import Path
import csv,json,hashlib,collections,time,os,sys
R=Path('/home/scv/shared/phase222_20261024');S=Path('/home/scv/Cloud-Adapter-light/src')
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import numpy as np
 from PIL import Image
 assert not (R/'INPUT_PREFLIGHT_ACCEPTED.json').exists();os.chdir(S)
 for n,h in js(R/'CODE_LOCK.json').items():assert sha(R/'code'/n)==h,n
 manifest=S/'work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv';assert sha(manifest)=='885c2d7f61cae23409a7b1ccbe65739c302fc1e59b089aab0cc05badf674872b'
 statuspath=S/'research_plans/protocol_data/l8_biome_usgs_shadow_status.csv';status={v['scene']:v['usgs_shadows'].lower() for v in csv.DictReader(statuspath.open())};assert len(status)==96 and sum(x=='yes' for x in status.values())==32
 allrows=list(csv.DictReader(manifest.open()));pools={};rawscenes={}
 old=js(R/'OLD212_INPUT_SUPPORT.json')['pools'];test=js(R/'EXPECTED_Source__test.json');oldtest={v['name']:v for v in test['per_input']}
 for pool,split,count in [('train','target_train',1560),('val','target_val',963),('test','target_test',2167)]:
  raw=[v for v in allrows if v['new_split']==split];rawscenes[pool]=set(v['scene'] for v in raw)
  rows=sorted([v for v in raw if pool=='test' or status[v['scene']]=='yes'],key=lambda v:(v['scene'],v['name']));assert len(rows)==count
  for v in rows:
   for k in ['image_path','mask_path']:
    p=Path(v[k]);p=p if p.is_absolute() else S/p;v[k]=str(p);assert p.is_file();v[k.replace('_path','_sha256')]=sha(p)
   v['usgs_shadows']=status[v['scene']]
  pools[pool]=rows
  if pool=='val':assert {v['name']:(v['image_sha256'],v['mask_sha256']) for v in rows}=={v['name']:(v['image_sha256'],v['mask_sha256']) for v in old['l8']}
  if pool=='test':assert {v['name']:(v['image_sha256'],v['mask_sha256']) for v in rows}=={n:(v['image_sha256'],v['mask_sha256']) for n,v in oldtest.items()}
 assert all(not rawscenes[a]&rawscenes[b] for a,b in [('train','val'),('train','test'),('val','test')])
 assert len({v['scene'] for v in pools['val']})==8 and len({v['scene'] for v in pools['test']})==16
 strat={k:sorted({v['scene'] for v in pools['test'] if v['usgs_shadows']==k}) for k in ['yes','no']};assert list(map(len,strat.values()))==[6,10]
 expected_metadata={v['scene']:v['usgs_shadows'] for v in js(R/'OLD213_TEST_METADATA.json')['L8_scene_shadow_status']};assert all(status[k]==v for k,v in expected_metadata.items())
 source=old['source'];assert len(source)==535 and len({v['scene'] for v in source})==107
 for v in source:assert sha(v['image_path'])==v['image_sha256'] and sha(v['mask_path'])==v['mask_sha256']
 for v in js(R/'PHASE216_SOURCE_POOL_REQUEST.json')['selected']:assert sha(S/'data/cloudsen12_high_l1c/img_dir/train'/v['png_name'])==v['png_sha256']
 support={};mapping=np.array([0,2,1,1],np.int64)
 for pool in ['train','val','test']:
  counts=collections.defaultdict(lambda:np.zeros(3,np.int64));ign=collections.Counter()
  for v in pools[pool]:
   y=np.asarray(Image.open(v['mask_path']));assert y.ndim==2 and set(np.unique(y))<=set([0,1,2,3,255]);valid=y!=255;counts[v['scene']]+=np.bincount(mapping[y[valid]],minlength=3);ign[v['scene']]+=int((~valid).sum())
  support[pool]={'inputs':len(pools[pool]),'unique_scenes':len(counts),'scene_ids':sorted(counts),'class_reference_support':{k:v.tolist() for k,v in counts.items()},'ignore_pixels':dict(ign),'all_usgs_shadows_yes':pool!='test'}
 lock={'source':source,'val':pools['val'],'test':pools['test']};(R/'INPUT_SUPPORT_LOCK.json').write_text(json.dumps(lock,indent=2));(R/'L8_TRAIN_INPUT_LOCK.json').write_text(json.dumps(pools['train'],indent=2))
 for p in ['source','val','test']:
  c=js(R/f'EXPECTED_Source__{p}.json');assert len(c['per_input'])==len(lock[p]);assert set(c['per_group'])=={v['scene'] for v in lock[p]}
 receipt={'status':'passed','neural_calls':0,'epoch':time.time(),'manifest_SHA':sha(manifest),'shadow_metadata_SHA':sha(statuspath),'mapping_L8':[0,2,1,1],'mapping_Source':[0,1,1,2],'support':support,'unfiltered_scene_intersections':{'train_val':[],'train_test':[],'val_test':[]},'test_strata':strat,'source_inputs':535,'source_ROIs':107,'source_replay_inputs':256,'source_reference_train_test_reads':0,'source_replay_spatial_independence':'unknown; old spatial caveat preserved','input_lock_SHA':sha(R/'INPUT_SUPPORT_LOCK.json'),'train_input_lock_SHA':sha(R/'L8_TRAIN_INPUT_LOCK.json')}
 (R/'INPUT_PREFLIGHT_ACCEPTED.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()

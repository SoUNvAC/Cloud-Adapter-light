"""Independent CPU training/cost/interpolation audit; only endpoint heads are exported."""
from pathlib import Path
import json,hashlib,collections,tarfile,resource
R=Path('/home/scv/shared/phase224_20261024');S=Path('/home/scv/Cloud-Adapter-light/src')
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):return [json.loads(v) for v in p.read_text().splitlines()]
def main():
 import torch,numpy as np
 from PIL import Image
 torch.set_num_threads(2);assert (R/'EXIT_CODE').read_text().strip()=='0';complete=js(R/'COMPLETE.json');assert complete['units']==100460 and complete['GPU_seconds']<21600;lock=js(R/'EXPERIMENT_LOCK.json');counts=collections.Counter();source=torch.load(S/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth',map_location='cpu')['state_dict'];cache={};endpoints=set()
 for job,meta in lock['jobs'].items():
  root=R/job;seq=js(root/'TARGET_SOURCE_SCHEDULE.json');assert sha(root/'TARGET_SOURCE_SCHEDULE.json')==meta['original_schedule_SHA'];assert [{'source_sample_id':v['source_sample_id'],'source_png_name':v['source_png_name']} for v in seq]==js(R/'SOURCE_SEQUENCE_LOCK.json');assert sorted(collections.Counter(v['source_sample_id'] for v in seq).values())==[15]*96+[16]*160
  cfg=js(root/'TRAINING_CONFIG_LOCK.json');old=js(root/'HISTORICAL_TRAINING_CONFIG_LOCK.json');old['KL']['lambda']=4;assert cfg==old
  g=js(root/'GATE_ACCEPTED.json');assert g['target_loss_H']==g['historical_target_loss_D'] and abs(g['initial_KL'])<1e-6 and g['lambda']==4 and g['gate_updates']==0 and g['trainable_each']==1316516 and all(n.startswith('decode_head.') for n in g['grad_H'])
  initial=torch.load(root/'heads/initial.pth',map_location='cpu')['state_dict'];assert all(torch.equal(v,source['decode_head.'+n]) for n,v in initial.items());ledger=lines(root/'CALL_LEDGER.jsonl');assert [v['ordinal'] for v in ledger]==list(range(1,16005));c=collections.Counter(v['kind'] for v in ledger);assert c=={'target_full_segmentor':4001,'source_shared_backbone':4001,'source_teacher_head':4001,'source_student_head':4001};counts.update(c)
  access=js(root/'ACCESS_AUDIT.json');assert not access['forbidden_attempts'] and access['source_train_label_reads']==access['source_test_label_reads']==access['source_val_label_opens']==0
  loss=lines(root/'H_LOSSES.jsonl');assert len(loss)==4000;historical=lines(Path(meta['historical_root'])/f"seed{meta['seed']}/D_LOSSES.jsonl");assert [v['lr'] for v in loss]==[v['lr'] for v in historical]
  mapping=np.array([2,2,0,0,0,1,0] if meta['target']=='SPARCS' else [0,2,1,1],np.int64)
  for i,(row,v) in enumerate(zip(seq,loss)):
   assert v['step']==i+1 and v['target_input_SHA']==row['image_SHA']==historical[i]['target_input_SHA'] and v['source_sample_id']==row['source_sample_id'] and v['lambda']==4 and v['finite_gradients'];assert np.isfinite(v['total_loss']) and v['source_KL']>=-1e-6 and abs(v['source_KL_scaled4']-4*v['source_KL'])<1e-10 and abs(v['total_loss']-float(np.float32(np.float32(v['target_loss'])+np.float32(v['source_KL_scaled4']))))<1e-6
   p=row['mask_path']
   if p not in cache:
    with Image.open(p) as im:cache[p]=np.asarray(im).copy()
   yy,xx=[int(d['value']) for d in row['draws'][:2]];v0=cache[p][yy:yy+512,xx:xx+512];out=np.full(v0.shape,255,np.int64);valid=v0<len(mapping);out[valid]=mapping[v0[valid]]
   if row['flip']:assert row['flip_direction']=='horizontal';out=np.flip(out,1)
   assert hashlib.sha256(out[None].copy().tobytes()).hexdigest()==row['mask_SHA']
  end=torch.load(root/'heads/H_iter4000.pth',map_location='cpu');assert end['steps']==4000 and end['seed']==meta['seed'] and end['arm']=='H';assert sha(root/'heads/H_iter4000.pth')==js(root/'H_TRAIN_COMPLETE.json')['head_SHA'];endpoints.add(str((root/'heads/H_iter4000.pth').relative_to(R)));done=js(root/'TRAINING_COMPLETE.json');assert done['teacher_digest']==g['teacher_digest'] and done['nonhead_digest']==g['nonhead_digest'] and js(root/'H_TRAIN_COMPLETE.json')['frozen_nonhead_digest']==g['nonhead_digest']
  for f in [.25,.5,.75]:
   h=torch.load(R/f'interpolated_heads/{job}_H_a{f}.pth',map_location='cpu')['state_dict'];assert all(torch.equal(h[n],(v.double()+f*(end['state_dict'][n].double()-v.double())).to(v.dtype)) for n,v in initial.items())
 ev=lines(R/'evaluation/CALL_LEDGER.jsonl');assert [v['ordinal'] for v in ev]==list(range(64017,100461));counts.update(v['kind'] for v in ev);assert sum(counts.values())==100460
 for v in js(R/'evaluation/HISTORICAL_D_REPRODUCTION.json').values():assert v['all535_inputs_and107_ROIs_exact'] and v['prediction_SHA']==v['expected_SHA']
 (R/'REMOTE_CPU_ACCEPTANCE.json').write_text(json.dumps({'status':'passed','units':100460,'counts':dict(counts),'four_H_fits4000':True,'only_lambda_changed':True,'all16000_target_mask_augment_SHA_independently_checked':True,'source_RGB_sequence_exact':True,'historical_D_reproduced':True,'all12_interpolants_float64_exact':True,'new_model_calls':0,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},indent=2));files=[p for p in R.rglob('*') if p.is_file() and p.name!='DELIVERY_MANIFEST.json' and not p.name.endswith('.tmp') and (p.suffix!='.pth' or str(p.relative_to(R)) in endpoints)];m={str(p.relative_to(R)):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in files};(R/'DELIVERY_MANIFEST.json').write_text(json.dumps(m,indent=2));archive=R.parent/'phase224_delivery.tar.gz';partial=archive.with_name(archive.name+'.partial');assert not archive.exists() and not partial.exists()
 with tarfile.open(partial,'w:gz') as t:
  for p in files+[R/'DELIVERY_MANIFEST.json']:t.add(p,arcname=str(p.relative_to(R)),recursive=False)
 partial.replace(archive);print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'SHA':sha(archive),'remote_scope_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file())}))
if __name__=='__main__':main()

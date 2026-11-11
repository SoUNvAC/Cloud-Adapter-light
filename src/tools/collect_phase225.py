"""CPU-only exact weight-use, original-mask, fit/freeze and interpolation audit."""
from pathlib import Path
import json,hashlib,collections,tarfile,resource,sys
R=Path('/home/scv/shared/phase225_20261024');S=Path('/home/scv/Cloud-Adapter-light/src')
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):return [json.loads(v) for v in p.read_text().splitlines()]
def main():
 import torch,numpy as np
 from PIL import Image
 sys.path.insert(0,str(R/'code'));from phase225_weights import load
 torch.set_num_threads(2);assert (R/'EXIT_CODE').read_text().strip()=='0';complete=js(R/'COMPLETE.json');assert complete['units']==124268 and complete['GPU_seconds']<21600;source=torch.load(S/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth',map_location='cpu')['state_dict'];allcounts=collections.Counter();cache={};proofs={};endpoints=set();mapping=np.array([0,2,1,1],np.int64)
 for seed in [21701,21702]:
  root=R/f'seed{seed}';schedule=js(root/'TARGET_SOURCE_SCHEDULE.json');weights,proof=load(R/'weights',schedule);assert sha(root/'TARGET_SOURCE_SCHEDULE.json')==sha(Path('/home/scv/shared/phase222_20261024')/f'seed{seed}/TARGET_SOURCE_SCHEDULE.json');initial=torch.load(root/'heads/initial.pth',map_location='cpu')['state_dict'];assert all(torch.equal(v,source['decode_head.'+n]) for n,v in initial.items());g=js(root/'GATE_ACCEPTED.json');assert g['target_losses']['R']==g['target_losses']['Q']==g['historical_target_loss_D'] and all(abs(v)<1e-6 for v in g['initial_KL'].values()) and not g['weight_requires_grad'] and g['gate_updates']==0 and g['trainable_each']==1316516
  for grads in g['gradients'].values():assert grads and all(n.startswith('decode_head.') for n in grads)
  ledger=lines(root/'CALL_LEDGER.jsonl');assert [v['ordinal'] for v in ledger]==list(range(1,32009));counts=collections.Counter(v['kind'] for v in ledger);assert counts=={'target_full_segmentor':8002,'source_shared_backbone':8002,'source_teacher_head':8002,'source_student_head':8002};allcounts.update(counts);done=js(root/'TRAINING_COMPLETE.json');assert done['nonhead_digest']==g['nonhead_digest'] and done['teacher_digest']==g['teacher_digest'];access=js(root/'ACCESS_AUDIT.json');assert not access['forbidden_attempts'] and access['source_train_label_reads']==access['source_test_label_reads']==access['source_val_label_opens']==0
  for row in schedule:
   p=row['mask_path']
   if p not in cache:
    with Image.open(p) as im:cache[p]=np.asarray(im).copy()
   yy,xx=[int(v['value']) for v in row['draws'][:2]];v=cache[p][yy:yy+512,xx:xx+512];y=np.full(v.shape,255,np.int64);valid=v<4;y[valid]=mapping[v[valid]]
   if row['flip']:assert row['flip_direction']=='horizontal';y=np.flip(y,1)
   assert hashlib.sha256(y[None].copy().tobytes()).hexdigest()==row['mask_SHA']
  totals={}
  old=lines(Path('/home/scv/shared/phase222_20261024')/f'seed{seed}/D_LOSSES.jsonl')
  for arm in ['R','Q']:
   loss=lines(root/f'{arm}_LOSSES.jsonl');assert len(loss)==4000 and [v['lr'] for v in loss]==[v['lr'] for v in old];seen=[]
   for i,v in enumerate(loss):
    row=schedule[i];sid=str(row['source_sample_id']);csvweight=float(weights[sid][arm+'_weight']);used=float(np.float32(csvweight));assert v['step']==i+1 and v['target_input_SHA']==row['image_SHA']==old[i]['target_input_SHA'] and str(v['source_sample_id'])==sid and v['weight_CSV']==csvweight and v['weight_used']==used and not v['weight_requires_grad'] and v['gradient_finite'];assert np.isfinite(v['total_loss']) and v['source_KL']>=-1e-6 and abs(v['weighted_KL']-float(np.float32(np.float32(v['source_KL'])*np.float32(used))))<1e-6 and abs(v['total_loss']-float(np.float32(np.float32(v['target_loss'])+np.float32(v['weighted_KL']))))<1e-6;seen.append(used)
   assert abs(float(np.mean(seen))-1)<1e-6;totals[arm]={'actual_float32_weight_sum4000':float(np.sum(seen)),'mean':float(np.mean(seen)),'all4000_sample_id_and_weight_pairs_exact':True};ck=torch.load(root/f'heads/{arm}_iter4000.pth',map_location='cpu');assert ck['steps']==4000 and ck['seed']==seed and ck['arm']==arm and sha(root/f'heads/{arm}_iter4000.pth')==js(root/f'{arm}_TRAIN_COMPLETE.json')['head_SHA'];assert js(root/f'{arm}_TRAIN_COMPLETE.json')['frozen_nonhead_digest']==g['nonhead_digest'];endpoints.add(f'seed{seed}/heads/{arm}_iter4000.pth')
   for a in [.25,.5,.75]:
    h=torch.load(R/f'interpolated_heads/{seed}_{arm}_a{a}.pth',map_location='cpu')['state_dict'];assert all(torch.equal(h[n],(v.double()+a*(ck['state_dict'][n].double()-v.double())).to(v.dtype)) for n,v in initial.items())
  assert totals['R']['actual_float32_weight_sum4000']==totals['Q']['actual_float32_weight_sum4000'];proofs[str(seed)]={'frozen_weights':proof,'actual_training':totals}
 ev=lines(R/'evaluation/CALL_LEDGER.jsonl');assert [v['ordinal'] for v in ev]==list(range(64017,124269));allcounts.update(v['kind'] for v in ev);assert sum(allcounts.values())==124268
 for v in js(R/'evaluation/HISTORICAL_D_REPRODUCTION.json').values():assert v['all535_input_and107_ROI_confusions_exact'] and v['prediction_SHA']==v['expected_SHA']
 (R/'WEIGHT_USE_ACCEPTED.json').write_text(json.dumps(proofs,indent=2));(R/'REMOTE_CPU_ACCEPTANCE.json').write_text(json.dumps({'status':'passed','units':124268,'counts':dict(allcounts),'all16000_original_targets_masks_weights_finite_grads_checked':True,'new_model_calls':0,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},indent=2));files=[p for p in R.rglob('*') if p.is_file() and p.name!='DELIVERY_MANIFEST.json' and not p.name.endswith('.tmp') and (p.suffix!='.pth' or str(p.relative_to(R)) in endpoints)];m={str(p.relative_to(R)):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in files};(R/'DELIVERY_MANIFEST.json').write_text(json.dumps(m,indent=2));archive=R.parent/'phase225_delivery.tar.gz';partial=archive.with_name(archive.name+'.partial');assert not archive.exists() and not partial.exists()
 with tarfile.open(partial,'w:gz') as t:
  for p in files+[R/'DELIVERY_MANIFEST.json']:t.add(p,arcname=str(p.relative_to(R)),recursive=False)
 partial.replace(archive);print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'SHA':sha(archive),'remote_scope_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file())}))
if __name__=='__main__':main()

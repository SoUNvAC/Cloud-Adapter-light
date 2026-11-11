"""Independent CPU checks and bounded small shared delivery; no model forwards."""
from pathlib import Path
import collections,hashlib,json,tarfile,resource,math
R=Path('/home/scv/shared/phase221_20261024');SRC=Path('/home/scv/Cloud-Adapter-light/src');OLD=R.parent/'phase217_20261023'
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(v) for v in p.read_text().splitlines()]
def main():
 import numpy as np,torch
 from PIL import Image
 torch.set_num_threads(2);assert np.__version__=='1.26.4'
 assert (R/'EXIT_CODE').read_text().strip()=='0';done=js(R/'COMPLETE.json');assert done['units']==75018 and done['GPU_seconds']<14400
 archive=R.parent/'phase221_delivery.tar.gz';assert not archive.exists()
 sourcepath=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth';assert sha(sourcepath)=='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
 source=torch.load(sourcepath,map_location='cpu')['state_dict'];counts=collections.Counter();weightrows=js(R/'weight_preparation/WEIGHT_MANIFEST.json');assert len(weightrows)==256
 weightproof=[]
 for row in weightrows:
  sid=row['sample_id'];f=R/f'weight_preparation/masks/{sid}.npz';assert sha(f)==row['packed_masks_SHA']
  with np.load(f,allow_pickle=False) as z:tp=z['teacher_prediction'].reshape(-1);g=np.unpackbits(z['G_bits'],count=262144).astype(np.float32)+.5;p_high=np.unpackbits(z['P_bits'],count=262144)
  assert hashlib.sha256(tp.tobytes()).hexdigest()==row['teacher_prediction_SHA'];oldroot=R.parent/'phase219_20261023/weight_preparation';old={v['sample_id']:v for v in js(oldroot/'WEIGHT_MANIFEST.json')};cf=oldroot/f'confidence/{sid}.npy';assert sha(cf)==old[sid]['confidence_file_SHA'];confidence=np.load(cf,allow_pickle=False).reshape(-1);order=np.lexsort((np.arange(262144),-confidence));gg=np.full(262144,.5,np.float32);gg[order[:131072]]=1.5;assert np.array_equal(g,gg) and hashlib.sha256(g.tobytes()).hexdigest()==old[sid]['G_SHA']==row['G_SHA']
  n=np.empty_like(g);p=np.empty_like(g)
  for c in range(3):
   ids=np.where(tp==c)[0].astype('<i8');q=row['classes'][c];assert q['support']==len(ids)
   if not len(ids):assert q['mean_g'] is None;continue
   mean=float(sum(float(g[i]) for i in ids)/len(ids));assert mean==q['mean_g'];n[ids]=np.array([float(g[i])/mean for i in ids],np.float64).astype(np.float32);seed=int.from_bytes(hashlib.sha256(('phase221|'+sid+'|'+str(c)).encode()).digest()[:8],'big');ix=np.random.Generator(np.random.PCG64(seed)).permutation(len(ids)).astype('<i8');p[ids]=n[ids][ix];assert q['seed']==seed and hashlib.sha256(ix.tobytes()).hexdigest()==q['permutation_SHA'] and hashlib.sha256(ids[ix].tobytes()).hexdigest()==q['permuted_global_indices_SHA'];assert abs(n[ids].mean(dtype=np.float64)-1)<=1e-6 and np.array_equal(np.sort(n[ids]),np.sort(p[ids])) and np.array_equal(p_high[ids],g[ids][ix]==1.5)
  assert hashlib.sha256(n.tobytes()).hexdigest()==row['N_SHA'] and hashlib.sha256(p.tobytes()).hexdigest()==row['P_SHA'];weightproof.append({'sample_id':sid,'independent_CPU_float64_class_normalization_and_PCG64_checked':True,'N_SHA':row['N_SHA'],'P_SHA':row['P_SHA']})
 wp=lines(R/'weight_preparation/CALL_LEDGER.jsonl');assert [v['ordinal'] for v in wp]==list(range(1,513));counts.update(v['kind'] for v in wp)
 masks={};mapping=np.array([2,2,0,0,0,1,0],np.int64);gates=[]
 for seed in [21701,21702]:
  d=R/f'seed{seed}';schedule=js(d/'TARGET_SOURCE_SCHEDULE.json');assert len(schedule)==4000 and sha(d/'TARGET_SOURCE_SCHEDULE.json')==sha(OLD/f'seed{seed}/TARGET_SOURCE_SCHEDULE.json') and sha(d/'TRAINING_CONFIG_LOCK.json')==sha(OLD/f'seed{seed}/TRAINING_CONFIG_LOCK.json')
  assert [{'source_sample_id':v['source_sample_id'],'source_png_name':v['source_png_name']} for v in schedule]==js(R/'SOURCE_SEQUENCE_LOCK.json')
  h0=torch.load(d/'heads/initial.pth',map_location='cpu')['state_dict'];assert all(torch.equal(v,source['decode_head.'+n]) for n,v in h0.items())
  gate=js(d/'GATE_ACCEPTED.json');gates.append(gate);assert gate['target_loss_N']==gate['target_loss_P'] and gate['gate_updates']==0 and gate['weight_parameters']==0 and gate['trainable_each']==1316516 and abs(gate['initial_N_KL'])<1e-6 and abs(gate['initial_P_KL'])<1e-6
  assert all(n.startswith('decode_head.') for n in gate['grad_N']+gate['grad_P'])
  led=lines(d/'CALL_LEDGER.jsonl');assert [v['ordinal'] for v in led]==list(range(1,32009));cc=collections.Counter(v['kind'] for v in led);assert cc=={k:8002 for k in ['target_full_segmentor','source_shared_backbone','source_teacher_head','source_student_head']};counts.update(cc)
  access=js(d/'ACCESS_AUDIT.json');assert not access['forbidden_attempts'] and access['source_train_label_reads']==access['source_test_label_reads']==access['source_val_label_opens']==0
  complete=js(d/'TRAINING_COMPLETE.json');assert complete['nonhead_digest']==gate['nonhead_digest'] and complete['teacher_digest']==gate['teacher_digest']
  for v in schedule:
   p=v['mask_path']
   if p not in masks:
    with Image.open(p) as im:masks[p]=np.asarray(im).copy()
   y,x=[int(q['value']) for q in v['draws'][:2]];v0=masks[p][y:y+512,x:x+512];out=np.full(v0.shape,255,np.int64);ok=v0<7;out[ok]=mapping[v0[ok]]
   if v['flip']:assert v['flip_direction']=='horizontal';out=np.flip(out,1)
   assert hashlib.sha256(out[None].copy().tobytes()).hexdigest()==v['mask_SHA']
  oldlr=[v['lr'] for v in lines(OLD/f'seed{seed}/C_LOSSES.jsonl')]
  for arm in ['N','P']:
   endpoint=d/f'heads/{arm}_iter4000.pth';receipt=js(d/f'{arm}_TRAIN_COMPLETE.json');assert sha(endpoint)==receipt['head_SHA'] and receipt['frozen_nonhead_digest']==gate['nonhead_digest'] and receipt['teacher_digest']==gate['teacher_digest']
   ck=torch.load(endpoint,map_location='cpu');assert ck['steps']==4000 and ck['seed']==seed and ck['arm']==arm;h=ck['state_dict']
   for a in [.25,.5,.75]:
    hh=torch.load(R/f'interpolated_heads/{seed}_{arm}_a{a}.pth',map_location='cpu')['state_dict'];assert all(torch.equal(hh[n],(v.double()+a*(h[n].double()-v.double())).to(v.dtype)) for n,v in h0.items())
   losses=lines(d/f'{arm}_LOSSES.jsonl');assert len(losses)==4000 and [v['lr'] for v in losses]==oldlr
   for i,v in enumerate(losses):
    assert v['step']==i+1 and v['target_input_SHA']==schedule[i]['image_SHA'] and v['source_sample_id']==schedule[i]['source_sample_id'] and v['gradient_finite'] and not v['weights_require_grad']
    raw=v['source_KL_unweighted'];weighted=v['source_KL_weighted'];assert all(math.isfinite(q) for q in [raw,weighted,v['total_loss']]);assert v['source_KL']==weighted and abs(v['total_loss']-v['target_loss']-weighted)<8*np.finfo(np.float32).eps*max(1,abs(v['total_loss']))
    assert abs(sum(q['weighted_KL_contribution'] for q in v['teacher_class_KL'])-weighted)<=2e-6 and sum(q['support'] for q in v['teacher_class_KL'])==262144
    for q in v['teacher_class_KL']:assert q['weight_mean'] is None or abs(q['weight_mean']-1)<=1e-6
   assert js(d/f'{arm}_INITIAL.json')['source_head_matches'] and sum(v['parameters'] for v in js(d/f'{arm}_INITIAL.json')['optimizer_groups'])==1316516
  fixture=js(d/'SMALL_FORMULA_CHECK.json');arr=np.array(fixture['source_semantic'],float);b=np.array(fixture['student_semantic'],float);p=np.exp(arr-arr.max(1,keepdims=True));p/=p.sum(1,keepdims=True);q=np.exp(b-b.max(1,keepdims=True));q/=q.sum(1,keepdims=True);kl=(p*np.log(p/q)).sum(1)
  assert np.allclose(kl,fixture['CPU_explicit_pixel_KL'],atol=1e-6,rtol=0)
  for arm in ['N','P']:assert abs(np.mean(kl*np.array(fixture['weights'][arm]))-fixture['weighted_KL_GPU'][arm])<1e-6
 assert len(masks)==60 and gates[0]['nonhead_digest']==gates[1]['nonhead_digest'] and gates[0]['teacher_digest']==gates[1]['teacher_digest']
 ev=R/'evaluation';led=lines(ev/'CALL_LEDGER.jsonl');assert [v['ordinal'] for v in led]==list(range(64529,75019));counts.update(v['kind'] for v in led);assert sum(counts.values())==75018 and all(v['matches216'] for v in js(ev/'SOURCE_PARITY.json'))
 receipt={'status':'passed','total_units':75018,'calls':dict(counts),'four_fits4000':True,'unchanged_initial_Source':True,'independent_class_normalization_weight_proofs':weightproof,'numpy':np.__version__,'all_8000_target_crop_flip_masks_checked':True,'all_16000_weighted_losses_checked':True,'interpolation_exact_CPU_float64':True,'no_source_train_test_reference_reads':True,'shared_4000_source_sequence_and_original_target_sequences':True,'new_model_calls':0,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 assert receipt['CPU_peak_bytes']<16*1024**3;(R/'REMOTE_CPU_ACCEPTANCE.json').write_text(json.dumps(receipt,indent=2))
 names=[p for p in R.rglob('*') if p.is_file() and (p.suffix not in ['.pth','.npy'] or p.name in ['N_iter4000.pth','P_iter4000.pth']) and not p.name.endswith('.tmp') and p.name!='DELIVERY_MANIFEST.json']
 manifest={str(p.relative_to(R)):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in names};(R/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
 partial=archive.with_name(archive.name+'.partial');assert not partial.exists()
 with tarfile.open(partial,'w:gz') as f:
  for p in names+[R/'DELIVERY_MANIFEST.json']:f.add(p,arcname=str(p.relative_to(R)),recursive=False)
 partial.replace(archive)
 print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'SHA':sha(archive),'remote_scope_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file()),'units':75018}))
if __name__=='__main__':main()

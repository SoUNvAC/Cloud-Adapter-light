"""Independent CPU checks and bounded small shared delivery; no model forwards."""
from pathlib import Path
import collections,hashlib,json,tarfile,resource,math
R=Path('/home/scv/shared/phase219_20261023');SRC=Path('/home/scv/Cloud-Adapter-light/src');OLD=R.parent/'phase217_20261023'
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(v) for v in p.read_text().splitlines()]
def main():
 import numpy as np,torch
 from PIL import Image
 torch.set_num_threads(2);assert np.__version__=='1.26.4'
 assert (R/'EXIT_CODE').read_text().strip()=='0';done=js(R/'COMPLETE.json');assert done['units']==73946 and done['GPU_seconds']<14400
 archive=R.parent/'phase219_delivery.tar.gz';assert not archive.exists()
 sourcepath=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth';assert sha(sourcepath)=='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
 source=torch.load(sourcepath,map_location='cpu')['state_dict'];counts=collections.Counter();weightrows=js(R/'weight_preparation/WEIGHT_MANIFEST.json');assert len(weightrows)==256
 weightproof=[]
 for row in weightrows:
  sid=row['sample_id'];f=R/f'weight_preparation/confidence/{sid}.npy';p=R/f'weight_preparation/masks/{sid}.npz';assert sha(f)==row['confidence_file_SHA'] and sha(p)==row['packed_masks_SHA']
  c=np.load(f,allow_pickle=False).reshape(-1);assert c.size==262144 and np.isfinite(c).all() and hashlib.sha256(c.tobytes()).hexdigest()==row['confidence_raw_SHA']
  # A separate lexicographic implementation exposes the row-major tie rule.
  rank=np.lexsort((np.arange(c.size,dtype=np.int64),-c));g=np.full(c.size,.5,np.float32);g[rank[:131072]]=1.5
  seed=int.from_bytes(hashlib.sha256(('phase219|'+sid).encode()).digest()[:8],'big');perm=np.random.Generator(np.random.PCG64(seed)).permutation(c.size).astype('<i8');s=g[perm]
  assert seed==row['local_PCG64_seed'] and hashlib.sha256(perm.tobytes()).hexdigest()==row['permutation_SHA'] and hashlib.sha256(rank.astype('<i8').tobytes()).hexdigest()==row['stable_rank_SHA']
  with np.load(p,allow_pickle=False) as z:
   for a,v in [('G',g),('S',s)]:
    decoded=np.unpackbits(z[a],count=c.size).astype(np.float32)+.5
    assert np.array_equal(decoded,v) and hashlib.sha256(v.astype('<f4').tobytes()).hexdigest()==row[a+'_SHA'] and np.count_nonzero(v==1.5)==131072 and v.mean()==1
  weightproof.append({'sample_id':sid,'independent_lexsort_and_PCG64_checked':True,'G_S_differing_pixels':int(np.count_nonzero(g!=s)),'confidence_SHA':row['confidence_raw_SHA'],'G_SHA':row['G_SHA'],'S_SHA':row['S_SHA']})
 wp=lines(R/'weight_preparation/CALL_LEDGER.jsonl');assert [v['ordinal'] for v in wp]==list(range(1,513));counts.update(v['kind'] for v in wp)
 masks={};mapping=np.array([2,2,0,0,0,1,0],np.int64);gates=[]
 for seed in [21701,21702]:
  d=R/f'seed{seed}';schedule=js(d/'TARGET_SOURCE_SCHEDULE.json');assert len(schedule)==4000 and sha(d/'TARGET_SOURCE_SCHEDULE.json')==sha(OLD/f'seed{seed}/TARGET_SOURCE_SCHEDULE.json') and sha(d/'TRAINING_CONFIG_LOCK.json')==sha(OLD/f'seed{seed}/TRAINING_CONFIG_LOCK.json')
  assert [{'source_sample_id':v['source_sample_id'],'source_png_name':v['source_png_name']} for v in schedule]==js(R/'SOURCE_SEQUENCE_LOCK.json')
  h0=torch.load(d/'heads/initial.pth',map_location='cpu')['state_dict'];assert all(torch.equal(v,source['decode_head.'+n]) for n,v in h0.items())
  gate=js(d/'GATE_ACCEPTED.json');gates.append(gate);assert gate['target_loss_G']==gate['target_loss_S'] and gate['gate_updates']==0 and gate['weight_parameters']==0 and gate['trainable_each']==1316516 and abs(gate['initial_G_KL'])<1e-6 and abs(gate['initial_S_KL'])<1e-6
  assert all(n.startswith('decode_head.') for n in gate['grad_G']+gate['grad_S'])
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
  for arm in ['G','S']:
   endpoint=d/f'heads/{arm}_iter4000.pth';receipt=js(d/f'{arm}_TRAIN_COMPLETE.json');assert sha(endpoint)==receipt['head_SHA'] and receipt['frozen_nonhead_digest']==gate['nonhead_digest'] and receipt['teacher_digest']==gate['teacher_digest']
   ck=torch.load(endpoint,map_location='cpu');assert ck['steps']==4000 and ck['seed']==seed and ck['arm']==arm;h=ck['state_dict']
   for a in [.25,.5,.75]:
    hh=torch.load(R/f'interpolated_heads/{seed}_{arm}_a{a}.pth',map_location='cpu')['state_dict'];assert all(torch.equal(hh[n],(v.double()+a*(h[n].double()-v.double())).to(v.dtype)) for n,v in h0.items())
   losses=lines(d/f'{arm}_LOSSES.jsonl');assert len(losses)==4000 and [v['lr'] for v in losses]==oldlr
   for i,v in enumerate(losses):
    assert v['step']==i+1 and v['target_input_SHA']==schedule[i]['image_SHA'] and v['source_sample_id']==schedule[i]['source_sample_id'] and v['gradient_finite'] and not v['weights_require_grad']
    hi=v['high_weight_region_KL'];lo=v['low_weight_region_KL'];raw=v['source_KL_unweighted'];weighted=v['source_KL_weighted'];assert all(math.isfinite(q) for q in [hi,lo,raw,weighted,v['total_loss']])
    assert abs(raw-(hi+lo)/2)<2e-6 and abs(weighted-(.75*hi+.25*lo))<2e-6 and v['source_KL']==weighted and abs(v['total_loss']-v['target_loss']-weighted)<1e-3
   assert js(d/f'{arm}_INITIAL.json')['source_head_matches'] and sum(v['parameters'] for v in js(d/f'{arm}_INITIAL.json')['optimizer_groups'])==1316516
  fixture=js(d/'SMALL_FORMULA_CHECK.json');arr=np.array(fixture['source_semantic'],float);b=np.array(fixture['student_semantic'],float);p=np.exp(arr-arr.max(1,keepdims=True));p/=p.sum(1,keepdims=True);q=np.exp(b-b.max(1,keepdims=True));q/=q.sum(1,keepdims=True);kl=(p*np.log(p/q)).sum(1)
  assert np.allclose(kl,fixture['pixel_KL_GPU'],atol=1e-6,rtol=0)
  for arm in ['G','S']:assert abs(np.mean(kl*np.array(fixture['weights'][arm]))-fixture['weighted_KL_GPU'][arm])<1e-6
 assert len(masks)==60 and gates[0]['nonhead_digest']==gates[1]['nonhead_digest'] and gates[0]['teacher_digest']==gates[1]['teacher_digest']
 ev=R/'evaluation';led=lines(ev/'CALL_LEDGER.jsonl');assert [v['ordinal'] for v in led]==list(range(64529,73947));counts.update(v['kind'] for v in led);assert sum(counts.values())==73946 and all(v['matches216'] for v in js(ev/'SOURCE_PARITY.json'))
 receipt={'status':'passed','total_units':73946,'calls':dict(counts),'four_fits4000':True,'unchanged_initial_Source':True,'independent_lexsort_weight_proofs':weightproof,'numpy':np.__version__,'all_8000_target_crop_flip_masks_checked':True,'all_16000_weighted_losses_checked':True,'interpolation_exact_CPU_float64':True,'no_source_train_test_reference_reads':True,'shared_4000_source_sequence_and_original_target_sequences':True,'new_model_calls':0,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 assert receipt['CPU_peak_bytes']<16*1024**3;(R/'REMOTE_CPU_ACCEPTANCE.json').write_text(json.dumps(receipt,indent=2))
 names=[p for p in R.rglob('*') if p.is_file() and p.suffix not in ['.pth','.npy'] and not p.name.endswith('.tmp') and p.name!='DELIVERY_MANIFEST.json']
 manifest={str(p.relative_to(R)):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in names};(R/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
 partial=archive.with_name(archive.name+'.partial');assert not partial.exists()
 with tarfile.open(partial,'w:gz') as f:
  for p in names+[R/'DELIVERY_MANIFEST.json']:f.add(p,arcname=str(p.relative_to(R)),recursive=False)
 partial.replace(archive)
 print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'SHA':sha(archive),'remote_scope_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file()),'units':73946}))
if __name__=='__main__':main()

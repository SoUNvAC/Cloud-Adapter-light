"""No forwards: all training/augmentation/head/cost audits and small delivery."""
from pathlib import Path
import collections,hashlib,json,tarfile,time,resource
ROOT=Path('/home/scv/shared/phase222_20261024');SRC=Path('/home/scv/Cloud-Adapter-light/src')
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 import numpy as np,torch
 from PIL import Image
 torch.set_num_threads(2)
 assert (ROOT/'EXIT_CODE').read_text().strip()=='0';archive=ROOT.parent/'phase222_delivery.tar.gz';assert not archive.exists()
 complete=js(ROOT/'COMPLETE.json');assert complete['units']==99190 and complete['GPU_seconds']<21600
 assert sha(SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')=='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
 source=torch.load(SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth',map_location='cpu')['state_dict'];mask_cache={};mapping=np.array([0,2,1,1],np.int64);allcounts=collections.Counter();gate=[]
 schedules={};models={};checked_images=set()
 for s in [21701,21702]:
  r=ROOT/f'seed{s}';schedule=js(r/'TARGET_SOURCE_SCHEDULE.json');schedules[s]=sha(r/'TARGET_SOURCE_SCHEDULE.json');assert len(schedule)==4000
  assert [{'source_sample_id':x['source_sample_id'],'source_png_name':x['source_png_name']} for x in schedule]==js(ROOT/'SOURCE_SEQUENCE_LOCK.json')
  assert sorted(collections.Counter(x['source_sample_id'] for x in schedule).values())==[15]*96+[16]*160
  initial=torch.load(r/'heads/initial.pth',map_location='cpu')['state_dict'];assert all(torch.equal(v,source['decode_head.'+n]) for n,v in initial.items())
  g=js(r/'GATE_ACCEPTED.json');assert g['target_loss_C']==g['target_loss_D'] and abs(g['initial_KL'])<1e-6 and g['trainable_each']==1316516;gate.append(g)
  ledger=[json.loads(x) for x in (r/'CALL_LEDGER.jsonl').read_text().splitlines()];assert [x['ordinal'] for x in ledger]==list(range(1,20006));count=collections.Counter(x['kind'] for x in ledger);allcounts.update(count)
  assert count=={'target_full_segmentor':8002,'source_shared_backbone':4001,'source_teacher_head':4001,'source_student_head':4001}
  completed=js(r/'TRAINING_COMPLETE.json');access=js(r/'ACCESS_AUDIT.json')
  assert completed['total_units']==20005 and not access['forbidden_attempts'] and access['source_train_label_reads']==access['source_test_label_reads']==access['source_val_label_opens']==0
  assert completed['nonhead_digest']==g['nonhead_digest'] and completed['teacher_digest']==g['teacher_digest'] and all(n.startswith('decode_head.') for n in g['grad_C']+g['grad_D'])
  locked={v['name']:v for v in js(ROOT/'L8_TRAIN_INPUT_LOCK.json')}
  for row in schedule:
   p=row['mask_path'];record=locked[Path(p).name];
   if p not in checked_images:
    assert sha(Path(p))==record['mask_sha256'] and sha(Path(row['image_path']))==record['image_sha256'];checked_images.add(p)
   if p not in mask_cache:
    with Image.open(p) as im:mask_cache[p]=np.asarray(im).copy()
   y,x=[int(d['value']) for d in row['draws'][:2]];v=mask_cache[p][y:y+512,x:x+512];out=np.full(v.shape,255,np.int64);valid=v<4;out[valid]=mapping[v[valid]]
   if row['flip']:assert row['flip_direction']=='horizontal';out=np.flip(out,1)
   assert hashlib.sha256(out[None].copy().tobytes()).hexdigest()==row['mask_SHA']
  lr={}
  for a in ['C','D']:
   ck=torch.load(r/f'heads/{a}_iter4000.pth',map_location='cpu');assert ck['steps']==4000 and ck['seed']==s;h=ck['state_dict'];models[s,a]=h
   assert sha(r/f'heads/{a}_iter4000.pth')==js(r/f'{a}_TRAIN_COMPLETE.json')['head_SHA']
   losses=[json.loads(x) for x in (r/f'{a}_LOSSES.jsonl').read_text().splitlines()];assert len(losses)==4000
   assert all(x['step']==i+1 and x['target_input_SHA']==schedule[i]['image_SHA'] for i,x in enumerate(losses));lr[a]=[x['lr'] for x in losses]
   assert all(np.isfinite(x['source_KL']) and x['source_KL']>=-1e-6 and np.isfinite(x['total_loss']) for x in losses)
   assert a=='D' or all(x['source_KL']==0 for x in losses)
   assert all(x['source_sample_id']==(schedule[i]['source_sample_id'] if a=='D' else None) for i,x in enumerate(losses))
   assert sum(g['parameters'] for g in js(r/f'{a}_INITIAL.json')['optimizer_groups'])==1316516
   endreceipt=js(r/f'{a}_TRAIN_COMPLETE.json');assert endreceipt['frozen_nonhead_digest']==completed['nonhead_digest'] and endreceipt['teacher_digest']==completed['teacher_digest']
   for f in [.25,.5,.75]:
    t=torch.load(ROOT/f'interpolated_heads/{s}_{a}_a{f}.pth',map_location='cpu')['state_dict'];assert all(torch.equal(t[n],(v.double()+f*(h[n].double()-v.double())).to(v.dtype)) for n,v in initial.items())
  assert lr['C']==lr['D']
 assert schedules[21701]!=schedules[21702] and 0<len(mask_cache)<=1560
 assert gate[0]['nonhead_digest']==gate[1]['nonhead_digest'] and gate[0]['teacher_digest']==gate[1]['teacher_digest']
 evledger=[json.loads(x) for x in (ROOT/'evaluation/CALL_LEDGER.jsonl').read_text().splitlines()];assert [x['ordinal'] for x in evledger]==list(range(40011,99191));allcounts.update(x['kind'] for x in evledger)
 assert sum(allcounts.values())==99190 and all(x['matches216'] for x in js(ROOT/'evaluation/SOURCE_PARITY.json'))
 receipt={'status':'passed','calls':dict(allcounts),'total_units':99190,'four_fits4000':True,'initials_match_original_Source':True,'two_seeds_different_target_sequences':True,'fixed_source_sequence_both_seeds':True,'all8000_target_parent_crop_flip_masks_independently_checked':True,'mapping_L8':[0,2,1,1],'unique_training_masks_checked':len(mask_cache),'L8_input_preflight':js(ROOT/'INPUT_PREFLIGHT_ACCEPTED.json'),'all_internal_interpolants_exact_float64':True,'same_source_teacher_and_backbone':True,'source_train_and_test_labels_read':0,'new_model_calls':0,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 (ROOT/'REMOTE_CPU_ACCEPTANCE.json').write_text(json.dumps(receipt,indent=2))
 endpoint_names={f'seed{s}/heads/{a}_iter4000.pth' for s in [21701,21702] for a in ['C','D']}
 names=[p for p in ROOT.rglob('*') if p.is_file() and (p.suffix!='.pth' or str(p.relative_to(ROOT)) in endpoint_names) and not p.name.endswith('.tmp') and p.name!='DELIVERY_MANIFEST.json']
 manifest={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in names};(ROOT/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
 partial=archive.with_name(archive.name+'.partial');assert not partial.exists()
 with tarfile.open(partial,'w:gz') as t:
  for p in names+[ROOT/'DELIVERY_MANIFEST.json']:t.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
 partial.replace(archive)
 print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'SHA':sha(archive),'remote_scope_bytes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),'receipt':receipt}))
if __name__=='__main__':main()

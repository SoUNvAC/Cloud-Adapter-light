"""CPU independent fit, FP32 anchor, target-mask and ledger audit; head-only delivery."""
from pathlib import Path
import json,hashlib,collections,sys,tarfile,resource
R=Path('/home/scv/shared/phase227_20261025');S=Path('/home/scv/Cloud-Adapter-light/src')
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):return [json.loads(v) for v in p.read_text().splitlines()]
def main():
 import numpy as np,torch
 from PIL import Image
 torch.set_num_threads(2);assert (R/'EXIT_CODE').read_text().strip()=='0';assert js(R/'COMPLETE.json')['units']==102217 and js(R/'COMPLETE.json')['GPU_seconds']<21600;lock=js(R/'EXPERIMENT_LOCK.json');ck=torch.load(S/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth',map_location='cpu')['state_dict'];initial={k[len('decode_head.'):]:v for k,v in ck.items() if k.startswith('decode_head.')};assert sum(v.numel() for v in initial.values())==1316516;torch.save({'state_dict':initial,'Source_SHA':'64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'},R/'anchor_head.pth');endpoints={'anchor_head.pth'};counts=collections.Counter();cache={};fits={}
 for job,meta in lock['jobs'].items():
  root=R/job;old=Path(f"/home/scv/shared/phase{meta['old_phase']}_{meta['date']}/seed{meta['seed']}");g=js(root/'GATE_ACCEPTED.json');anc=js(root/'ANCHOR_PARAMETERS.json');assert g['gate_updates']==0 and g['trainable_each']==anc['numel']==1316516 and anc['independent_storage'] and anc['buffer_elements']==0 and g['anchor_digest']==anc['anchor_digest'];assert all(torch.equal(v,initial[k]) for k,v in torch.load(root/'heads/initial.pth',map_location='cpu')['state_dict'].items());assert js(root/'FORMULA_ACCEPTED.json')['status']=='passed'
  for a,v in g['arms'].items():assert v['initial_Omega']==0 and v['anchor_gradient_zero'] and v['target_loss']==g['historical_target_loss_C']==js(root/'HISTORICAL_GATE_ACCEPTED.json')['target_loss_C'] and v['gradients'] and all(n.startswith('decode_head.') for n in v['gradients'])
  access=js(root/'ACCESS_AUDIT.json');assert not access['forbidden_attempts'] and all(access[k]==0 for k in ['source_train_RGB_reads','source_train_label_reads','source_test_label_reads','source_val_label_opens','source_training_uses','source_gate_uses','source_teacher_forward','source_replay_cache_reads']);led=lines(root/'CALL_LEDGER.jsonl');assert [v['ordinal'] for v in led]==list(range(1,8003)) and collections.Counter(v['kind'] for v in led)=={'target_full_segmentor':8002};counts.update(v['kind'] for v in led);schedule=js(root/'TARGET_SOURCE_SCHEDULE.json');assert sha(root/'TARGET_SOURCE_SCHEDULE.json')==meta['schedule_SHA']==sha(old/'TARGET_SOURCE_SCHEDULE.json');mapping=np.array([2,2,0,0,0,1,0] if meta['domain']=='SPARCS' else [0,2,1,1],np.int64)
  for row in schedule:
   if row['mask_path'] not in cache:
    with Image.open(row['mask_path']) as im:cache[row['mask_path']]=np.asarray(im).copy()
   yy,xx=[int(z['value']) for z in row['draws'][:2]];v=cache[row['mask_path']][yy:yy+512,xx:xx+512];y=np.full(v.shape,255,np.int64);valid=v<len(mapping);y[valid]=mapping[v[valid]]
   if row['flip']:assert row['flip_direction']=='horizontal';y=np.flip(y,1)
   assert hashlib.sha256(y[None].copy().tobytes()).hexdigest()==row['mask_SHA']
  oldloss=lines(old/'C_LOSSES.jsonl');done=js(root/'TRAINING_COMPLETE.json');assert done['anchor_digest']==g['anchor_digest'] and done['nonhead_digest']==g['nonhead_digest'];fit={}
  for a,gamma in [('W1',1),('W10',10)]:
   rr=lines(root/f'{a}_LOSSES.jsonl');assert len(rr)==4000;assert js(root/f'{a}_INITIAL.json')['optimizer_groups']==js(old/'C_INITIAL.json')['optimizer_groups'];assert js(root/f'{a}_INITIAL.json')['source_head_matches']
   for i,v in enumerate(rr):
    assert v['step']==i+1 and v['target_input_SHA']==schedule[i]['image_SHA']==oldloss[i]['target_input_SHA'] and v['lr']==oldloss[i]['lr'] and v['source_sample_id'] is None and v['gamma']==gamma and v['gradient_finite'];assert v['Omega']>=0 and np.isfinite(v['total_loss']) and abs(v['scaled_Omega']-float(np.float32(gamma*np.float32(v['Omega']))))<1e-5 and abs(v['total_loss']-float(np.float32(np.float32(v['target_loss'])+np.float32(v['scaled_Omega']))))<1e-5 and np.isclose(v['L2_distance_before_update'],np.sqrt(2*v['Omega']),rtol=1e-6,atol=1e-6)
   hpath=root/f'heads/{a}_iter4000.pth';h=torch.load(hpath,map_location='cpu');receipt=js(root/f'{a}_TRAIN_COMPLETE.json');assert sha(hpath)==receipt['head_SHA'] and h['steps']==4000 and h['arm']==a and h['seed']==meta['seed'];assert set(h['state_dict'])==set(initial);omega=.5*sum(((v.float()-initial[n].float())**2).sum() for n,v in h['state_dict'].items());assert np.isclose(float(omega),receipt['endpoint_Omega'],rtol=1e-5,atol=1e-5) and np.isclose(float((2*omega).sqrt()),receipt['endpoint_L2'],rtol=1e-5,atol=1e-5)
   for alpha in [.25,.5,.75]:
    interpolated=torch.load(R/meta['domain']/f'interpolated_heads/{meta["seed"]}_{a}_a{alpha}.pth',map_location='cpu')['state_dict'];assert all(torch.equal(interpolated[n],(v.double()+alpha*(h['state_dict'][n].double()-v.double())).to(v.dtype)) for n,v in initial.items())
   endpoints.add(job+f'/heads/{a}_iter4000.pth');fit[a]={'Omega_CPU':float(omega),'Omega_GPU':receipt['endpoint_Omega'],'head_SHA':sha(hpath),'steps':4000,'gamma':gamma,'optimizer_exact_original_C':True}
  fits[job]=fit
 for domain,start,end in [('SPARCS',32009,42500),('L8',42501,102217)]:
  e=R/domain/'evaluation';led=lines(e/'CALL_LEDGER.jsonl');assert [v['ordinal'] for v in led]==list(range(start,end+1));counts.update(v['kind'] for v in led)
  for v in js(e/'HISTORICAL_D_REPRODUCTION.json').values():assert v['all535_input_and107_ROI_confusions_exact'] and v['prediction_SHA']==v['expected_SHA']
 assert sum(counts.values())==102217 and counts['target_full_segmentor']==32008 and counts['diagnostic_teacher_head']==535
 acceptance={'status':'passed','units':102217,'counts':dict(counts),'all32000_target_masks_and_losses_checked':True,'all8_anchors_and_optimizers_checked':True,'training_source_RGB_label_cache_teacher_access_zero':True,'fits':fits,'new_model_calls':0,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024};assert acceptance['CPU_peak_bytes']<16*1024**3;(R/'REMOTE_CPU_ACCEPTANCE.json').write_text(json.dumps(acceptance,indent=2));files=[p for p in R.rglob('*') if p.is_file() and p.name!='DELIVERY_MANIFEST.json' and not p.name.endswith('.tmp') and (p.suffix!='.pth' or str(p.relative_to(R)) in endpoints)];m={str(p.relative_to(R)):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in files};(R/'DELIVERY_MANIFEST.json').write_text(json.dumps(m,indent=2));archive=R.parent/'phase227_delivery.tar.gz';part=archive.with_name(archive.name+'.partial');assert not archive.exists() and not part.exists()
 with tarfile.open(part,'w:gz') as tar:
  for p in files+[R/'DELIVERY_MANIFEST.json']:tar.add(p,arcname=str(p.relative_to(R)),recursive=False)
 part.replace(archive);print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'SHA':sha(archive),'remote_scope_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file())}))
if __name__=='__main__':main()

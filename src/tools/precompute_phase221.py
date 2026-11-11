"""Freeze256 class-normalized confidence weights under unchanged219 TRAIN runtime; zero fitting."""
from pathlib import Path
import argparse,json,hashlib,os,resource,sys,time,traceback
R=Path('/home/scv/shared/phase221_20261024');SRC=Path('/home/scv/Cloud-Adapter-light/src')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text())
def main():
 out=R/'weight_preparation';out.mkdir(exist_ok=False);os.chdir(SRC);sys.path[:0]=[str(R/'code'),str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1'
 import numpy as np,torch
 import torch.nn.functional as F
 from export_phase12_onnx import build_wrapper
 from validate_phase12_onnxruntime import load_rgb_image
 from phase219_weighting import build,rawsha
 from phase221_weighting import normalize,decode,boundary_tests
 assert np.__version__=='1.26.4';torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=True)
 from controller_phase221 import host
 start=time.monotonic();units=0;ledger=(out/'CALL_LEDGER.jsonl').open('x',buffering=1);request=js(R/'PHASE216_SOURCE_POOL_REQUEST.json');source=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth';assert sha(source)=='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9';(out/'masks').mkdir();(out/'EXECUTION_STARTED.json').write_text(json.dumps({'PID':os.getpid(),'epoch':time.time(),'units_planned':512,'train_runtime':True,'host':host()}))
 def count(kind,name):
  nonlocal units
  units+=1;ledger.write(json.dumps({'ordinal':units,'kind':kind,'sample_id':name})+'\n')
 def deny(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=os.fsdecode(args[0]).replace('\\','/')
   if 'cloudsen12_high_l1c/ann_dir/' in p or 'cloudsen12_high_l1c/img_dir/test/' in p:raise RuntimeError('No source references or official test in weight preparation')
 sys.addaudithook(deny)
 try:
  wrapper,_=build_wrapper(argparse.Namespace(config='configs/protocol/phase64_source_parent_rgb.py',checkpoint=str(source),precision='fp32',active_block_indices=None));model=wrapper.segmentor.eval()
  for p in model.parameters():p.requires_grad=False
  initial={n:t.detach().cpu().clone() for n,t in list(model.named_parameters())+list(model.named_buffers())};manifest=[]
  oldroot=R.parent/'phase219_20261023/weight_preparation';old={v['sample_id']:v for v in js(oldroot/'WEIGHT_MANIFEST.json')};old220={v['sample_id']:v for v in [json.loads(q) for q in (R.parent/'phase220_20261024/replay/REPLAY_WEIGHTS.jsonl').read_text().splitlines()]};freq={}
  for v in js(R/'SOURCE_SEQUENCE_LOCK.json'):freq[v['source_sample_id']]=freq.get(v['source_sample_id'],0)+1
  (out/'PURE_ARRAY_BOUNDARY_TESTS.json').write_text(json.dumps(boundary_tests(),indent=2))
  with torch.no_grad():
   for i,r in enumerate(request['selected'],1):
    sid=r['sample_id'];path=SRC/'data/cloudsen12_high_l1c/img_dir/train'/r['png_name'];assert sha(path)==r['png_sha256'];x=torch.from_numpy(load_rgb_image(path,512)).cuda();count('source_weight_backbone',sid);features=model.extract_feat(wrapper.normalize(x));count('source_weight_teacher_head',sid);cls,masks=model.decode_head(features,None);s=torch.einsum('bqc,bqhw->bchw',F.softmax(cls[-1].float(),-1)[...,:-1],F.interpolate(masks[-1].float(),size=(512,512),mode='bilinear',align_corners=False).sigmoid());p=F.log_softmax(s.float(),1).exp();c=p.max(1).values[0].cpu().numpy().copy()
    before=np.random.get_state();g,unused,gproof=build(c,sid);after=np.random.get_state();assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    for key in ['confidence_raw_SHA','G_SHA','S_SHA','permutation_SHA','stable_rank_SHA']:assert gproof[key]==old[sid][key]
    cp=oldroot/f'confidence/{sid}.npy';assert sha(cp)==old[sid]['confidence_file_SHA'] and np.array_equal(np.load(cp,allow_pickle=False),c)
    with np.load(oldroot/f'masks/{sid}.npz',allow_pickle=False) as zz:assert np.array_equal(np.unpackbits(zz['G'],count=262144).astype(np.float32)+.5,g)
    zz=s.clamp_min(0);tp=(zz/zz.sum(1,keepdim=True).clamp_min(1e-12)).argmax(1)[0].cpu().numpy().astype(np.uint8);assert rawsha(tp)==old220[sid]['teacher_prediction_SHA'];before=np.random.get_state();n,w,proof,pbits=normalize(g,tp,sid);after=np.random.get_state();assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    file=out/f'masks/{sid}.npz';np.savez_compressed(file,G_bits=np.packbits(g==1.5),P_bits=pbits,teacher_prediction=tp);proof.update({'confidence_raw_SHA':gproof['confidence_raw_SHA'],'original219_G_SHA':gproof['G_SHA'],'original219_S_SHA':gproof['S_SHA'],'repeat_count':freq[sid],'packed_masks_SHA':sha(file),'source_image_SHA':r['png_sha256'],'source_png_name':r['png_name']})
    with np.load(file,allow_pickle=False) as z:assert rawsha(decode(z,proof,'N'))==proof['N_SHA'] and rawsha(decode(z,proof,'P'))==proof['P_SHA']
    manifest.append(proof)
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<16*1024**3 and max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())<20*1024**3 and time.monotonic()-start<14400
    if i%32==0:(out/'RUN_STATUS.json').write_text(json.dumps({'input':i,'units':units,'PID':os.getpid()}));host()
  now=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(t,now[n].detach().cpu()) for n,t in initial.items());assert units==512
  (out/'WEIGHT_MANIFEST.json').write_text(json.dumps(manifest,indent=2));(out/'COMPLETE.json').write_text(json.dumps({'units':512,'GPU_seconds':time.monotonic()-start,'numpy':np.__version__,'Source_tensors_unchanged':True,'all_source_reference_reads':0,'Source_train_eval_teacher_under_training_cudnn_flags':True,'p_implementation':'exp(log_softmax), exact217/219 FP32 KL convention','old_G_confidence_and220_training_teacher_prediction_reproduced':True,'new_normalization':'CPUfloat64 then float32, no epsilon/clip/re-scaling','CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_allocated_peak':torch.cuda.max_memory_allocated(),'GPU_reserved_peak':torch.cuda.max_memory_reserved()},indent=2));(out/'EXIT_CODE').write_text('0\n')
 except BaseException:
  (out/'FAILURE.txt').write_text(traceback.format_exc());(out/'FAILURE_COST.json').write_text(json.dumps({'units':units,'GPU_seconds':time.monotonic()-start}));(out/'EXIT_CODE').write_text('1\n');raise
 finally:ledger.close()
if __name__=='__main__':main()

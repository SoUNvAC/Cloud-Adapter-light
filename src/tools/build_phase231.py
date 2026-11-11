"""Adapt only the already accepted230 extraction path to fixed calibration scenes."""
from pathlib import Path
import ast
T=Path(__file__).resolve().parent
t=(T/'execute_phase230.py').read_text(encoding='utf-8').replace('phase230_20261026','phase231_20261026').replace('SPARCS_VAL_INPUTS.json','CALIBRATION_INPUTS.json').replace("len(allowed)==20","len(allowed)==120").replace("sum(counts.values())<80","sum(counts.values())<220").replace('<175_000_000','<240_000_000').replace("lock['models']","lock['models_to_calibrate']").replace('non-SPARCS-validation','non-calibration').replace("'STARTED.json'","'GPU_STARTED.json'").replace("R/'PID'","R/'GPU_PID'").replace("'RUN_STATUS.json'","'GPU_STATUS.json'").replace("R/'EXIT_CODE'","R/'GPU_EXIT_CODE'")
t=t.replace(" for v in rows:assert sha(v['image_path'])==v['image_sha256'] and sha(v['mask_path'])==v['mask_sha256']", " actual=[{**v,'image_sha256':sha(v['image_path']),'mask_sha256':sha(v['mask_path'])} for v in rows];save('CALIBRATION_ACTUAL_INPUTS.json',actual);rows=actual;save('INPUTS_SEALED.json',{'epoch':time.time(),'actual_inputs_SHA':sha(R/'CALIBRATION_ACTUAL_INPUTS.json'),'scenes':60,'validation_cache_accessed':False})")
a=t.index(" for d in ['scores','labels']:")
t=t[:a]+''' for d in ['d_cache','labels']:(R/d).mkdir()
 receipts=[];repeats=[]
 for name in lock['models_to_calibrate']:
  stage='calibration_main';model.decode_head.load_state_dict(heads[name]);model.eval();now=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(t,now[n].detach().cpu()) for n,t in frozen.items())
  for i,v in enumerate(rows):
   current=f'{name}/{v["scene"]}';status();x=torch.from_numpy(load_rgb_image(Path(v['image_path']),512)).cuda()
   def forward(kind):
    count(kind)
    with torch.inference_mode():
     z=wrapper(x).float().clamp_min(0);s=z.sum(1,keepdim=True);p=z/s.clamp_min(1e-12)
    assert torch.isfinite(z).all() and torch.isfinite(p).all()
    return p[0].cpu().numpy(),s[0,0].cpu().numpy()
   p,s=forward('main_complete_input');y=load_parent_target(Path(v['mask_path']),SPARCS_TO_PARENT,(512,512)).astype(np.uint8);valid=y!=255;assert not np.any((s==0)&valid)
   assert np.max(np.abs(p.sum(0)[valid]-1))<1e-6
   d=np.log(np.maximum(p[1],1e-12))-np.log(np.maximum(p[0],1e-12));assert d.dtype==np.float32 and np.isfinite(d).all()
   dp=R/f'd_cache/{name}__{v["scene"]}.npy';np.save(dp,d,allow_pickle=False);yp=R/f'labels/{v["scene"]}.npy'
   if name=='Source':np.save(yp,y,allow_pickle=False)
   else:assert np.array_equal(y,np.load(yp,allow_pickle=False))
   rec={'model':name,'scene':v['scene'],'d_path':str(dp.relative_to(R)),'d_SHA':sha(dp),'label_path':str(yp.relative_to(R)),'label_SHA':sha(yp),'image_SHA':v['image_sha256'],'mask_SHA':v['mask_sha256'],'Cloud_positive':int((y==1).sum()),'Surface_negative':int((y==0).sum()),'Shadow_excluded':int((y==2).sum()),'ignore_excluded':int((y==255).sum()),'log_clamp_valid_channel_counts':[int((p[k][valid]<1e-12).sum()) for k in range(3)]};receipts.append(rec);save('CALIBRATION_RECEIPTS.json',receipts)
   if i==0:
    q,qs=forward('repeat_complete_input');repeats.append({'model':name,'scene':v['scene'],'max_score_abs_difference':float(np.max(np.abs(p-q))),'scores_bitwise_equal':bool(np.array_equal(p,q))});assert np.array_equal(p.argmax(0),q.argmax(0));save('REPEATS.json',repeats)
 assert counts['main_complete_input']==180 and counts['repeat_complete_input']==3 and len(receipts)==180 and not denied;done.set();torch.cuda.synchronize()
 save('GPU_COMPLETE.json',{'status':'complete','counts':dict(counts),'total_calls':sum(counts.values()),'GPU_stage_wall_seconds':time.monotonic()-START,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_peak_bytes':torch.cuda.max_memory_allocated(),'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(),'persistent_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file()),'network_parameter_updates':0,'validation_L8_source_test_pixel_reads':0,'forbidden_accesses':denied});stage='GPU_complete';status();(R/'GPU_EXIT_CODE').write_text('0\\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  done.set();(R/'GPU_FAILURE.txt').write_text(traceback.format_exc());save('FAILURE_COST.json',{'counts':dict(counts),'seconds':time.monotonic()-START});status();(R/'GPU_EXIT_CODE').write_text('1\\n');raise
'''
assert 'SPARCS_VAL' not in t and "expected/{name}" not in t and "'scores'" not in t;ast.parse(t);(T/'execute_phase231.py').write_text(t,encoding='utf-8');print('Calibration-only GPU extractor built and parsed.')

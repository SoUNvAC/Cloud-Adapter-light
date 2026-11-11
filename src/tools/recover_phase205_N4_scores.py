"""Budgeted artifact repair: retain original N4 final scores; no new fit/comparison."""
import gc,hashlib,json,pathlib,sys,time,traceback,zipfile
ROOT=pathlib.Path('/home/scv/shared/phase205_20261019');sys.path.insert(0,str(ROOT/'code'))
import execute_phase205 as core
def main():
 assert core.load(ROOT/'CONTROLLER_STATUS.json')['status']=='completed_pending_controlled_delivery_local_reference_join'
 assert sum(1 for _ in (ROOT/'CALL_LEDGER.jsonl').open())==6600
 guard,thread,hooks=core.setup('N4_score_recovery');out=core.OUT
 try:
  import numpy as np,torch
  from phase119_common import catalog_image
  from phase118_common import image,coordinates
  from omnicloudmask import model_utils as mu,cloud_mask as cm
  model,_=core.build('N');model.eval()
  for p in model.parameters():p.requires_grad_(False)
  initial=core.digest(model.state_dict());plans=core.load(ROOT/'EVALUATION_INPUT_PLAN.json')['rows'];expected={r['product_id']:r for r in core.load(ROOT/'EXPECTED_SUPPORT.json')['rows']};coord=coordinates();index={};receipts=[];core.S['kind']='inference';core.S['status']='scoring_original_N4_artifact_repair'
  for row in plans:
   guard(True);pool,pid=row['pool'],row['product_id'];core.S.update(pool=pool,product_id=pid,patch_completed=0);actual=[]
   if pool=='Catalogue':
    arr,valid,_=catalog_image(row['group']);flat=np.flatnonzero(valid);yy,xx=np.unravel_index(flat,valid.shape);joint=np.ones(len(flat),bool);points=None
   else:
    points=[q for q in coord if q['product_id']==pid];arr,v,bindings=image(row['product'],'C20',points);assert bindings==row['input_bindings'];yy=np.array([int(q['row20']) for q in points]);xx=np.array([int(q['col20']) for q in points]);joint=v[yy,xx].copy();del v
   cpu=np.zeros((4,len(yy)),np.float32);grad=np.zeros(len(yy),np.float32);latest={};oldinfer=mu.infer_batch;oldstore=mu.store_results;before=core.S['calls']
   def capture(ms,batch):
    result=oldinfer(ms,batch);latest['raw']=result.detach().cpu().numpy().copy();return result
   def store(pred_batch,index_batch,pred_tracker,gradient,grad_tracker=None):
    g=gradient.detach().cpu().numpy()
    for i,(t,b,l,r) in enumerate(index_batch):
     inside=(yy>=t)&(yy<b)&(xx>=l)&(xx<r);dy=yy[inside]-t;dx=xx[inside]-l;w=g[dy,dx];cpu[:,inside]+=latest['raw'][i,:,dy,dx].T*w[None];grad[inside]+=w;actual.append([t,b,l,r]);core.S['patch_completed']=len(actual)
    latest['gradient']=grad_tracker;return oldstore(pred_batch,index_batch,pred_tracker,gradient,grad_tracker)
   mu.infer_batch=capture;mu.store_results=store
   try:official=cm.predict_from_array(arr,patch_size=1000,patch_overlap=300,batch_size=1,inference_device='cuda',mosaic_device='cpu',inference_dtype=torch.float32,export_confidence=True,softmax_output=True,custom_models=[model],model_version=4,pred_classes=4,compile_models=False,no_data_value=0,apply_no_data_mask=True)
   finally:mu.infer_batch=oldinfer;mu.store_results=oldstore
   assert actual==row['actual_patch_coordinates'] and core.S['calls']-before==len(actual);assert np.array_equal(grad,latest['gradient'].numpy()[yy,xx]);author=np.any(arr[:,yy,xx]!=0,axis=0);joint &= author&(grad>0);logits=cpu/np.maximum(grad[None],np.finfo(np.float32).tiny);raw=np.argmax(logits,0).astype(np.uint8);parent=np.array([0,1,1,2],np.uint8)[raw];soft=np.exp(logits-logits.max(0,keepdims=True));soft/=soft.sum(0,keepdims=True);probs=np.clip(soft+np.float32(.001),.001,.999)*author[None];error=float(np.max(abs(probs[:,joint]-official[:,yy[joint],xx[joint]])));assert error<=1e-5
   target=out/'scores'/pool;target.mkdir(parents=True,exist_ok=True)
   if pool=='Catalogue':
    path=target/(pid+'.npz');np.savez_compressed(path,scores=probs[3],valid=joint,flat_indices=flat,raw_class=raw,parent_class=parent);assert int(joint.sum())==expected[pid]['count']
   else:
    payload=[]
    for i,q in enumerate(points):v={k:q[k] for k in ('point_id','product_id','group_id','row10','col10','row20','col20')};v.update(valid=bool(joint[i]),shadow_confidence=float(probs[3,i]) if joint[i] else None,raw_class=int(raw[i]) if joint[i] else None,parent_class=int(parent[i]) if joint[i] else None);payload.append(v)
    path=target/pid/'UNLABELLED_POINT_SCORES.json';core.write(path,dict(policy='original_N4_auxiliary_retention_only',points=payload,labels_accessed=False));assert [q['point_id'] for i,q in enumerate(points) if joint[i]]==expected[pid]['common_image_valid_point_ids']
   index[pool+'/'+pid]=core.sha(path);receipts.append(dict(pool=pool,product_id=pid,calls=core.S['calls']-before,confidence_error=error,same_original_support=True));core.write(out/'SCENE_RECEIPTS.json',receipts);guard(True);print(pool,pid,core.S['calls'],flush=True);del arr,cpu,grad,latest,official,logits,soft,probs,raw,parent;gc.collect();torch.cuda.empty_cache()
  assert core.digest(model.state_dict())==initial;guard(True);core.S['status']='completed';core.write(out/'RUN_STATUS.json',core.S);core.write(out/'SCORES_SEALED.json',dict(scores=index,unchanged_original_N4=True,only_missing_output_recovery=True,no_new_fit_or_comparison=True,prior_main_calls=6600,all_extra_calls=core.S['calls'],cumulative_calls=6600+core.S['calls'],cumulative_stage_seconds=core.S['cumulative_stage_seconds'],source_code_sha256=core.sha(pathlib.Path(__file__)),original_six_policy_scores_unchanged=True));(out/'EXIT_CODE').write_text('0\n')
  paths=[p for p in out.rglob('*') if p.is_file() and p.suffix in ('.npz','.json')];manifest=dict(files={p.relative_to(out).as_posix():core.sha(p) for p in paths},calls=core.S['calls'],cumulative_calls=6600+core.S['calls'],cumulative_stage_seconds=core.S['cumulative_stage_seconds']);core.write(out/'DELIVERY_MANIFEST.json',manifest);archive=out/'N4_scores_delivery.zip'
  with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as z:
   z.write(out/'DELIVERY_MANIFEST.json','DELIVERY_MANIFEST.json')
   for p in paths:z.write(p,p.relative_to(out).as_posix())
  core.write(out/'DELIVERY_RECEIPT.json',dict(bytes=archive.stat().st_size,sha256=core.sha(archive),cumulative_calls=6600+core.S['calls'],cumulative_stage_seconds=core.S['cumulative_stage_seconds'],root_new_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())))
 finally:
  core.STOP.set();thread.join(timeout=30)
  for h in hooks:h.remove()
if __name__=='__main__':
 try:main()
 except BaseException:
  import os
  core.write(ROOT/'N4_score_recovery/FAILURE.json',dict(failure=traceback.format_exc(),state=core.S));(ROOT/'N4_score_recovery/EXIT_CODE').write_text('1\n');traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)

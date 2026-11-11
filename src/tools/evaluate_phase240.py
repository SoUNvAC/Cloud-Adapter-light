"""Fixed val candidates followed by sealed test selections, no score archive."""
from phase240_common import *
import copy,sys,traceback

def main():
 global budget
 stage=sys.argv[1];assert stage in ['val','test'];assert not (R/f'{stage.upper()}_STARTED.json').exists();check_bundle()
 save(f'{stage.upper()}_STARTED.json',{'epoch':time.time(),'PID':os.getpid()})
 if stage=='test':assert sha(R/'VAL_SELECTION_LOCK.json')==js(R/'VAL_SELECTION_SEAL.json')['SHA'];selection=js(R/'VAL_SELECTION_LOCK.json')['selections'];save('TEST_ACCESS_STARTED.json',{'epoch':time.time(),'selection_SHA':sha(R/'VAL_SELECTION_LOCK.json')})
 os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1'
 import numpy as np,torch,rasterio
 from rasterio.windows import Window
 from PIL import Image
 from validate_phase12_onnxruntime import load_rgb_image
 from evaluate_phase64_source_only_parent import load_parent_target,SPARCS_TO_PARENT
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.use_deterministic_algorithms(False)
 assert sha(SOURCE)==js(R/'LOCK.json')['Source_SHA256']
 budget=Budget('evaluate_'+stage);wrapper,_=build(budget);model=wrapper.segmentor.eval();model.requires_grad_(False);initial={n:t.detach().cpu().clone() for n,t in model.decode_head.state_dict().items()}
 frozen={n:t.detach().cpu().clone() for n,t in dict(list(model.named_parameters())+list(model.named_buffers())).items() if not n.startswith('decode_head.')};E=js(R/'EVAL_INPUTS.json');T=js(R/'TRAIN_INPUTS.json');plan=js(R/'NATIVE_PLAN.json');valrefs={v['name']:v for v in js(R/'L8_VAL_NATIVE_REFERENCES.json')};expected={(v['split'],v['name']):v for v in js(R/'EXPECTED_NATIVE234.json')};handles={};native_receipts=[]
 trainpaths={str(Path(v[k]).resolve()) for rows in T.values() for v in rows for k in ['image_path','mask_path']};testpaths={str(Path(v[k]).resolve()) for pools in E.values() for v in pools['test'] for k in ['image_path','mask_path']}
 def audit(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=str(Path(os.fsdecode(args[0])).resolve())
   if p in trainpaths or (stage=='val' and p in testpaths) or ('cloudsen12_high_l1c' in p and Path(p).suffix.lower() in ['.png','.npz','.tif']):raise RuntimeError('236 evaluation access denied '+p)
 sys.addaudithook(audit)
 def label(v,domain):
  if domain=='SPARCS':return load_parent_target(Path(v['mask_path']),SPARCS_TO_PARENT,(512,512)).astype(np.uint8)
  if stage=='val':
   r=valrefs[v['name']];assert sha(r['file'])==r['sha256']
   with np.load(r['file'],allow_pickle=False) as z:a=z['native'];valid=z['valid'];truth=z['truth']
  else:
   p=plan['test'][v['scene']]
   if v['scene'] not in handles:assert sha(p['path'])==p['SHA'];handles[v['scene']]=rasterio.open(p['path'])
   a=handles[v['scene']].read(1,window=Window(int(v['y']),int(v['x']),512,512),boundless=True,fill_value=0);valid=a!=0;truth=np.zeros(a.shape,np.uint8);truth[a==64]=2;truth[(a==192)|(a==255)]=1;truth[~valid]=255
  lut=np.zeros(256,np.uint8);lut[64]=1;lut[192]=2;lut[255]=3;assert np.array_equal(lut[a],np.asarray(Image.open(v['mask_path'])))
  ex=expected[stage,v['name']];assert hashlib.sha256(a.tobytes()).hexdigest()==ex['native_array_SHA'];assert hashlib.sha256(valid.tobytes()).hexdigest()==ex['valid_array_SHA']
  native_receipts.append({'stage':stage,'name':v['name'],'native_SHA':ex['native_array_SHA'],'valid_SHA':ex['valid_array_SHA'],'support':[int((truth==c).sum()) for c in range(3)],'Fill':int((truth==255).sum()),'old_PNG_exact':True})
  return truth
 def miou(cm):
  c=np.asarray(cm,np.float64);u=c.sum(0)+c.sum(1)-c.diagonal();return float((c.diagonal()[u>0]/u[u>0]).mean()*100)
 def cid(c):return json.dumps(c,sort_keys=True)
 def nk(c):return 'Source' if c['source'] else f"{c['branch']}:{c['step']}"
 expected_source=js(R/'EXPECTED_SOURCE236.json');allchoices=[];summary={};gaps=[]
 for domain in ['L8','SPARCS']:
  good=[arm for arm in ['A','B','G'] if (R/'fits'/f'{domain}_{arm}'/'COMPLETE.json').exists()]
  gaps.extend({'domain':domain,'branch':arm,'reason':'training_not_complete_no_model_substitution'} for arm in ['A','B','G'] if arm not in good)
  def candidate(arm,step,source=False,strategy=None):return dict(seed=21701,branch=arm,strategy=strategy or arm,step=step,source=source)
  if stage=='val':
   candidates=[candidate(arm,step,step is None) for arm in good for step in [None,500,1000,2000,4000]]
  else:
   candidates=[candidate('Source',None,True)]+[{k:v for k,v in s.items() if k in ['seed','branch','strategy','step','source']} for s in selection if s['domain']==domain]
   candidates += [candidate(arm,4000,strategy=arm+'_endpoint') for arm in good]
  nets={};headhash={}
  for c in candidates:
   if c['source']:continue
   p=R/'fits'/f"{domain}_{c['branch']}"/f"head_{c['step']}.pth"
   assert sha(p)==js(p.parent/f"SAVE_{c['step']}.json")['head_SHA']
   headhash[p.relative_to(R).as_posix()]=sha(p);nets[nk(c)]=torch.load(p,map_location='cpu')['state_dict']
  assert len(nets)<=(12 if stage=='val' else 6)
  totals={cid(c):np.zeros((3,3),np.int64) for c in candidates};scenes={cid(c):{} for c in candidates}
  record=(R/f'{domain}_{stage}_PER_INPUT.jsonl').open('x',buffering=1)
  def store(c,cm,predsha,v):
   ident=cid(c);totals[ident]+=cm;scenes[ident].setdefault(v['scene'],np.zeros((3,3),np.int64));scenes[ident][v['scene']]+=cm
   record.write(json.dumps({'domain':domain,'split':stage,'name':v['name'],'scene':v['scene'],'usgs_shadows':v.get('usgs_shadows'),**c,'confusion':cm.tolist(),'prediction_SHA':predsha,'source_reused_236':c['source']})+'\n')
  for index,v in enumerate(E[domain][stage]):
   assert sha(v['image_path'])==v['image_sha256'] and sha(v['mask_path'])==v['mask_sha256']
   y=label(v,domain);valid=y!=255;source=expected_source[domain][stage][v['name']];scm=np.asarray(source['confusion'],np.int64)
   assert np.array_equal(scm.sum(1),np.bincount(y[valid],minlength=3)),'Cached Source/current support drift'
   for c in candidates:
    if c['source']:store(c,scm,source['prediction_SHA'],v)
   x=torch.from_numpy(load_rgb_image(Path(v['image_path']),512)).cuda()
   for key,head in nets.items():
    model.decode_head.load_state_dict(head);budget.current={'domain':domain,'split':stage,'input':index,'name':v['name'],'network':key};budget.count('evaluation_full_input')
    with torch.inference_mode():
     prob=wrapper(x).float().clamp_min(0);prob=prob/prob.sum(1,keepdim=True).clamp_min(1e-12);pred=prob.argmax(1)[0].cpu().numpy().astype(np.uint8)
    del prob
    cm=np.bincount((3*y[valid].astype(np.int64)+pred[valid]).ravel(),minlength=9).reshape(3,3);ph=hashlib.sha256(pred.tobytes()).hexdigest()
    for c in candidates:
     if nk(c)==key:store(c,cm,ph,v)
    del pred
   del x
   if index%100==0:budget.check()
  record.close()
  table=[{'domain':domain,**json.loads(k),'confusion':v.tolist(),'pooled_mIoU':miou(v),'scenes':{g:c.tolist() for g,c in scenes[k].items()}} for k,v in totals.items()]
  save(f'{domain}_{stage}_SUMMARY.json',table);summary[domain]={'networks_per_input':len(nets),'inputs':len(E[domain][stage]),'heads':headhash,'per_input_SHA':sha(R/f'{domain}_{stage}_PER_INPUT.jsonl')}
  if stage=='val':
   for arm in good:
    pool=[t for t in table if t['branch']==arm];assert len(pool)==5
    chosen=min(pool,key=lambda c:(-c['pooled_mIoU'],0 if c['source'] else 1,c['step'] or 0));allchoices.append({k:v for k,v in chosen.items() if k!='scenes'})
  states=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(t,states[n].detach().cpu()) for n,t in frozen.items())
 if stage=='val':
  save('VAL_SELECTION_LOCK.json',{'selections':allchoices,'C_baseline_selections':js(R/'C_BASELINE_VAL_SELECTION.json'),'gaps':gaps,'val_tables':{d:sha(R/f'{d}_val_SUMMARY.json') for d in summary},'code_SHA':{p.name:sha(p) for p in (R/'code').glob('*.py')},'input_SHA':sha(R/'EVAL_INPUTS.json'),'rule':js(R/'LOCK.json')['selection']})
  save('VAL_SELECTION_SEAL.json',{'SHA':sha(R/'VAL_SELECTION_LOCK.json'),'epoch':time.time()})
 save(f'{stage.upper()}_NATIVE_ACCEPTANCE.json',native_receipts);save(f'{stage.upper()}_COMPLETE.json',{'stage':stage,'pools':summary,'gaps':gaps,'no_saved_scores':True,'all_nonhead_states_Source':True});budget.finish('passed')
 for h in handles.values():h.close()

if __name__=='__main__':
 try:main()
 except BaseException:
  save(sys.argv[1].upper()+'_FAILURE.json',{'traceback':traceback.format_exc()})
  if 'budget' in globals():budget.finish('failed')
  raise

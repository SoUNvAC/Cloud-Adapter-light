"""Build the frozen219 evaluator variant, pairing only alpha1 N/P source inference."""
from pathlib import Path
import ast
R=Path(__file__).resolve().parent
ONE=r'''  teacher_cache={};high_cache={};teacher_sha=hashlib.sha256();coupled_outputs={};dstreamhash={str(s):hashlib.sha256() for s in SEEDS};expected_val=js(ROOT/'EXPECTED_VAL.json');sample_ids=js(ROOT/'SOURCE_VAL_SAMPLE_IDS.json')
  confidence_stream=(OUT/'VAL_WEIGHT_DIAGNOSTIC.jsonl').open('x',buffering=1);state_stream=(OUT/'JOINT_COUNTS.jsonl').open('x',buffering=1)
  from phase221_weighting import normalize,rawsha
  dhead=copy.deepcopy(teacher).eval();phead=copy.deepcopy(teacher).eval();dmodels={}
  for s in SEEDS:
   v=js(ROOT/'HISTORICAL_MODEL_LOCK.json')['students'][f'{s}/D'];assert sha(v['path'])==v['SHA'];dmodels[s]=torch.load(v['path'],map_location='cpu')['state_dict']
  for head in [dhead,phead]:
   for v in head.parameters():v.requires_grad=False
  maps={'source':SOURCE_TO_PARENT,'val':SPARCS_TO_PARENT,'test':SPARCS_TO_PARENT}
  def pred(z):
   z=z.float().clamp_min(0);return (z/z.sum(1,keepdim=True).clamp_min(1e-12)).argmax(1)[0].cpu().numpy().astype(np.uint8)
  def auxiliary(head,features,kind):
   count(kind);cl,ma=head(features,None);return torch.einsum('bqc,bqhw->bchw',torch.nn.functional.softmax(cl[-1].float(),-1)[...,:-1],torch.nn.functional.interpolate(ma[-1].float(),size=(512,512),mode='bilinear',align_corners=False).sigmoid())
  def one(r,pool,kind):
   x=torch.from_numpy(load_rgb_image(Path(r['image_path']),512)).cuda();count(kind);box={};coupled=pool=='source' and alpha==1 and arm=='N';extra=coupled and r['name'] not in teacher_cache;handle=model.decode_head.register_forward_pre_hook(lambda mod,args:box.update(features=args[0])) if coupled else None
   with torch.inference_mode():
    z=wrapper(x).float();prediction=pred(z)
    if handle is not None:handle.remove()
    y=load_parent_target(Path(r['mask_path']),maps[pool],prediction.shape);valid=y!=255
    if coupled:
     assert valid.all(),'Do not change old no-ignore source support';e=expected_val['per_input'][r['name']]
     if extra:
      ts=auxiliary(teacher,box['features'],'diagnostic_teacher_head');tp=pred(ts);conf=torch.nn.functional.log_softmax(ts,1).exp().max(1).values[0].cpu().numpy();order=np.argsort(-conf.ravel(),kind='stable');g=np.full(conf.size,.5,np.float32);g[order[:131072]]=1.5;high=g.reshape(512,512)==1.5;assert rawsha(high)==e['high_mask_SHA'] and rawsha(tp)==e['teacher_prediction_SHA'];teacher_cache[r['name']]=tp;high_cache[r['name']]=high;teacher_sha.update(tp.tobytes());n,p,proof,pbits=normalize(g,tp,sample_ids[r['name']]);n=n.reshape(512,512);p=p.reshape(512,512)
      # A small integer histogram permits independent reference/teacher-class weight reconstruction.
      p_high=np.unpackbits(pbits,count=262144).reshape(512,512).astype(bool)
      jw=np.bincount(y[valid]*12+tp[valid].astype(np.int64)*4+high[valid].astype(np.int64)*2+p_high[valid].astype(np.int64),minlength=36).reshape(3,3,2,2)
      confidence_stream.write(json.dumps({'name':r['name'],'ROI':r['scene'],'sample_id':sample_ids[r['name']],'high_mask_SHA':rawsha(high),'teacher_prediction_SHA':rawsha(tp),'weight_proof':proof,'weight_count_axes':['reference','teacher_prediction','G_high','P_high'],'weight_counts':jw.tolist(),'interpretation':'source-val diagnostic weights, not actual replay-training gradients'})+'\n')
     tp=teacher_cache[r['name']];high=high_cache[r['name']];ds=auxiliary(dhead,box['features'],'diagnostic_D_head');dp=pred(ds);ps=auxiliary(phead,box['features'],'candidate_shared_P_head');pp=pred(ps);dstreamhash[str(seed)].update(dp.tobytes());assert rawsha(dp)==e['students'][f'{seed}/D']['prediction_SHA'];dcm=np.bincount(y[valid]*3+dp[valid],minlength=9).reshape(3,3);assert dcm.tolist()==e['students'][f'{seed}/D']['confusion']
     if index==1:
      model.decode_head.load_state_dict(heads[seed,'P',1]);count('P_shared_path_gate_full');assert torch.equal(wrapper(x).float(),ps);model.decode_head.load_state_dict(heads[seed,'N',1]);save(f'P_SHARED_PATH_GATE_{seed}.json',{'first_input':r['name'],'full_semantics_exact':True,'prediction_SHA':rawsha(pp),'additional_calls':1})
     code=(~high[valid]).astype(np.int64)*243+y[valid]*81+tp[valid].astype(np.int64)*27+dp[valid].astype(np.int64)*9+prediction[valid].astype(np.int64)*3+pp[valid].astype(np.int64);j=np.bincount(code,minlength=486).reshape(2,3,3,3,3,3);state_stream.write(json.dumps({'seed':seed,'name':r['name'],'ROI':r['scene'],'image_SHA':r['image_sha256'],'mask_SHA':r['mask_sha256'],'valid_pixels':int(valid.sum()),'teacher_prediction_SHA':rawsha(tp),'high_mask_SHA':rawsha(high),'student_prediction_SHA':{'D':rawsha(dp),'N':rawsha(prediction),'P':rawsha(pp)},'J':j.tolist()})+'\n');coupled_outputs.clear();coupled_outputs.update({'P_prediction':pp,'P_confusion':np.bincount(3*y[valid]+pp[valid],minlength=9).reshape(3,3)})
   return np.bincount(3*y[valid]+prediction[valid],minlength=9).reshape(3,3),prediction
'''
EVAL=r'''  def evaluate(s,a,f,pool):
   nonlocal stage,seed,arm,alpha,index
   stage=pool;seed=s;arm=a;alpha=f;sethead(heads[s,a,f]);gs=collections.defaultdict(lambda:np.zeros((3,3),np.int64));total=np.zeros((3,3),np.int64);per=[];ph=hashlib.sha256();paired=(pool=='source' and a=='N' and f==1)
   if paired:
    dhead.load_state_dict(dmodels[s]);phead.load_state_dict(heads[s,'P',1]);pg=collections.defaultdict(lambda:np.zeros((3,3),np.int64));pt=np.zeros((3,3),np.int64);pper=[];phash=hashlib.sha256()
   for index,r in enumerate(support[pool],1):
    c,p=one(r,pool,'candidate_full_segmentor');total+=c;gs[r['scene']]+=c;ph.update(p.tobytes());per.append({'name':r['name'],'scene':r['scene'],'confusion':c.tolist()})
    if paired:
     pc=coupled_outputs['P_confusion'];pp=coupled_outputs['P_prediction'];pt+=pc;pg[r['scene']]+=pc;phash.update(pp.tobytes());pper.append({'name':r['name'],'scene':r['scene'],'confusion':pc.tolist()})
   result={'seed':s,'arm':a,'alpha':f,'pool':pool,'confusion':total.tolist(),'per_group':{g:c.tolist() for g,c in sorted(gs.items())},'per_input':per,'prediction_SHA':ph.hexdigest()};save(f'{s}_{a}_a{f}__{pool}.json',result)
   if paired:
    save(f'{s}_P_a1__source.json',{'seed':s,'arm':'P','alpha':1,'pool':'source','confusion':pt.tolist(),'per_group':{g:c.tolist() for g,c in sorted(pg.items())},'per_input':pper,'prediction_SHA':phash.hexdigest(),'shared_features_with_N':True});assert dstreamhash[str(s)].hexdigest()==expected_val['student_prediction_SHA'][f'{s}/D']
   status();return result
'''
def main():
 t=(R/'evaluate_phase219.py').read_text(encoding='utf-8').replace('phase219_20261023','phase221_20261024').replace('80000','85000').replace('from controller_phase219 import host','from controller_phase221 import host').replace("['G','S']","['N','P']")
 t=t.replace("sys.path[:0]=[str(SRC),str(SRC/'tools')]","sys.path[:0]=[str(ROOT/'code'),str(SRC),str(SRC/'tools')]")
 start=t.index('  teacher_cache={};');end=t.index('  parity=[]',start);t=t[:start]+ONE+t[end:]
 start=t.index('  def evaluate(');end=t.index('  def score(',start);t=t[:start]+EVAL+"  for s in SEEDS:\n   for a in ['N','P']:\n    for f in AS[1:]:\n     if not (a=='P' and f==1):evaluate(s,a,f,'source')\n"+t[end:]
 t=t.replace("{'Source_parity_full_segmentor':3,'candidate_full_segmentor':8880,'diagnostic_teacher_head':535}","{'Source_parity_full_segmentor':3,'candidate_full_segmentor':7810,'candidate_shared_P_head':1070,'diagnostic_teacher_head':535,'diagnostic_D_head':1070,'P_shared_path_gate_full':2}")
 ast.parse(t);(R/'evaluate_phase221.py').write_text(t,encoding='utf-8')
if __name__=='__main__':main()

"""Generate fixed L8 evaluator, with two paired endpoint heads sharing source features."""
from pathlib import Path
import ast
R=Path(__file__).resolve().parents[2]
def main():
 t=(R/'src/tools/evaluate_phase217.py').read_text(encoding='utf-8').replace('import argparse,collections','import copy,torch_placeholder,collections' if False else 'import argparse,copy,collections')
 t=t.replace('phase217_20261023','phase222_20261024').replace('controller_phase217','controller_phase222').replace('SPARCS_TO_PARENT','L8_TO_PARENT').replace('<60000','<120000').replace('<14400','<21600').replace("ROOT/'PHASE216_EVAL_INPUT_LOCK.json'","ROOT/'INPUT_SUPPORT_LOCK.json'").replace('[535,10,10]','[535,963,2167]')
 t=t.replace("caches={};expected=js(ROOT/'SOURCE_CACHE_SHAS.json')","caches={}")
 t=t.replace("assert sha(OLD/'evaluation/Source__source.json')==expected['source'];caches['source']=js(OLD/'evaluation/Source__source.json')","caches['source']=js(ROOT/'EXPECTED_Source__source.json')")
 t=t.replace("path=OLD/f'evaluation/Source__{pool}.json';assert sha(path)==expected[pool];caches[pool]=js(path)","path=ROOT/f'EXPECTED_Source__{pool}.json';caches[pool]=js(path)")
 t=t.replace("        ","        ")
 marker="  def one(r,pool,kind):"
 start=t.index(marker);end=t.index('  parity=[]',start)
 t=t[:start]+'''  teacher=copy.deepcopy(model.decode_head).eval();partner=copy.deepcopy(model.decode_head).eval()
  for h in [teacher,partner]:
   for p in h.parameters():p.requires_grad=False
  captured={};teacher_cache={};joint_stream=(OUT/'JOINT_COUNTS.jsonl').open('x',buffering=1);joint_rows=0;shared_gates=[]
  hook=model.decode_head.register_forward_pre_hook(lambda module,args:captured.update(features=args[0]))
  def rawsha(x):return hashlib.sha256(x.tobytes()).hexdigest()
  def predict(z):
   z=z.float().clamp_min(0);return (z/z.sum(1,keepdim=True).clamp_min(1e-12)).argmax(1)[0].cpu().numpy().astype(np.uint8)
  def semantic(h,features,kind):
   import torch.nn.functional as F
   count(kind);c,m=h(features,None)
   return torch.einsum('bqc,bqhw->bchw',c[-1].float().softmax(-1)[...,:-1],F.interpolate(m[-1].float(),size=(512,512),mode='bilinear',align_corners=False).sigmoid())
  def one(r,pool,kind):
   nonlocal joint_rows
   x=torch.from_numpy(load_rgb_image(Path(r['image_path']),512)).cuda();count(kind)
   with torch.inference_mode():
    z=wrapper(x);features=captured.pop('features');pred=predict(z)
    paired=pool=='source' and arm=='C' and alpha==1 and kind=='candidate_full_segmentor'
    if paired:
     partner.load_state_dict(heads[seed,'D',1]);partner.eval();dsem=semantic(partner,features,'candidate_shared_D_head');dp=predict(dsem)
     if seed==21701:
      ts=semantic(teacher,features,'diagnostic_teacher_head');tp=predict(ts);teacher_cache[r['name']]=tp
     else:tp=teacher_cache[r['name']]
     if index==1:
      sethead(heads[seed,'D',1]);count('shared_D_gate_full');official=wrapper(x);captured.clear();assert torch.equal(official,dsem) and np.array_equal(predict(official),dp);sethead(heads[seed,'C',1]);shared_gates.append({'seed':seed,'name':r['name'],'semantic_bitwise_equal':True})
   y=load_parent_target(Path(r['mask_path']),maps[pool],pred.shape);valid=y!=255;c=np.bincount(3*y[valid]+pred[valid],minlength=9).reshape(3,3)
   if paired:
    dc=np.bincount(3*y[valid]+dp[valid],minlength=9).reshape(3,3);tc=np.bincount(3*y[valid]+tp[valid],minlength=9).reshape(3,3)
    old=next(v for v in caches['source']['per_input'] if v['name']==r['name']);assert tc.tolist()==old['confusion']
    J=np.bincount(27*y[valid]+9*tp[valid]+3*pred[valid]+dp[valid],minlength=81).reshape(3,3,3,3)
    joint_stream.write(json.dumps({'seed':seed,'name':r['name'],'ROI':r['scene'],'image_SHA':r['image_sha256'],'mask_SHA':r['mask_sha256'],'valid_pixels':int(valid.sum()),'reference_teacher_C_D':J.tolist(),'teacher_prediction_SHA':rawsha(tp),'C_prediction_SHA':rawsha(pred),'D_prediction_SHA':rawsha(dp)})+'\\n');joint_rows+=1
    pair_data[r['name']]=(dc,dp)
   return c,pred
''' +t[end:]
 # Use int64 labels to avoid uint8 mixed-radix overflow for four-axis counts.
 t=t.replace('27*y[valid]+9*tp[valid]+3*pred[valid]+dp[valid]','27*y[valid].astype(np.int64)+9*tp[valid].astype(np.int64)+3*pred[valid].astype(np.int64)+dp[valid]')
 t=t.replace('  def evaluate(s,a,f,pool):','  pair_data={}\n  def evaluate(s,a,f,pool):')
 t=t.replace("'scene':r['scene'],'confusion':c.tolist()}","'scene':r['scene'],'confusion':c.tolist(),'prediction_SHA':rawsha(p)}")
 old="  for s in SEEDS:\n   for a in ['C','D']:\n    for f in AS[1:]:evaluate(s,a,f,'source')"
 new="""  for s in SEEDS:
   for a in ['C','D']:
    for f in AS[1:]:
     if a=='D' and f==1:continue
     evaluate(s,a,f,'source')
     if a=='C' and f==1:
      gs=collections.defaultdict(lambda:np.zeros((3,3),np.int64));total=np.zeros((3,3),np.int64);per=[];ph=hashlib.sha256()
      for r in support['source']:
       c,p=pair_data.pop(r['name']);gs[r['scene']]+=c;total+=c;ph.update(p.tobytes());per.append({'name':r['name'],'scene':r['scene'],'confusion':c.tolist(),'prediction_SHA':rawsha(p)})
      save(f'{s}_D_a1__source.json',{'seed':s,'arm':'D','alpha':1,'pool':'source','confusion':total.tolist(),'per_group':{g:c.tolist() for g,c in sorted(gs.items())},'per_input':per,'prediction_SHA':ph.hexdigest(),'path':'same frozen features as C endpoint; shared/full semantic gate passed'})
  assert joint_rows==1070 and not pair_data;save('SHARED_PATH_GATES.json',shared_gates)
"""
 assert old in t;t=t.replace(old,new)
 t=t.replace("assert counts=={'Source_parity_full_segmentor':3,'candidate_full_segmentor':8880};done.set()","assert counts=={'Source_parity_full_segmentor':3,'candidate_full_segmentor':57570,'candidate_shared_D_head':1070,'diagnostic_teacher_head':535,'shared_D_gate_full':2};joint_stream.close();done.set()")
 ast.parse(t);(R/'src/tools/evaluate_phase222.py').write_text(t,encoding='utf-8')
if __name__=='__main__':main()

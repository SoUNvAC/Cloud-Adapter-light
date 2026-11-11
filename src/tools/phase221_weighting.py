"""Teacher-class normalization of frozen G; local class-conditional PCG64 only."""
import numpy as np,hashlib
def rawsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def normalize(g,t,sample_id):
 g=np.asarray(g,dtype=np.float32).ravel();t=np.asarray(t,dtype=np.uint8).ravel();assert g.shape==t.shape and len(g)>0 and np.isin(g,[.5,1.5]).all() and (t<3).all()
 n=np.empty_like(g);p=np.empty_like(g);p_high=np.empty(g.size,bool);proof=[]
 for c in range(3):
  ids=np.flatnonzero(t==c).astype('<i8');seed=int.from_bytes(hashlib.sha256(('phase221|'+str(sample_id)+'|'+str(c)).encode()).digest()[:8],'big')
  if not len(ids):proof.append({'class':c,'support':0,'mean_g':None,'N_mean':None,'P_mean':None,'seed':seed,'permutation_SHA':None});continue
  mean=float(g[ids].astype(np.float64).mean());assert .5<=mean<=1.5;n[ids]=(g[ids].astype(np.float64)/mean).astype(np.float32);perm=np.random.Generator(np.random.PCG64(seed)).permutation(len(ids)).astype('<i8');p[ids]=n[ids][perm];p_high[ids]=(g[ids][perm]==1.5)
  assert abs(n[ids].mean(dtype=np.float64)-1)<=1e-6 and abs(p[ids].mean(dtype=np.float64)-1)<=1e-6 and np.array_equal(np.sort(n[ids]),np.sort(p[ids]))
  proof.append({'class':c,'support':len(ids),'mean_g':mean,'N_mean':float(n[ids].mean(dtype=np.float64)),'P_mean':float(p[ids].mean(dtype=np.float64)),'N_sum':float(n[ids].sum(dtype=np.float64)),'P_sum':float(p[ids].sum(dtype=np.float64)),'g_high_pixels':int((g[ids]==1.5).sum()),'unique_N_weights':np.unique(n[ids]).tolist(),'seed':seed,'class_indices_SHA':rawsha(ids),'permutation_SHA':rawsha(perm),'permuted_global_indices_SHA':rawsha(ids[perm])})
 assert np.isfinite(n).all() and np.isfinite(p).all() and n.min()>=1/3 and n.max()<=3 and p.min()>=1/3 and p.max()<=3 and abs(n.mean(dtype=np.float64)-1)<=1e-6 and abs(p.mean(dtype=np.float64)-1)<=1e-6
 return n,p,{'sample_id':str(sample_id),'numpy':np.__version__,'teacher_prediction_SHA':rawsha(t),'G_SHA':rawsha(g),'N_SHA':rawsha(n),'P_SHA':rawsha(p),'N_global_mean':float(n.mean(dtype=np.float64)),'P_global_mean':float(p.mean(dtype=np.float64)),'classes':proof},np.packbits(p_high)
def decode(z,proof,arm):
 t=z['teacher_prediction'].reshape(-1);high=np.unpackbits(z['G_bits' if arm=='N' else 'P_bits'],count=len(t));lookup=np.ones((3,2),np.float32)
 for r in proof['classes']:
  if r['support']:lookup[r['class']]=np.array([.5/r['mean_g'],1.5/r['mean_g']],np.float64).astype(np.float32)
 return lookup[t,high].reshape(1,512,512)
def boundary_tests():
 cases=[([.5,1.5,.5,1.5],[0,0,0,0]),([.5,1.5,.5,1.5],[0,1,1,1]),([.5,.5,1.5,1.5],[0,0,1,1]),([.5],[2])];proof=[]
 for i,(g,t) in enumerate(cases):
  before=np.random.get_state();n,p,r,_=normalize(g,t,'boundary'+str(i));after=np.random.get_state();assert np.array_equal(before[1],after[1]) and before[2:]==after[2:]
  for c in np.unique(t):assert abs(n[np.array(t)==c].mean(dtype=np.float64)-1)<=1e-6
  if i in [2,3]:assert np.all(n==1) and np.array_equal(n,p)
  proof.append(r)
 return {'status':'passed','empty_singleton_constant_g_single_class':proof,'global_RNG_unchanged':True,'new_model_calls':0}
def verify_small_formula(torch):
 a=np.array([[1,.5,.1],[.8,.8,.1],[.2,.5,.9],[2,.2,.1],[.1,.4,.3],[.1,1,.3],[.8,.2,.3],[.3,.2,.6]],float);b=a+np.array([[.1,-.1,.2]])
 with torch.no_grad():
  aa=torch.tensor(a,device='cuda',dtype=torch.float32);bb=torch.tensor(b,device='cuda',dtype=torch.float32);lp=torch.log_softmax(aa,1);lq=torch.log_softmax(bb,1);k=(lp.exp()*(lp-lq)).sum(1);t=aa.argmax(1).cpu().numpy().astype(np.uint8);c=lp.exp().max(1).values.cpu().numpy();rank=np.argsort(-c,kind='stable');g=np.full(8,.5,np.float32);g[rank[:4]]=1.5;n,p,proof,_=normalize(g,t,'small_formula');pp=np.exp(a-a.max(1,keepdims=True));pp/=pp.sum(1,keepdims=True);qq=np.exp(b-b.max(1,keepdims=True));qq/=qq.sum(1,keepdims=True);explicit=(pp*np.log(pp/qq)).sum(1);r={'source_semantic':a.tolist(),'student_semantic':b.tolist(),'teacher_predictions':t.tolist(),'G':g.tolist(),'weights':{'N':n.tolist(),'P':p.tolist()},'weighted_KL_GPU':{},'network_calls':0,'weight_proof':proof,'CPU_explicit_pixel_KL':explicit.tolist()}
  for arm,w in [('N',n),('P',p)]:
   v=float((torch.from_numpy(w).cuda()*k).mean());assert abs(v-float(np.mean(w*explicit)))<=1e-6;r['weighted_KL_GPU'][arm]=v
  return r

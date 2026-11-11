"""Fixed label-free weighting; every random operation uses a local PCG64 generator."""
import hashlib,numpy as np
def rawsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def build(confidence,sample_id):
 c=np.asarray(confidence,dtype=np.float32).reshape(-1);assert c.size==512*512 and np.isfinite(c).all()
 order=np.argsort(-c,kind='stable');g=np.full(c.size,.5,np.float32);g[order[:c.size//2]]=1.5
 seed=int.from_bytes(hashlib.sha256(('phase219|'+str(sample_id)).encode()).digest()[:8],'big');ix=np.random.Generator(np.random.PCG64(seed)).permutation(c.size).astype('<i8');s=g[ix]
 assert np.count_nonzero(g==1.5)==np.count_nonzero(s==1.5)==131072 and g.mean()==s.mean()==1
 return g,s,{'sample_id':str(sample_id),'numpy_version':np.__version__,'local_PCG64_seed':seed,'confidence_raw_SHA':rawsha(c),'G_SHA':rawsha(g.astype('<f4')),'S_SHA':rawsha(s.astype('<f4')),'permutation_SHA':rawsha(ix),'stable_rank_SHA':rawsha(order.astype('<i8')),'high_pixels':131072,'low_pixels':131072,'mean_weight':1}
def decode(bits):return np.unpackbits(bits,count=512*512).astype(np.float32).reshape(1,512,512)+.5
def verify_small_formula(torch):
 import math
 a=[[1,.5,.1],[.8,.8,.1],[.2,.5,.9],[2,.2,.1],[.1,.4,.3],[.1,1,.3],[.8,.2,.3],[.3,.2,.6]];b=[[.9,.4,.2],[.7,.9,.2],[.3,.4,.7],[1,.5,.3],[.3,.3,.2],[.2,.7,.6],[.7,.4,.2],[.5,.3,.3]]
 with torch.no_grad():
  lp=torch.log_softmax(torch.tensor(a,device='cuda',dtype=torch.float32),1);lq=torch.log_softmax(torch.tensor(b,device='cuda',dtype=torch.float32),1);p=lp.exp();k=(p*(lp-lq)).sum(1);c=p.max(1).values.cpu().numpy();order=np.argsort(-c,kind='stable');g=np.full(8,.5,np.float32);g[order[:4]]=1.5;ix=np.random.Generator(np.random.PCG64(219)).permutation(8);s=g[ix];cpu=[]
  for x,y in zip(a,b):
   pp=[math.exp(v-max(x)) for v in x];qq=[math.exp(v-max(y)) for v in y];pp=[v/sum(pp) for v in pp];qq=[v/sum(qq) for v in qq];cpu.append(sum(v*math.log(v/qq[i]) for i,v in enumerate(pp)))
  result={'source_semantic':a,'student_semantic':b,'unweighted_KL_GPU':float(k.mean()),'pixel_KL_GPU':k.cpu().tolist(),'weights':{'G':g.tolist(),'S':s.tolist()},'weighted_KL_GPU':{},'network_calls':0}
  for arm,w in [('G',g),('S',s)]:
   v=float((torch.tensor(w,device='cuda')*k).mean());assert abs(v-sum(x*y for x,y in zip(cpu,w))/8)<1e-6;result['weighted_KL_GPU'][arm]=v
  return result

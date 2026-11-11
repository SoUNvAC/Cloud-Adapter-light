"""Read the frozen CSV by official sample_id; never refit, rank or permute."""
from pathlib import Path
import csv,json,hashlib,collections,math
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(folder,sequence):
 folder=Path(folder);manifest=json.loads((folder/'SHA256.json').read_text(encoding='utf-8-sig'))
 for n,h in manifest.items():assert sha(folder/n)==h,n
 rows=list(csv.DictReader((folder/'REPLAY_WEIGHTS.csv').open(encoding='utf-8-sig')));assert len(rows)==256;by={v['sample_id']:v for v in rows};assert len(by)==256;counts=collections.Counter(str(v['source_sample_id']) for v in sequence);assert set(by)==set(counts) and sum(counts.values())==4000;proof={}
 for sid,v in by.items():assert int(v['repeat_count'])==counts[sid] and int(v['repeat_count']) in [15,16] and v['Q_weight_source_id'] in by and int(by[v['Q_weight_source_id']]['repeat_count'])==int(v['repeat_count']) and float(v['Q_weight'])==float(by[v['Q_weight_source_id']]['R_weight'])
 for repeat in [15,16]:
  rr=[v for v in rows if int(v['repeat_count'])==repeat];assert len({v['Q_weight_source_id'] for v in rr})==len(rr) and collections.Counter(v['R_weight'] for v in rr)==collections.Counter(v['Q_weight'] for v in rr)
 for arm in ['R','Q']:
  total=math.fsum(counts[sid]*float(v[arm+'_weight']) for sid,v in by.items());assert abs(total/4000-1)<1e-12;proof[arm]={'weighted4000_sum':total,'weighted4000_mean':total/4000,'stratum_histograms':{str(k):dict(collections.Counter(v[arm+'_weight'] for v in rows if int(v['repeat_count'])==k)) for k in [15,16]}}
 return by,{'status':'passed','CSV_SHA':sha(folder/'REPLAY_WEIGHTS.csv'),'files_SHA':manifest,'sample_ids':256,'steps':4000,'weights':proof,'no_rank_permutation_or_fit':True}
def verify_formula(torch):
 import numpy as np
 p=torch.tensor([[[.2,.3],[.5,.1]],[[.3,.2],[.2,.6]],[[.5,.5],[.3,.3]]],dtype=torch.float64);q=torch.tensor([[[.3,.2],[.4,.2]],[[.2,.4],[.3,.5]],[[.5,.4],[.3,.3]]],dtype=torch.float64);pixel=(p*(p.log()-q.log())).sum(0);r=[]
 for w in [2.000666888962988,.6668889629876625]:
  scalar=torch.tensor(w,dtype=torch.float32,device='cuda',requires_grad=False);gpu=scalar*pixel.float().cuda().mean();explicit=float(np.float32(w))*sum(float(p[c,y,x])*np.log(float(p[c,y,x])/float(q[c,y,x])) for c in range(3) for y in range(2) for x in range(2))/4;assert abs(float(gpu)-explicit)<=1e-6 and not scalar.requires_grad;r.append({'weight_CSV':w,'weight_used_float32':float(scalar),'CPU_explicit':explicit,'GPU':float(gpu)})
 return {'status':'passed','tolerance':1e-6,'checks':r,'model_calls':0}

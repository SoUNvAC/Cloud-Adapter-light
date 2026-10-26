"""Generated-data arithmetic checks only, permitted before the target-reference barrier."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from phase317_common import *
from analyze_phase317 import exact_ap
from analyze_phase304 import ap
def main():
 resource_check();assert 'torch' not in sys.modules and not (ROOT/'SYNTHETIC_ACCEPTANCE.json').exists();start=time.monotonic();rng=np.random.Generator(np.random.PCG64(31700));cases=[]
 for n,constant in [(27,False),(1000107,False),(1000107,True)]:
  y=rng.integers(0,2,size=n).astype(bool);s=np.ones(n,np.float32) if constant else rng.integers(0,17,size=n).astype(np.float32)/16
  a=exact_ap(y,s);b=ap(y,s);assert abs(a-b)<1e-12;cases.append(dict(generated_samples=n,constant_scores=constant,exact_AP=a,independent_AP=b))
 assert exact_ap(np.ones(8,bool),np.ones(8,np.float32)) is None and exact_ap(np.zeros(8,bool),np.ones(8,np.float32)) is None
 write(ROOT/'SYNTHETIC_ACCEPTANCE.json',dict(status='passed',cases=cases,generated_data_only=True,target_references_opened=False,new_neural_calls=0,seconds=time.monotonic()-start,resources=resource_check()));print('Phase317 generated-data AP checks passed, including million-sample tie-boundary; no target references')
if __name__=='__main__':main()

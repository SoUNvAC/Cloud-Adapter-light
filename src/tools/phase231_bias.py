"""Only the preregistered equal-scene BCE bias derivative bisection."""
import numpy as np
def sigmoid(x):return np.exp(-np.logaddexp(0.,-x))
def gradient(chunks,b):return float(np.mean([np.mean(sigmoid(d+np.float64(b))-y,dtype=np.float64) for d,y in chunks],dtype=np.float64))
def loss(chunks,b):return float(np.mean([np.mean(np.logaddexp(0.,d+np.float64(b))-y*(d+np.float64(b)),dtype=np.float64) for d,y in chunks],dtype=np.float64))
def solve(chunks):
 assert chunks and all(d.dtype==y.dtype==np.float64 and d.shape==y.shape and d.size and np.isfinite(d).all() for d,y in chunks)
 P=sum(int(y.sum()) for d,y in chunks);N=sum(len(y)-int(y.sum()) for d,y in chunks);assert P>0 and N>0
 lo,hi=-5.,5.;gl=gradient(chunks,lo);gh=gradient(chunks,hi);trace=[]
 if gl>=0:b=lo;boundary=True
 elif gh<=0:b=hi;boundary=True
 else:
  for i in range(60):
   mid=(lo+hi)/2.;gm=gradient(chunks,mid);assert np.isfinite(gm);trace.append({'iteration':i+1,'lo':lo,'hi':hi,'mid':mid,'g':gm})
   if gm>0:hi=mid
   else:lo=mid
  b=(lo+hi)/2.;boundary=False
 return {'b':b,'L0':loss(chunks,0.),'Lb':loss(chunks,b),'g_b':gradient(chunks,b),'g_minus5':gl,'g_plus5':gh,'boundary':boundary,'iterations':len(trace),'trace':trace,'Cloud_positive':P,'Surface_negative':N,'scenes':len(chunks)}
def handcheck():
 x=[(np.array([0.,0.,0.,0.]),np.array([1.,1.,1.,0.]))];r=solve(x);assert abs(r['b']-np.log(3))<1e-12 and r['Lb']<=r['L0'] and abs(r['g_b'])<1e-14
 x=[(np.array([20.,20.]),np.array([1.,0.]))];r=solve(x);assert r['b']==-5 and r['boundary']
 x=[(np.array([-20.,-20.]),np.array([1.,0.]))];r=solve(x);assert r['b']==5 and r['boundary']
 return {'analytic_log3':True,'both_boundaries':True,'passed':True}
if __name__=='__main__':print(handcheck())

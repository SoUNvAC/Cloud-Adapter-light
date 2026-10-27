"""Image-only fixed sun support; no reference imports or truth pixels."""
from phase319_common import *
import math
import numpy as np
from scipy.signal import fftconvolve
def kernel(M,azimuth,elevation,size=512):
 M=np.asarray(M,float);lo=200/math.tan(math.radians(elevation));hi=12000/math.tan(math.radians(elevation))
 tol=max(np.linalg.norm(M@np.array(v)) for v in [(.5,.5),(.5,-.5)])
 radius=hi+tol;inverse=np.linalg.inv(M);bx=int(math.ceil(radius*np.linalg.norm(inverse[0])))+1;by=int(math.ceil(radius*np.linalg.norm(inverse[1])))+1
 assert max(bx,by)<20000,'theoretical kernel would exceed safe bounded CPU allocation; preserve gap, no range reduction'
 dx=np.arange(-bx,bx+1,dtype=np.int32);keepx=[];keepy=[];full=0;extrema=[0,0,0,0]
 for dy in range(-by,by+1):
  east=M[0,0]*dx+M[0,1]*dy;north=M[1,0]*dx+M[1,1]*dy;dist=np.hypot(east,north);angle=np.degrees(np.arctan2(east,north));eligible=(dist>0)&(dist>=max(0,lo-tol))&(dist<=hi+tol)&(np.abs((angle-azimuth+180)%360-180)<=5+1e-12)
  xx=dx[eligible];full+=len(xx)
  if len(xx):extrema=[min(extrema[0],int(xx.min())),max(extrema[1],int(xx.max())),min(extrema[2],dy),max(extrema[3],dy)]
  if abs(dy)<size:
   xx=xx[np.abs(xx)<size]
   if len(xx):keepx.append(xx);keepy.append(np.full(len(xx),dy,np.int32))
 xx=np.concatenate(keepx) if keepx else np.array([],np.int32);yy=np.concatenate(keepy) if keepy else np.array([],np.int32)
 K=np.zeros((2*size-1,2*size-1),np.float64);K[yy+size-1,xx+size-1]=1
 return K,xx,yy,dict(azimuth_deg=float(azimuth%360),elevation_deg=elevation,minimum_distance_m=lo,maximum_distance_m=hi,radial_half_diagonal_tolerance_m=tol,angle_halfwidth_deg=5,full_integer_offsets=full,in_field_possible_offsets=len(xx),full_offset_extrema=extrema,zero_displacement_excluded=True)
def count(domain,K):
 values=fftconvolve(domain.astype(np.float64),K[::-1,::-1],mode='same');rounded=np.rint(values);error=float(np.abs(values-rounded).max());assert error<1e-6 and rounded.min()>=0
 return rounded.astype(np.int64),error
def direct(domain,xx,yy,r,c):
 x=c+xx;y=r+yy;inside=(x>=0)&(x<domain.shape[1])&(y>=0)&(y<domain.shape[0]);return int(domain[y[inside],x[inside]].sum())
def synthetic():
 tests=[];M=np.array([[30,0],[0,-30]],float)
 for az,dr,dc in [(0,-10,0),(90,0,10),(180,10,0),(270,0,-10)]:
  K,xx,yy,p=kernel(M,az,45,64);cloud=np.zeros((64,64),bool);cloud[32+dr,32+dc]=True;v,err=count(cloud,K);assert v[32,32]==1
  kr,_,_,_=kernel(M,az+180,45,64);rev,_=count(cloud,kr);assert rev[32,32]==0
  assert direct(cloud,xx,yy,32,32)==v[32,32];tests.append(dict(test='sun cardinal shadow-to-cloud direction',azimuth=az,passed=True,FFT_error=err))
 K,x,y,p=kernel(M,90,45,64);valid=np.ones((64,64),bool);counts,_=count(valid,K);assert (counts<p['full_integer_offsets']).all();assert not count(np.zeros_like(valid),K)[0].any();tests.append(dict(test='edge unknown, full out-of-field kernel, zero cloud',passed=True))
 # The same 300m physical displacement has 10 native pixels but 5.12 output pixels after1000->512.
 native=30;output=30*1000/512;assert abs(300/output-5.12)<1e-12
 angle=np.radians(30);rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);N=rot@M;v=N@np.array([10,0]);az=np.degrees(np.arctan2(v[0],v[1]));K,x,y,p=kernel(N,az,45,64);assert np.any((x==10)&(y==0));assert abs(np.linalg.norm(v)-300)<1e-10
 K2,x2,y2,_=kernel(np.array([[output,0],[0,-output]]),90,45,64);assert np.any((x2==5)&(y2==0));assert not (np.zeros_like(valid)&(count(cloud,K2)[0]>0)).any()
 tests.append(dict(test='meters, negative affine Y, native/output scale, grid rotation, zero Shadow empty selection',passed=True,native_pixel_m=native,output_pixel_m=output,zero_Shadow_count=0));return tests
def main():
 guard();assert not (ROOT/'SELECTION_LOCK.json').exists();pre=load(ROOT/'PREREGISTRATION.json');meta=load(ROOT/'GEOMETRY_INPUTS.json');write(ROOT/'SYNTHETIC_QA.json',dict(status='passed',tests=synthetic(),uses_real_labels=False));(ROOT/'masks').mkdir();(ROOT/'kernels').mkdir();sealed=[];totals={p:0 for p in POOLS};qachecks=[]
 teacherlocks={p:{r['name']:r for r in load(PRIOR/p/'PSEUDO_LOCK.json')['rows']} for p in POOLS}
 for rec in meta:
  guard();pool,n=rec['pool'],rec['name'];entry=teacherlocks[pool][n];path=Path(entry['file']);assert pin(path,'frozen317 teacher score/pred/pseudo/valid')['sha256']==entry['sha256']
  with np.load(path) as z:s=z['scores'];pred=z['pred'];pseudo=z['pseudo'];valid=z['valid']
  den=s.sum(0);q=np.divide(s,den[None],out=np.zeros_like(s),where=den[None]>1e-12);assert np.array_equal(s.argmax(0),pred)
  domain=valid&(den>1e-12);ash=domain&(pred==2)&(q[2]>=.9);cloud=domain&(pred==1);assert np.array_equal(ash,pseudo==2);totals[pool]+=int(ash.sum());rec['A_Sh']=int(ash.sum());rec['teacher_sha256']=entry['sha256']
  sun=np.zeros_like(ash);reverse=sun.copy();conf=sun.copy();complete=sun.copy();reversecomplete=sun.copy();qa=[]
  if rec['status']=='qualified':
   for label,az in [('sun',rec['azimuth_deg']),('reverse',rec['azimuth_deg']+180)]:
    K,xx,yy,par=kernel(rec['gridcheck']['matrix_east_north_per_output_pixel'],az,rec['elevation_deg']);cc,err=count(cloud,K);vv,ev=count(domain,K);support=ash&(cc>0);covered=ash&(vv==par['full_integer_offsets']);assert not np.any(vv>par['full_integer_offsets'])
    indices=np.flatnonzero(ash);sample=np.unique(np.concatenate([indices[:32],indices[-32:],np.array([0,256*512+256,512*512-1])]))
    for idx in sample:
     r,c=divmod(int(idx),512);assert direct(cloud,xx,yy,r,c)==cc[r,c] and direct(domain,xx,yy,r,c)==vv[r,c]
    kp=ROOT/'kernels'/(pool+'_'+n+'_'+label+'.npz');np.savez_compressed(kp,dx=xx.astype(np.int16),dy=yy.astype(np.int16));qa.append(dict(direction=label,parameters=par,kernel_path=str(kp),kernel_sha256=sha(kp),FFT_integer_error_cloud=err,FFT_integer_error_valid=ev,direct_samples=len(sample),direct_agreement=True))
    if label=='sun':sun=support;complete=covered
    else:reverse=support;reversecomplete=covered
   k=int(sun.sum());ids=np.flatnonzero(ash);ordered=np.lexsort((ids,-q[2].ravel()[ids]));conf.ravel()[ids[ordered[:k]]]=True;assert int(conf.sum())==k
  observed=ash&~sun&complete;unknown=ash&~sun&~complete;assert np.array_equal(sun|observed|unknown,ash)
  p=ROOT/'masks'/(pool+'_'+n+'.npz');np.savez_compressed(p,original_A_Sh=ash,G_sun=sun,G_reverse=reverse,G_conf=conf,unmatched_observed=observed,unmatched_unknown=unknown,supported_context_incomplete=sun&~complete,reverse_context_incomplete=reverse&~reversecomplete)
  sealed.append(dict(pool=pool,name=n,scene=rec['scene'],status=rec['status'],file=str(p),sha256=sha(p),counts={k:int(v.sum()) for k,v in dict(original_A_Sh=ash,G_sun=sun,G_reverse=reverse,G_conf=conf,unmatched_observed=observed,unmatched_unknown=unknown,supported_context_incomplete=sun&~complete).items()},teacher_sha256=entry['sha256'],kernels=qa));print(pool,n,'A_Sh',int(ash.sum()),'sun',int(sun.sum()),flush=True)
 assert totals==EXPECTED
 write(ROOT/'GEOMETRY_INPUTS.json',meta);write(ROOT/'SELECTION_LOCK.json',dict(status='sealed',epoch=time.time(),preregistration_sha256=sha(ROOT/'PREREGISTRATION.json'),inputs_sha256=sha(ROOT/'GEOMETRY_INPUTS.json'),original_A_Sh=totals,rows=sealed,all_masks_before_truth=True,neural_calls=0,resources=guard()));write(ROOT/'STATUS.json',dict(status='selections_sealed',epoch=time.time()))
 # Human-readable complete160 member table retains failed metadata products.
 table(ROOT/'GEOMETRY_INPUTS.csv',[{k:(v if not isinstance(v,dict) else v) for k,v in dict(pool=r['pool'],name=r['name'],scene=r['scene'],biome=r['biome'],mtl_path=r['mtl_path'],mtl_sha256=r.get('mtl_sha256'),mtl_identity=r.get('mtl_identity'),acquired=r.get('acquired'),azimuth_deg=r.get('azimuth_deg'),elevation_deg=r.get('elevation_deg'),sun_units=r.get('sun_units'),native_tiff=r['native_tiff'],native_tiff_sha256=r.get('native_tiff_sha256'),crs=r.get('crs'),native_affine=r.get('native_affine'),output_affine=r.get('output_affine'),index_identity=r.get('index_identity'),gridcheck=r.get('gridcheck'),A_Sh=r['A_Sh'],status=r['status'],missing=r['missing']).items()} for r in meta])
if __name__=='__main__':main()

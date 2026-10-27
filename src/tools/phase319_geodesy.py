"""WGS84 Vincenty local geodesics, no extra dependency; rasterio handles CRS."""
import math
A=6378137.;F=1/298.257223563;B=(1-F)*A
def coefficients(cos2alpha):
 u=cos2alpha*(A*A-B*B)/(B*B)
 return 1+u/16384*(4096+u*(-768+u*(320-175*u))),u/1024*(256+u*(-128+u*(74-47*u)))
def correction(b,sin_sigma,cos_sigma,cos2m):
 return b*sin_sigma*(cos2m+b/4*(cos_sigma*(-1+2*cos2m*cos2m)-b/6*cos2m*(-3+4*sin_sigma*sin_sigma)*(-3+4*cos2m*cos2m)))
def inverse(lon1,lat1,lon2,lat2):
 u1=math.atan((1-F)*math.tan(math.radians(lat1)));u2=math.atan((1-F)*math.tan(math.radians(lat2)));su1,cu1=math.sin(u1),math.cos(u1);su2,cu2=math.sin(u2),math.cos(u2);L=math.radians(lon2-lon1);lam=L
 for _ in range(100):
  sl,cl=math.sin(lam),math.cos(lam);ss=math.hypot(cu2*sl,cu1*su2-su1*cu2*cl)
  if ss==0:return 0.,0.
  cs=su1*su2+cu1*cu2*cl;sigma=math.atan2(ss,cs);sa=cu1*cu2*sl/ss;ca2=1-sa*sa;c2m=cs-2*su1*su2/ca2 if ca2>1e-15 else 0.;c=F/16*ca2*(4+F*(4-3*ca2));new=L+(1-c)*F*sa*(sigma+c*ss*(c2m+c*cs*(-1+2*c2m*c2m)))
  if abs(new-lam)<1e-13:break
  lam=new
 else:raise RuntimeError('local Vincenty inverse did not converge')
 aa,bb=coefficients(ca2);distance=B*aa*(sigma-correction(bb,ss,cs,c2m));az=math.degrees(math.atan2(cu2*math.sin(lam),cu1*su2-su1*cu2*math.cos(lam)))
 return az,distance
def forward(lon,lat,azimuth,distance):
 az=math.radians(azimuth);u=math.atan((1-F)*math.tan(math.radians(lat)));su,cu=math.sin(u),math.cos(u);sa=cu*math.sin(az);ca2=1-sa*sa;sig1=math.atan2(math.tan(u),math.cos(az));aa,bb=coefficients(ca2);sigma=distance/(B*aa)
 for _ in range(100):
  ss,cs=math.sin(sigma),math.cos(sigma);c2m=math.cos(2*sig1+sigma);new=distance/(B*aa)+correction(bb,ss,cs,c2m)
  if abs(new-sigma)<1e-13:break
  sigma=new
 else:raise RuntimeError('local Vincenty forward did not converge')
 ss,cs=math.sin(sigma),math.cos(sigma);tmp=su*ss-cu*cs*math.cos(az);lat2=math.atan2(su*cs+cu*ss*math.cos(az),(1-F)*math.hypot(sa,tmp));lam=math.atan2(ss*math.sin(az),cu*cs-su*ss*math.cos(az));c=F/16*ca2*(4+F*(4-3*ca2));L=lam-(1-c)*F*sa*(sigma+c*ss*(c2m+c*cs*(-1+2*c2m*c2m)))
 return lon+math.degrees(L),math.degrees(lat2)
def qa():
 assert abs(inverse(0,0,1,0)[1]-111319.49079327357)<1e-5
 assert abs(inverse(0,0,0,1)[1]-110574.38855779878)<1e-5
 rows=[]
 for lon,lat in [(0,0),(-69,-18),(100,70),(-120,45)]:
  for az in [0,45,90,180,270]:
   ll=forward(lon,lat,az,1000);a,d=inverse(lon,lat,*ll);assert abs(d-1000)<1e-5 and abs((a-az+180)%360-180)<1e-6;rows.append(dict(lon=lon,lat=lat,azimuth=az,distance_error_m=abs(d-1000),azimuth_error_deg=abs((a-az+180)%360-180)))
 return dict(status='passed',ellipsoid='WGS84 a6378137 inverse_flattening298.257223563',published_equator_and_meridian_one_degree_distances_checked=True,roundtrips=rows)

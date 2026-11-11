"""Use existing remote geospatial libraries on metadata streamed through stdin."""
from pathlib import Path
import json,csv,subprocess,time,hashlib

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase245/phase245_20261031'

REMOTE=r'''
import importlib.util,json,sys,math,collections
available={x:importlib.util.find_spec(x) is not None for x in ['shapely','rasterio']}
out={'available':available,'new_model_calls':0,'image_or_label_payloads_opened':0,'remote_files_written':0,'results':[]}
if all(available.values()):
 import shapely,rasterio
 from rasterio.warp import transform_geom
 from shapely.geometry import shape,box,Polygon,mapping
 from shapely import wkt
 out['versions']={'shapely':shapely.__version__,'rasterio':rasterio.__version__,'GDAL':rasterio.__gdal_version__}
 def project(g,src,dst):return shape(transform_geom(src,dst,mapping(g),precision=-1))
 source=collections.defaultdict(list)
 for row in DATA['source']:
  epsg=row['proj_epsg']
  geom=wkt.loads(row['proj_geometry']);coords=list(geom.exterior.coords);dense=[]
  for a,b in zip(coords,coords[1:]):
   dense.extend([(a[0]+(b[0]-a[0])*t/16,a[1]+(b[1]-a[1])*t/16) for t in range(16)])
  g=project(Polygon(dense),'EPSG:'+epsg,'EPSG:4326');source[(row['role'],row['s2_id'])].append(g)
 for c in DATA['candidates']:
  g=shape(c['geometry']);xmin,ymin,xmax,ymax=g.bounds
  if xmax-xmin>180:
   out['results'].append({'id':c['id'],'status':'dateline_pending'});continue
  lon=(xmin+xmax)/2;lat=(ymin+ymax)/2
  prj=f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m'
  candidate=project(g,'EPSG:4326',prj);hits=[];unknown=0
  for h in DATA['history']:
   b=h.get('wgs84_bounds')
   if not b:unknown+=1;continue
   if b[0]>b[2]:unknown+=1;continue
   dx=min(180,3/(111*max(.01,math.cos(math.radians(max(abs(lat),abs(b[1]),abs(b[3])))))));dy=3/110
   if xmax+dx<b[0] or b[2]+dx<xmin or ymax+dy<b[1] or b[3]+dy<ymin:continue
   key=(h['role'],h['product']);exact=source.get(key)
   if exact:
    distances=[candidate.distance(project(s,'EPSG:4326',prj)) for s in exact];distance=min(distances);basis='Source_projected_patch_polygon'
   else:
    # Densify envelope edges before metric transformation.
    coords=list(box(*b).exterior.coords);dense=[]
    for a,z in zip(coords,coords[1:]):dense.extend([(a[0]+(z[0]-a[0])*t/32,a[1]+(z[1]-a[1])*t/32) for t in range(32)])
    distance=candidate.distance(project(Polygon(dense),'EPSG:4326',prj));basis='historical_conservative_envelope'
   if distance<=2000:hits.append({'history_product':h['product'],'role':h['role'],'distance_m':distance,'geometry_basis':basis})
  out['results'].append({'id':c['id'],'pool':c['pool'],'status':'geometry_flag' if hits else 'no_registered_geometry_intersection','hits':hits,'unresolved_history_geometries':unknown,'footprint_basis':'same-acquisition C2 L2 proxy, not original label grid','distance_threshold_m':2000})
 # Group all candidates transitively before exclusions, including flagged bridge scenes.
 pairs=[];parent={c['id']:c['id'] for c in DATA['candidates']}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 for i,a in enumerate(DATA['candidates']):
  g=shape(a['geometry']);lon=g.centroid.x;lat=g.centroid.y;crs=f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m'
  ga=project(g,'EPSG:4326',crs)
  for b in DATA['candidates'][i+1:]:
   h=shape(b['geometry'])
   if g.distance(h)>5:continue
   d=ga.distance(project(h,'EPSG:4326',crs))
   if d<=2000:
    pairs.append({'a':a['id'],'b':b['id'],'distance_m':d});u,v=find(a['id']),find(b['id']);parent[max(u,v)]=min(u,v)
 grouped=collections.defaultdict(list)
 for key in parent:grouped[find(key)].append(key)
 out['candidate_pairs']=pairs;out['candidate_components']=list(grouped.values())
 out['source_geometry_records']=len(DATA['source'])
print(json.dumps(out))
'''

def main():
 candidates=[]
 for folder,pool in [(ROOT/'outputs/phase244/phase244_20261031','prior_L8'),(OUT,'L7_Irish')]:
  for p in folder.glob('*_stac.json'):
   v=json.loads(p.read_text());items=v.get('items',[])
   if len(items)==1:candidates.append({'id':p.name[:-10],'pool':pool,'geometry':items[0]['geometry']})
 source=[];evidence={}
 for name,role in [('train','Source_train'),('val','Source_val')]:
  p=ROOT/f'outputs/phase65/source_metadata/acquired/{name}_metadata.csv'
  evidence[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
  for r in csv.DictReader(p.open(encoding='utf-8-sig')):source.append({k:r[k] for k in ['s2_id','proj_epsg','proj_geometry']}|{'role':role})
 history=json.loads((ROOT/'outputs/phase102/phase102_20261013/inputs/history_positions_only.json').read_text())['records']
 data={'candidates':candidates,'source':source,'history':history}
 program='DATA='+repr(data)+'\n'+REMOTE
 attempts=[]
 for i in range(4):
  try:
   p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs','/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -'],input=program,capture_output=True,text=True,encoding='utf-8',timeout=180)
   attempts.append({'attempt':i+1,'exit':p.returncode,'stderr':p.stderr[-1500:]})
   if p.returncode==0:
    result=json.loads(p.stdout);(OUT/'GEOMETRY_REFINEMENT.json').write_text(json.dumps(result,indent=2),encoding='utf-8');break
  except Exception as e:attempts.append({'attempt':i+1,'error':str(e)[:1000]})
  if i<3:time.sleep(3)
 (OUT/'GEOMETRY_EXECUTION.json').write_text(json.dumps({'attempts':attempts,'source_evidence':evidence,'metadata_bytes_streamed':len(program.encode()),'remote_writes':0},indent=2),encoding='utf-8')
 if (OUT/'GEOMETRY_REFINEMENT.json').exists():
  r=json.loads((OUT/'GEOMETRY_REFINEMENT.json').read_text());print(json.dumps({k:v for k,v in r.items() if k!='results'},indent=2));print('counts',__import__('collections').Counter((x.get('pool'),x['status']) for x in r['results']));print('clear',[x['id'] for x in r['results'] if x['status']=='no_registered_geometry_intersection'])
 else:print(json.dumps(attempts,indent=2))

if __name__=='__main__':main()

"""One bounded metadata-only Source overlap audit. Execute only on gzs."""
import os,sys,time,json,csv,hashlib,re,math,io,zipfile,resource,traceback,shutil,numbers
from pathlib import Path
from collections import Counter,defaultdict
R=Path(__file__).resolve().parents[2]
O=Path('/home/scv/shared/phase321_20261028')
D=Path('/home/scv/shared/data')
SOURCE='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
START=time.monotonic();PINS={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(n,v):
 p=O/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def read(p):
 p=Path(p).resolve();assert p.is_relative_to(R) or p.is_relative_to(Path('/home/scv/shared'))
 b=p.read_bytes();PINS[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)};return b
def js(p):return json.loads(read(p).decode('utf-8-sig'))
def rows(p):return list(csv.DictReader(read(p).decode('utf-8-sig').splitlines()))
def table(n,rs):
 if not rs:put(n+'.empty.json',[]);return
 with (O/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader()
  for r in rs:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def guard():
 assert time.monotonic()-START<1800 and 'torch' not in sys.modules
 mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
 assert mem>=4*1024**3 and shutil.disk_usage(O).free>=4*1024**3
 assert sum(p.stat().st_size for p in O.rglob('*') if p.is_file())*2<200000000
 return {'host_available_bytes':mem,'maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
def identity(p,tile=None,day=None):
 t=re.search(r'_T(\d{2}[A-Z]{3})(?:_|\.|$)',p)
 d=re.search(r'_MSIL[12][AC]_(\d{8}T\d{6})_',p)
 if not d:d=re.search(r'_V(\d{8}T\d{6})_',p)
 obs=d.group(1) if d else ''
 return {'tile':tile or (t.group(1) if t else ''),'day':day or obs[:8],'time':obs,'platform':p[:3]}
def main():
 assert sys.platform=='linux' and os.environ.get('CONDA_DEFAULT_ENV')=='cloud-lite-pt210'
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));O.mkdir(exist_ok=False)
 put('STARTED.json',{'pid':os.getpid(),'epoch':time.time(),'wall_cap_seconds':1800});guard()
 sys.path.insert(0,str(R/'src/work_dirs/phase65a/geospatial_python'))
 import shapefile
 from shapely import wkt
 from shapely.geometry import shape,mapping,Polygon,box,MultiPolygon
 from shapely.ops import unary_union
 from shapely.affinity import translate
 from shapely.strtree import STRtree
 from rasterio.warp import transform_geom
 def project(g,src,dst):return shape(transform_geom(src,dst,mapping(g),precision=-1))
 def dense(g):
  def ring(coords):
   out=[];c=list(coords)
   for a,b in zip(c[:-1],c[1:]):
    n=max(1,math.ceil(math.hypot(b[0]-a[0],b[1]-a[1])/250));out.extend((a[0]+(b[0]-a[0])*j/n,a[1]+(b[1]-a[1])*j/n) for j in range(n))
   return out+[out[0]]
  if g.geom_type=='MultiPolygon':return MultiPolygon([dense(x) for x in g.geoms])
  assert g.geom_type=='Polygon' and g.is_valid and not g.is_empty
  return Polygon(ring(g.exterior.coords),[ring(x.coords) for x in g.interiors])
 def geographic(g,crs):
  v=project(dense(g),crs,'EPSG:4326')
  if v.bounds[2]-v.bounds[0]>180:
   def unwrap(coords):
    out=[]
    for x,y in coords:
     if out:
      while x-out[-1][0]>180:x-=360
      while x-out[-1][0]<-180:x+=360
     out.append((x,y))
    return out
   assert v.geom_type=='Polygon';v=Polygon(unwrap(v.exterior.coords),[unwrap(x.coords) for x in v.interiors])
   v=unary_union([translate(v,xoff=k*360).intersection(box(-180,-90,180,90)) for k in [-1,0,1]])
  assert v.is_valid and not v.is_empty;return v
 # Mandatory algorithm checks: adjacent UTM zones and boundary/area decisions.
 a=box(0,0,10,10);assert a.intersection(box(10,0,20,10)).area==0
 assert not a.intersection(box(9.95,0,20,10)).area>1
 assert a.intersection(box(9,0,20,10)).area>1
 p=box(499900,4999900,500100,5000100);gg=geographic(p,'EPSG:32631')
 assert project(gg,'EPSG:4326','EPSG:32631').intersection(p).area>39999
 anti=geographic(box(820000,0,840000,20000),'EPSG:32660');assert anti.is_valid
 put('SYNTHETIC_CHECKS.json',{'area_threshold_boundary':True,'projection_roundtrip':True,'antimeridian':True})
 plan=read(R/'src/research_plans/phase321.md');read(Path(__file__))
 code={n:sha(R/'src/tools'/n) for n in ['audit_phase321.py','deliver_phase321.py']}
 # Hash checkpoint as bytes only; no deserialization.
 assert sha(R/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')==SOURCE
 meta=D/'phase65_source_metadata';sm=js(meta/'source_metadata_manifest.json')
 source=[]
 for split in ['train','val']:
  rs=rows(meta/f'{split}_metadata.csv');assert len(rs)=={'train':8490,'val':535}[split]
  assert sha(meta/f'{split}_metadata.csv')==sm['splits'][split]['sha256']
  for ordinal,r in enumerate(rs):source.append(dict(id=f'{split}:{ordinal}',split=split,ordinal=ordinal,original_index=r['index'],product=r['s2_id'],crs='EPSG:'+r['proj_epsg'],native=wkt.loads(r['proj_geometry']),**identity(r['s2_id'])))
 mapping_audit=js(R/'src/work_dirs/phase65a/source_mapping/mapping_audit.json');assert mapping_audit['status']=='verified_rgb_ordinal_mapping'
 mm={(x['split'],x['png_ordinal']):x for x in mapping_audit['mappings']}
 assert all(mm[(s['split'],s['ordinal'])]['exact_rgb_match'] and mm[(s['split'],s['ordinal'])]['product']==s['product'] and str(mm[(s['split'],s['ordinal'])]['original_index'])==s['original_index'] for s in source)
 targets=[]
 cat=D/'sentinel2_cloud_mask_catalogue_4172871';tags=rows(cat/'classification_tags.csv');tagmap={x['scene'].removesuffix('.SAFE'):x for x in tags}
 raw=read(cat/'shapefiles.zip')
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  for n in sorted(z.namelist()):
   if not n.endswith('.shp'):continue
   with shapefile.Reader(shp=io.BytesIO(z.read(n)),shx=io.BytesIO(z.read(n[:-4]+'.shx')),dbf=io.BytesIO(z.read(n[:-4]+'.dbf'))) as f:
    shapes=f.shapes();assert len(shapes)==1;g=shape(shapes[0].__geo_interface__)
   product=Path(n).stem;tag=tagmap[product]
   targets.append(dict(dataset='Catalogue',key=product,product=product,crs=z.read(n[:-4]+'.prj').decode(),native=g,geometry_basis='author precise subscene polygon',shadow_label=str(tag['shadows_marked']),**identity(product)))
 assert len(targets)==513
 alcd=D/'sentinel2_alcd_1460961/imagery';am=js(alcd/'product_manifest.json');ag={r['key']:r for r in js(alcd/'geometry_audit.json')['scenes']}
 for r in am['scenes']:
  geom=ag[r['key']];q=r['parameters'];product=Path(r['gcs_product_prefix'].rstrip('/')).name.removesuffix('.SAFE')
  assert geom['alignment_status']=='all_six_grids_match'
  targets.append(dict(dataset='ALCD',key=r['key'],product=product,crs=geom['crs'],native=box(*geom['bounds']),geometry_basis='existing verified raster grid extent; includes ignored areas',shadow_label='reference_semantics_previously_audited',**identity(product,tile=q['tile'],day=q['cloudy_date'])))
 pix=js(R/'src/research_results/phase102_20261013/inputs/candidate_products_metadata.json')
 for r in pix:
  if r['source']!='PixBox5036991':continue
  g=r.get('grid');native=box(g['ULX'],g['ULY']+g['NROWS']*g['YDIM'],g['ULX']+g['NCOLS']*g['XDIM'],g['ULY']) if g else None
  targets.append(dict(dataset='PixBox',key=r['product_id'],product=r['product_id'],crs=r.get('crs'),native=native,geometry_basis='parent native image grid; conservative relative to sampled points',shadow_label='point_reference',**identity(r['product_id'],tile=r['tile'],day=r['utc_day'])))
 assert Counter(x['dataset'] for x in targets)=={'Catalogue':513,'ALCD':37,'PixBox':29}
 pi=rows(R/'src/research_results/phase102_20261013/product_location_index.csv')
 evaluated={('PixBox',r['product_id']) for r in pi if r['source']=='PixBox5036991' and r['status']=='candidate_metadata_pass'};assert len(evaluated)==16
 # Phase317 CMC mapping is mandatory for sensitivity join.
 refs=js(Path('/home/scv/shared/phase317_20261026/REFERENCE_LOCK.json'))['rows']
 scene_map={r['name']:r['scene'] for r in refs if r['pool']=='CMC'}
 # Complete source and target inventory pinned before any matching.
 for s in source:
  try:s['geo']=geographic(s['native'],s['crs']);s['geometry_error']=''
  except Exception as e:s['geo']=None;s['geometry_error']=repr(e)
 for t in targets:
  try:
   assert t['native'] is not None and t['crs'];t['geo']=geographic(t['native'],t['crs']);t['geometry_error']=''
  except Exception as e:t['geo']=None;t['geometry_error']=repr(e)
 put('INPUT_LOCK.json',{'files':PINS,'Source_sha256':SOURCE,'code':code,'plan_sha256':hashlib.sha256(plan).hexdigest(),'targets':dict(Counter(x['dataset'] for x in targets)),'source_rows':dict(Counter(s['split'] for s in source)),'predictions_recomputed':False,'V4_scope':'reference only; full training manifest audit not performed'})
 good=[s for s in source if s['geo'] is not None];geoms=[s['geo'] for s in good];tree=STRtree(geoms);lookup={id(g):i for i,g in enumerate(geoms)}
 bytile=defaultdict(list)
 for s in source:bytile[s['tile']].append(s)
 pairs=[];results=[]
 for ti,t in enumerate(targets):
  if ti%20==0:guard()
  hits={s['id']:s for s in bytile[t['tile']]};errors=[]
  if t['geo'] is not None:
   for h in tree.query(t['geo']):
    s=good[int(h) if isinstance(h,numbers.Integral) else lookup[id(h)]];hits[s['id']]=s
  else:errors.append(t['geometry_error'])
  flags={split:{'same_scene_candidate':False,'same_tile_different_day':False,'spatial_overlap':False,'exact_time':False,'max_intersection_m2':0.0} for split in ['train','val']}
  for s in sorted(hits.values(),key=lambda s:s['id']):
   same=t['tile']==s['tile'] and t['day']==s['day'];tile=t['tile']==s['tile'] and t['day']!=s['day'];area=None;fra=None;frb=None
   if s['geo'] is not None and t['geo'] is not None:
    try:
     if not s['geo'].intersects(t['geo']):area=0.0;fra=0.0;frb=0.0
     else:
      sg=project(s['geo'],'EPSG:4326',t['crs']);area=sg.intersection(t['native']).area;fra=area/sg.area;frb=area/t['native'].area
    except Exception as e:errors.append(s['id']+':'+repr(e))
   spatial=area is not None and area>1.0
   exact=bool(same and t['time'] and s['time']==t['time'] and s['platform']==t['platform'])
   if same or tile or spatial:
    f=flags[s['split']]
    for k,v in [('same_scene_candidate',same),('same_tile_different_day',tile),('spatial_overlap',spatial),('exact_time',exact)]:f[k]|=v
    f['max_intersection_m2']=max(f['max_intersection_m2'],area or 0)
    pairs.append(dict(dataset=t['dataset'],target_key=t['key'],target_product=t['product'],source_id=s['id'],source_product=s['product'],source_split=s['split'],same_scene_candidate=same,exact_time=exact,same_tile_different_day=tile,spatial_overlap=spatial,intersection_m2=area,source_fraction=fra,target_fraction=frb))
  for split in ['train','val']:
   f=flags[split];unknown=bool(errors or any(s['split']==split and s['geo'] is None for s in source))
   category='same_scene_candidate' if f['same_scene_candidate'] else 'same_tile_different_day' if f['same_tile_different_day'] else 'spatial_overlap' if f['spatial_overlap'] else 'unknown' if unknown else 'none_detected'
   results.append(dict(dataset=t['dataset'],key=t['key'],product=t['product'],tile=t['tile'],day=t['day'],time=t['time'],source_split=split,category=category,**f,unknown=unknown,scene_clean=not(f['same_scene_candidate'] or unknown),strict_clean=not(f['same_scene_candidate'] or f['same_tile_different_day'] or f['spatial_overlap'] or unknown),geometry_basis=t['geometry_basis'],shadow_label=t['shadow_label'],evaluated=t['product'] in scene_map if t['dataset']=='Catalogue' else ('PixBox',t['product']) in evaluated,errors=errors))
 table('TARGET_OVERLAP.csv',results);table('MATCH_PAIRS.csv',pairs)
 summary=[]
 for ds in ['Catalogue','ALCD','PixBox']:
  for split in ['train','val']:
   rr=[r for r in results if r['dataset']==ds and r['source_split']==split];count=sum(r['same_scene_candidate'] for r in rr);pct=100*count/len(rr)
   summary.append(dict(dataset=ds,split=split,products=len(rr),same_scene_candidates=count,same_scene_pct=pct,extent='none_detected' if count==0 else 'small' if pct<=5 else 'large',same_tile_different_day=sum(r['same_tile_different_day'] for r in rr),spatial_overlap=sum(r['spatial_overlap'] for r in rr),unknown=sum(r['unknown'] for r in rr),scene_clean=sum(r['scene_clean'] for r in rr),strict_clean=sum(r['strict_clean'] for r in rr),evaluated_products=sum(r['evaluated'] for r in rr)))
 table('COUNTS.csv',summary)
 # Cached statistics only; same predictions, unchanged support on retained groups.
 per=rows(R/'src/research_results/phase317_20261026/PER_SCENE_METRICS.csv');cmc=[r for r in per if r['pool']=='CMC' and r['strategy']=='Source'];assert len(cmc)==32 and len(scene_map)==32
 pg=rows(R/'src/research_results/phase112_20261014/per_group_results.csv')
 sensitivity=[]
 def mean(v):return sum(v)/len(v) if v else None
 for roles in [('train',),('train','val')]:
  for policy in ['full','scene_clean','strict_clean']:
   retained=[r for r in results if r['dataset']=='Catalogue' and r['source_split'] in roles and (policy=='full' or r[policy])]
   accepted={r['product'] for r in retained if sum(q['product']==r['product'] for q in retained)==len(roles)}
   ss={scene_map[p] for p in accepted if p in scene_map};rr=[r for r in cmc if r['scene'] in ss];cm=[[0]*3 for _ in range(3)]
   for r in rr:
    c=json.loads(r['confusion']);cm=[[cm[i][j]+c[i][j] for j in range(3)] for i in range(3)]
   ious=[cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)-cm[i][i])*100 for i in range(3) if sum(cm[i])+sum(row[i] for row in cm)-cm[i][i]>0]
   sensitivity.append(dict(dataset='Catalogue',roles='+'.join(roles),policy=policy,retained_groups=len(rr),macro_AP_pct=mean([float(r['AP_pct']) for r in rr if r['AP_pct']!='NA']),macro_mIoU_pct=mean([float(r['mIoU_pct']) for r in rr]),pooled_mIoU_pct=mean(ious),macro_FPR_pct=mean([float(r['FPR_pct']) for r in rr if r['FPR_pct']!='NA']),confusion=cm,note='Phase317 cached sufficient statistics; pooled AP not reconstructible from group AP'))
   badtiles={r['tile'] for r in results if r['dataset']=='PixBox' and r['evaluated'] and r['source_split'] in roles and policy!='full' and not r[policy]}
   rr=[r for r in pg if not set(r['tiles'].split('|'))&badtiles]
   sensitivity.append(dict(dataset='PixBox',roles='+'.join(roles),policy=policy,retained_groups=len(rr),macro_AP_pct=mean([float(r['Source_AP'])*100 for r in rr if r['Source_AP'] not in ['NA','']]),macro_mIoU_pct=None,pooled_mIoU_pct=None,macro_FPR_pct=None,confusion=None,note='Phase112 cached Source group AP; entire group removed if any evaluated member excluded; pooled AP/FPR not present in sufficient statistics'))
 table('SOURCE_SENSITIVITY.csv',sensitivity)
 put('RESULT.json',{'status':'completed','counts':summary,'sensitivity':sensitivity,'source_geometry_unknown':sum(s['geo'] is None for s in source),'resources':guard(),'elapsed_seconds':time.monotonic()-START,'training':0,'model_loads':0,'forwards':0,'pixel_reads':0,'downloads':0,'V4':'reference only; OCM not audited; no independent external claim','ALCD_metrics':'NA: no frozen Source predictions','hard_negative_caution':'overlap with920 OCM hard negatives would primarily qualify false-positive improvements; not audited in this scoped phase'})
 put('INPUT_FINAL_SHA.json',{'files':PINS,'all_unchanged':all(sha(p)==v['sha256'] for p,v in PINS.items())})
 assert all(sha(p)==v['sha256'] for p,v in PINS.items());put('EXIT_CODE.json',{'exit_code':0})
 print(json.dumps(js(O/'RESULT.json'),ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except Exception:
  if O.exists():put('FAILURE.json',{'traceback':traceback.format_exc(),'elapsed':time.monotonic()-START});put('EXIT_CODE.json',{'exit_code':1})
  raise

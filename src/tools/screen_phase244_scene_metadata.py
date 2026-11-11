"""Scene identity and conservative footprint screen; labels remain unopened."""
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import urllib.request, json, re, csv, hashlib, math, collections

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/phase244/phase244_20261031'
HISTORY=ROOT/'outputs/phase102/phase102_20261013/inputs/history_positions_only.json'

def main():
 desc=OUT/'pixbox_landsat8_cmix_20150527_descriptions.txt'
 products=re.findall(r'^(\d+);0;(LC8\d{13}LGN\d{2})_MTL.dim',desc.read_text(),re.M)
 candidates=[]
 for key,name in products:
  dt=datetime.strptime(name[9:16],'%Y%j').strftime('%Y-%m-%d')
  candidates.append(dict(pool='PixBox_L8',id=name,point_product_id=key,path=name[3:6],row=name[6:9],date=dt,identity_evidence=desc.name))
 # Metadata transcribed from official WHU page, not inferred from labels/images.
 for path,row,date in [('118','032','2018-06-24'),('119','038','2018-05-14'),('123','039','2018-04-08'),('124','033','2018-04-15'),('126','035','2018-05-13'),('127','034','2018-06-23')]:
  candidates.append(dict(pool='WHU_Cloud',id=f'WHU_{path}{row}_{date}',point_product_id='',path=path,row=row,date=date,identity_evidence='https://gpcv.whu.edu.cn/data/WHU_Cloud_Dataset.html'))
 def query(c):
  dest=OUT/(c['id']+'_stac.json')
  if dest.exists():return json.loads(dest.read_text())
  q={'collections':['landsat-c2-l2'],'datetime':c['date']+'T00:00:00Z/'+c['date']+'T23:59:59Z','query':{'landsat:wrs_path':{'eq':c['path']},'landsat:wrs_row':{'eq':c['row']}},'limit':10}
  req=urllib.request.Request('https://planetarycomputer.microsoft.com/api/stac/v1/search',data=json.dumps(q).encode(),headers={'Content-Type':'application/json','User-Agent':'research-metadata-audit/1.0'})
  try:
   with urllib.request.urlopen(req,timeout=30) as f:
    b=f.read(1500000)
   v=json.loads(b);features=v.get('features',[])
   result={'query':q,'metadata_endpoint':req.full_url,'returned_count':len(features),'items':[{'id':x['id'],'bbox':x['bbox'],'geometry':x['geometry'],'properties':x['properties']} for x in features]}
  except Exception as e:result={'query':q,'error':str(e),'items':[]}
  dest.write_text(json.dumps(result,indent=2),encoding='utf-8');return result
 with ThreadPoolExecutor(max_workers=3) as ex:results=list(ex.map(query,candidates))
 h=json.loads(HISTORY.read_text())['records'];edges=[]
 def intersects(a,b):
  if a[0]>a[2] or b[0]>b[2]:return True # dateline ambiguity -> conservative flag
  lat=max(abs(a[1]),abs(a[3]),abs(b[1]),abs(b[3]));dx=min(180,2/(111.0*max(.01,math.cos(math.radians(lat)))));dy=2/110.0
  return not(a[2]+dx<b[0] or b[2]+dx<a[0] or a[3]+dy<b[1] or b[3]+dy<a[1])
 for c,v in zip(candidates,results):
  items=v['items'];boxes=[x['bbox'] for x in items]
  c['stac_items']=len(items);c['stac_ids']=';'.join(x['id'] for x in items)
  c['bbox']=json.dumps(boxes[0]) if len(items)==1 else ''
  exact=[x for x in h if c['id'] in x['product']]
  same_wrs=[x for x in h if re.search(r'LC8'+c['path']+c['row'],x['product'])]
  matches=[x for x in h if boxes and x.get('wgs84_bounds') and any(intersects(b,x['wgs84_bounds']) for b in boxes)]
  for kind,seq in [('exact_product',exact),('same_wrs',same_wrs),('conservative_bbox_2km',matches)]:
   for x in seq:edges.append({'candidate':c['id'],'kind':kind,'history_product':x['product'],'history_role':x['role'],'history_evidence':x.get('evidence','')})
  c['exact_history_products']=len(exact);c['same_wrs_history_records']=len(same_wrs);c['bbox_history_records']=len(matches)
  c['matched_roles']=';'.join(sorted({x['role'] for x in matches}))
  c['status']='recorded_product_exposure' if exact else 'historical_spatial_flag' if same_wrs or matches else 'identity_or_geometry_pending' if len(items)!=1 else 'preliminary_no_overlap_found'
  c['ready']=False
 # Candidate-candidate proximity also prevents treating products as independent groups.
 pair_edges=[]
 for i,a in enumerate(candidates):
  for b in candidates[i+1:]:
   if (a['path'],a['row'])==(b['path'],b['row']) or (a['bbox'] and b['bbox'] and intersects(json.loads(a['bbox']),json.loads(b['bbox']))):pair_edges.append({'a':a['id'],'b':b['id'],'basis':'same_wrs_or_conservative_bbox_2km'})
 for name,rows in [('CANDIDATES.csv',candidates),('HISTORY_MATCHES.csv',edges),('CANDIDATE_PAIRS.csv',pair_edges)]:
  with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['none']);w.writeheader();w.writerows(rows)
 summary={'counts':dict(collections.Counter(c['pool']+'|'+c['status'] for c in candidates)), 'candidate_count':len(candidates),'history_records':len(h),'history_SHA256':hashlib.sha256(HISTORY.read_bytes()).hexdigest(),'candidate_pair_flags':len(pair_edges),'unlocated':[c['id'] for c in candidates if c['stac_items']!=1],'image_downloads':0,'label_CSV_parsed':False,'new_model_calls':0,'geometry_limit':'C2 L2 same-acquisition footprint proxy, not exact original label grid. Conservative 2km bbox screen; flags may overexclude. No-overlap is not final admission.'}
 (OUT/'SCREEN_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print(json.dumps(summary,indent=2))
 print('\n'.join(f"{c['id']} {c['status']} sameWRS={c['same_wrs_history_records']} bbox={c['bbox_history_records']} roles={c['matched_roles']}" for c in candidates))

if __name__=='__main__':main()

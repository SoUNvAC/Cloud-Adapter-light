"""Official scene-table and STAC metadata screening, no label pixels."""
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import csv,json,re,urllib.request,math,collections

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase245/phase245_20261031'

def main():
 rows=json.loads((OUT/'IRISH_HTML_TABLES.json').read_text());scenes=[]
 for row in rows:
  if len(row)!=7:continue
  m=re.fullmatch(r'Path (\d+) Row (\d+)(?:-(\d+))?',row[1]['text'])
  if not m:continue
  path,r,variant=m.groups();date=datetime.strptime(row[2]['text'],'%m/%d/%Y').strftime('%Y-%m-%d')
  scenes.append({'id':f'Irish_{int(path):03d}{int(r):03d}_{date}','path':path.zfill(3),'row':r.zfill(3),'date':date,'official_shadow_status':row[5]['text'],'download_url':row[1]['href'],'variant':variant or ''})
 with (OUT/'IRISH_ALL_SCENES.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(scenes[0]));w.writeheader();w.writerows(scenes)
 eligible=[r for r in scenes if r['official_shadow_status']=='yes']
 def query(c):
  p=OUT/(c['id']+'_stac.json')
  if p.exists():return json.loads(p.read_text())
  q={'collections':['landsat-c2-l2'],'datetime':c['date']+'T00:00:00Z/'+c['date']+'T23:59:59Z','query':{'landsat:wrs_path':{'eq':c['path']},'landsat:wrs_row':{'eq':c['row']},'platform':{'eq':'landsat-7'}},'limit':10}
  try:
   req=urllib.request.Request('https://planetarycomputer.microsoft.com/api/stac/v1/search',data=json.dumps(q).encode(),headers={'Content-Type':'application/json','User-Agent':'research-metadata-audit/1.0'})
   with urllib.request.urlopen(req,timeout=25) as r:v=json.loads(r.read(1500000))
   result={'query':q,'items':[{'id':x['id'],'bbox':x['bbox'],'geometry':x['geometry'],'properties':x['properties']} for x in v.get('features',[])]}
  except Exception as e:result={'query':q,'items':[],'error':str(e)}
  p.write_text(json.dumps(result,indent=2),encoding='utf-8');return result
 with ThreadPoolExecutor(max_workers=3) as ex:results=list(ex.map(query,eligible))
 history=json.loads((ROOT/'outputs/phase102/phase102_20261013/inputs/history_positions_only.json').read_text())['records']
 def intersects(a,b):
  if a[0]>a[2] or b[0]>b[2]:return True
  lat=max(abs(a[1]),abs(a[3]),abs(b[1]),abs(b[3]));dx=min(180,2/(111*max(.01,math.cos(math.radians(lat)))));dy=2/110
  return not(a[2]+dx<b[0] or b[2]+dx<a[0] or a[3]+dy<b[1] or b[3]+dy<a[1])
 edges=[]
 for c,v in zip(eligible,results):
  items=v['items'];c['stac_items']=len(items);c['stac_ids']=';'.join(x['id'] for x in items)
  matches=[h for h in history if h.get('wgs84_bounds') and any(intersects(x['bbox'],h['wgs84_bounds']) for x in items)]
  for h in matches:edges.append({'candidate':c['id'],'history_product':h['product'],'history_role':h['role']})
  c['bbox_flags']=len(matches);c['matched_roles']=';'.join(sorted({h['role'] for h in matches}));c['status']='metadata_location_pending' if len(items)!=1 else 'historical_spatial_flag' if matches else 'preliminary_no_bbox_overlap'
 for name,rows in [('IRISH_SHADOW_CANDIDATES.csv',eligible),('IRISH_HISTORY_FLAGS.csv',edges)]:
  with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['none']);w.writeheader();w.writerows(rows)
 summary={'official_scenes':len(scenes),'shadow_yes':len(eligible),'shadow_no':len(scenes)-len(eligible),'status_counts':dict(collections.Counter(c['status'] for c in eligible)),'preliminary_candidates':[c['id'] for c in eligible if c['status']=='preliminary_no_bbox_overlap'],'new_model_calls':0,'label_pixels_read':0,'image_downloads':0}
 (OUT/'IRISH_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

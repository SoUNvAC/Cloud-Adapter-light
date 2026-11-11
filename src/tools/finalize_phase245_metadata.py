"""Freeze metadata-only groups; performance/label support remain unknown."""
from pathlib import Path
import json,csv,hashlib,collections,urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase245/phase245_20261031'

def main():
 result=json.loads((OUT/'GEOMETRY_REFINEMENT.json').read_text());lookup={r['id']:r for r in result['results']}
 rows=list(csv.DictReader((OUT/'IRISH_SHADOW_CANDIDATES.csv').open(encoding='utf-8-sig')));meta={r['id']:r for r in rows}
 parent={k:k for k in lookup}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a,b=find(a),find(b);parent[max(a,b)]=min(a,b)
 for comp in result['candidate_components']:
  for k in comp[1:]:union(comp[0],k)
 # Conservative same-day WRS-path grouping for this one Landsat-7 platform.
 for i,a in enumerate(rows):
  for b in rows[i+1:]:
   if (a['path'],a['date'])==(b['path'],b['date']):union(a['id'],b['id'])
 groups=collections.defaultdict(list)
 for k in meta:groups[find(k)].append(k)
 records=[]
 for members in sorted(groups.values()):
  members=sorted(members);eligible=all(lookup[k]['status']=='no_registered_geometry_intersection' for k in members)
  records.append({'group_id':hashlib.sha256('\n'.join(members).encode()).hexdigest(),'members':members,'representative':members[0],
   'metadata_spatial_eligible':eligible,'confirmation_ready':False,'reason':'no registered geometry intersection' if eligible else 'at least one related member has historical geometry flag',
   'positive_negative_support':'unopened_unknown','original_L1_label_grid':'unverified','selection':'lexicographic ID; no pixels, labels or model scores used'})
 eligible=[g for g in records if g['metadata_spatial_eligible']]
 (OUT/'L7_CANDIDATE_LOCK.json').write_text(json.dumps({'status':'metadata_only_not_performance_preregistration','rule':'same-day same-WRS-path OR footprint distance<=2000m, transitive before exclusion; full group excluded if any member flags history','groups':records},indent=2),encoding='utf-8')
 with (OUT/'L7_PRIORITY_REPRESENTATIVES.csv').open('w',encoding='utf-8-sig',newline='') as f:
  keys=['group_id','scene_id','path','row','date','original_archive_url','members','annotation_coverage','confirmation_ready']
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for g in eligible:
   r=meta[g['representative']];w.writerow({'group_id':g['group_id'],'scene_id':r['id'],'path':r['path'],'row':r['row'],'date':r['date'],'original_archive_url':r['download_url'].replace('http:','https:',1),'members':json.dumps(g['members']),'annotation_coverage':'official Shadows=yes; pixels/support unopened','confirmation_ready':False})
 # Probe three fixed representative archive headers only. Never consume a body.
 probes=[meta[g['representative']]['download_url'].replace('http:','https:',1) for g in eligible[:3]]
 def head(url):
  try:
   with urllib.request.urlopen(urllib.request.Request(url,method='HEAD',headers={'User-Agent':'Mozilla/5.0 research-metadata-audit'}),timeout=20) as r:
    return {'url':url,'status':r.status,'final_url':r.url,'content_type':r.headers.get('Content-Type'),'content_length':r.headers.get('Content-Length'),'body_read':False}
  except Exception as e:return {'url':url,'error':str(e),'body_read':False}
 with ThreadPoolExecutor(max_workers=3) as ex:access=list(ex.map(head,probes))
 (OUT/'ARCHIVE_HEADER_PROBES.json').write_text(json.dumps(access,indent=2),encoding='utf-8')
 summary={'L7_official_total':206,'L7_official_shadow_yes':45,'L7_geometric_no_intersection_scenes':sum(r['status']=='no_registered_geometry_intersection' for r in lookup.values() if r.get('pool')=='L7_Irish'),
  'L7_spatially_eligible_groups':len(eligible),'L7_spatially_eligible_members':sum(len(g['members']) for g in eligible),
  'L7_all_related_groups':len(records),'L7_blocked_groups':len(records)-len(eligible),
  'confirmation_ready_groups':0,'new_training':0,'new_forward':0,'image_or_label_pixels_read':0,'image_bytes_downloaded':0,
  'remaining_L8_candidates':[r['id'] for r in lookup.values() if r.get('pool')=='prior_L8' and r['status']=='no_registered_geometry_intersection'],
  'environment':'cloud-lite-pt210; existing Shapely2.1.2 / rasterio1.3.11 / GDAL3.9.2; pyproj unavailable, not installed',
  'unresolved_registered_history_geometries':max(r.get('unresolved_history_geometries',0) for r in lookup.values()),
  'archive_body_requests':0,'archive_header_probes':len(probes)}
 (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 checks={'official_206_and_45':len(rows)==45 and len(list(csv.DictReader((OUT/'IRISH_ALL_SCENES.csv').open(encoding='utf-8-sig'))))==206,
 'all45_have_unique_STAC_match':all(r['stac_items']=='1' for r in rows),'45_partitioned_once':len([k for g in records for k in g['members']])==len({k for g in records for k in g['members']})==45,
 'no_flagged_member_in_retained_group':all(lookup[k]['status']=='no_registered_geometry_intersection' for g in eligible for k in g['members']),
 'representatives_lexicographic':all(g['representative']==min(g['members']) for g in records),'Source_geometry_9025':result['source_geometry_records']==9025,
 'original_two_L8_candidates_remain':set(summary['remaining_L8_candidates'])=={'LC81970182015080LGN00','WHU_124033_2018-04-15'},
 'no_remote_writes_or_pixel_reads':result['remote_files_written']==result['image_or_label_payloads_opened']==result['new_model_calls']==0}
 (OUT/'VALIDATION.json').write_text(json.dumps({'passed':all(checks.values()),'checks':checks},indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));print(json.dumps(access,indent=2));print('checks_passed',all(checks.values()))

if __name__=='__main__':main()

"""Allowlisted coordinate projection; never interpret categorical reference fields."""
from pathlib import Path
import csv, io, json, zipfile, collections, math

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase244/phase244_20261031'

def main():
 candidates=list(csv.DictReader((OUT/'CANDIDATES.csv').open(encoding='utf-8-sig')))
 bykey={r['point_product_id']:r for r in candidates if r['pool']=='PixBox_L8'}
 h=json.loads((ROOT/'outputs/phase102/phase102_20261013/inputs/history_positions_only.json').read_text())['records']
 points=[];counts=collections.Counter();flags=collections.Counter();unknown=[]
 with zipfile.ZipFile(OUT/'PixBox-L8-CMIX.zip') as z:
  n=next(n for n in z.namelist() if n.lower().endswith('.csv'))
  with z.open(n) as f:
   reader=csv.reader(io.TextIOWrapper(f,encoding='utf-8-sig'));header=next(reader)
   allow=['ID','PRODUCT_ID','PIXEL_X','PIXEL_Y','LATITUDE','LONGITUDE'];idx=[header.index(k) for k in allow]
   for raw in reader:
    p=dict(zip(allow,(raw[i] for i in idx)));key=p['PRODUCT_ID'];counts[key]+=1
    if key not in bykey:unknown.append(key);continue
    lat=float(p['LATITUDE']);lon=float(p['LONGITUDE']);dx=min(180,2/(111*max(.01,math.cos(math.radians(lat)))));dy=2/110
    hits=[]
    for old in h:
     b=old.get('wgs84_bounds')
     if b and (b[0]>b[2] or b[0]-dx<=lon<=b[2]+dx) and b[1]-dy<=lat<=b[3]+dy:hits.append(old)
    p['product']=bykey[key]['id'];p['conservative_history_position_flag']=bool(hits);p['matched_roles']=';'.join(sorted({x['role'] for x in hits}))
    flags[key]+=bool(hits);points.append(p)
 with (OUT/'POSITIONS_ONLY.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
 summary={'points':len(points),'unique_products':len(counts),'unknown_product_keys':sorted(set(unknown)),
  'projected_fields':allow,'categorical_reference_fields_interpreted':False,'point_flags':sum(flags.values()),
  'by_product':[{'product':bykey[k]['id'],'points':v,'position_flags':flags[k]} for k,v in counts.items()],
  'note':'Coordinate-only conservative screen, not exact input-context footprint audit. No labels counted, no row removed or performance evaluated. Does not override whole-scene correlation policy.'}
 (OUT/'POSITION_SCREEN.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

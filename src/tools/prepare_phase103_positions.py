"""Derive16-product coordinate metadata only, on existing remote conda; never labels."""
import csv,hashlib,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'src/research_results/phase103_20261013'
OUT=Path('/home/scv/shared/phase103_positions_20261013')

def main():
    from rasterio.warp import transform
    members=json.loads((BASE/'member_grid_metadata.json').read_text());bylink={m['product_link']:m for m in members}
    prior=ROOT/'src/research_results/phase102_20261013/inputs/pixbox_positions_only.csv'
    positions=[];opaque=0
    with prior.open('rb') as f:
        header=f.readline().decode('utf-8-sig').strip().split(',');assert header[:2]==['point_id','product_link']
        for raw in f:
            link=raw.split(b',',2)[1].decode('ascii').strip()
            if link not in bylink:opaque+=1;continue
            positions.append(dict(zip(header,next(csv.reader([raw.decode().strip()])))))
    derived=[];checks=[]
    for link,m in sorted(bylink.items()):
        pts=[p for p in positions if p['product_link']==link]
        xx,yy=transform('EPSG:4326',m['crs'],[float(p['longitude']) for p in pts],[float(p['latitude']) for p in pts])
        reasons={};fractions=[]
        for p,x,y in zip(pts,xx,yy):
            g10,g20=m['grid10'],m['grid20'];pending=[]
            u10=(x-g10['ULX'])/g10['XDIM'];v10=(y-g10['ULY'])/g10['YDIM']
            u20=(x-g20['ULX'])/g20['XDIM'];v20=(y-g20['ULY'])/g20['YDIM']
            col10,row10=math.floor(u10),math.floor(v10);col20,row20=math.floor(u20),math.floor(v20)
            px,py=float(p['pixel_x']),float(p['pixel_y'])
            if not px.is_integer() or not py.is_integer():pending.append('noninteger_original_index')
            if col10!=px or row10!=py:pending.append('original_index_not_unique_zero_based_containing_pixel')
            if not(0<=col10<g10['NCOLS'] and 0<=row10<g10['NROWS']):pending.append('outside_grid10')
            if not(0<=col20<g20['NCOLS'] and 0<=row20<g20['NROWS']):pending.append('outside_grid20')
            # Fixed0.02m coordinate-rounding/projection boundary allowance; never search shifts.
            for axis,continuous,res in [('x10',u10,10),('y10',v10,10),('x20',u20,20),('y20',v20,20)]:
                if min(continuous%1,1-continuous%1)*res<=.02:pending.append('near_grid_boundary_'+axis)
            if col20!=col10//2 or row20!=row10//2:pending.append('inconsistent_nested_grids')
            for reason in pending:reasons[reason]=reasons.get(reason,0)+1
            fractions.append((u10%1,v10%1))
            derived.append({**p,'product_id':m['product_id'],'group_id':m['group_id'],'crs':m['crs'],
                'native_x_m':x,'native_y_m':y,'col10':col10,'row10':row10,'col20':col20,'row20':row20,
                'position_pending':bool(pending),'pending_reasons':'|'.join(pending),
                'cluster_1km':f"{m['crs']}:{math.floor(x/1000)}:{math.floor(y/1000)}"})
        checks.append({'product_id':m['product_id'],'points':len(pts),'pending_reasons':reasons,
            'min_fractional_x10':min(x[0] for x in fractions),'max_fractional_x10':max(x[0] for x in fractions),
            'min_fractional_y10':min(x[1] for x in fractions),'max_fractional_y10':max(x[1] for x in fractions),
            'index_rule':'floor of original XML native coordinates, exact original indices; no shift search'})
    OUT.mkdir(parents=True,exist_ok=True)
    file=OUT/'coordinate_metadata.csv'
    with file.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(derived[0]));w.writeheader();w.writerows(derived)
    report={'status':'passed_coordinate_metadata_only','products':len(members),'points':len(derived),
            'pending':sum(p['position_pending'] for p in derived),'checks':checks,
            'source_position_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),
            'member_grid_metadata_sha256':hashlib.sha256((BASE/'member_grid_metadata.json').read_bytes()).hexdigest(),
            'coordinate_metadata_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'class_values_read':False,'raster_opened':False}
    (OUT/'coordinate_metadata_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))

if __name__=='__main__':main()

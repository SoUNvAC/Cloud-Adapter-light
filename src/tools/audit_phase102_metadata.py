"""Phase102: CPU product/position/history audit; no raster, label, model or score I/O."""
import argparse,csv,hashlib,json,math,re
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'src/research_results/phase102_20261013'
INP=BASE/'inputs'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def csvread(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def components(ids,edges):
    parent={x:x for x in ids}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for a,b in edges:
        if a in parent and b in parent:
            x,y=find(a),find(b);parent[max(x,y)]=min(x,y)
    out=defaultdict(list)
    for x in sorted(ids):out[find(x)].append(x)
    return sorted(out.values())
def identities():
    rows=read(INP/'candidate_products_metadata.json');history=read(INP/'history_positions_only.json')['records']
    assert len({x['record_key'] for x in rows})==len(rows)
    assert len([x for x in rows if x['source']=='PixBox5036991'])==29
    patches=csvread(INP/'kappa_patch_positions.csv');assert len(patches)==9251
    assert len({x['record_key'] for x in patches})==1079
    points=csvread(INP/'pixbox_positions_only.csv');assert len(points)==17351
    assert set(points[0])=={'point_id','product_link','pixel_x','pixel_y','latitude','longitude'}
    links={x['product_link'] for x in rows if x['source']=='PixBox5036991'}
    assert {x['product_link'] for x in points}==links
    assert len({x['point_id'] for x in points})==len(points)
    for p in points:
        assert -180<=float(p['longitude'])<=180 and -90<=float(p['latitude'])<=90
    return rows,history,patches,points

def main(identity_only=False):
    rows,history,patches,points=identities()
    tilemap=defaultdict(list);idmap=defaultdict(list)
    for h in history:tilemap[h['tile']].append(h);idmap[h['product']].append(h)
    if identity_only:
        print(json.dumps({'status':'identity-only passed','candidate_records':len(rows),'history_records':len(history),
            'kappa_acquisition_keys_not_confirmed_unique_products':1079,'pixbox_products':29,
            'kappa_same_tile_excluded':sum(x['tile'] in tilemap for x in rows if x['source']=='KappaSet7100327'),
            'pixbox_same_tile_excluded':sum(x['tile'] in tilemap for x in rows if x['source']=='PixBox5036991'),
            'spatial_acceptance':'not performed locally; use existing remote conda geometry libraries'}));return
    # Phase86 helpers import vector coordinate transformations only; never call its main.
    from audit_phase86 import project_geometry,buffered_intersects
    from shapely.geometry import shape,box,Polygon
    from shapely.strtree import STRtree
    from rasterio.warp import transform
    import shapely,rasterio,numbers
    def dense(g,step):
        coords=list(g.exterior.coords);result=[]
        for a,b in zip(coords[:-1],coords[1:]):
            n=max(1,math.ceil(math.hypot(b[0]-a[0],b[1]-a[1])/step))
            result.extend((a[0]+(b[0]-a[0])*j/n,a[1]+(b[1]-a[1])*j/n) for j in range(n))
        return Polygon(result+[result[0]])
    def envelope(g):
        l,b,r,t=g.bounds;lat=.06;lon=min(180,.06/max(.001,math.cos(math.radians(min(89.99,max(abs(b),abs(t))+.06)))))
        return box(l-lon,b-lat,r+lon,t+lat)
    hg=[box(*h['wgs84_bounds']) for h in history if h['wgs84_bounds']]
    hh=[h for h in history if h['wgs84_bounds']];tree=STRtree(hg);lookup={id(g):i for i,g in enumerate(hg)}
    geo={};grid_checks=[];bylink=defaultdict(list)
    for p in points:bylink[p['product_link']].append(p)
    for r in rows:
        key=r['record_key'];r['spatial_unknown']=False;r['registration_metadata_pass']=False
        if r['grid']:
            s=r['grid'];native=box(s['ULX'],s['ULY']+s['NROWS']*s['YDIM'],s['ULX']+s['NCOLS']*s['XDIM'],s['ULY'])
            geo[key]=project_geometry(dense(native,250),'EPSG:'+r['crs'].split(':')[-1],'EPSG:4326')
            pts=bylink[r['product_link']];xx,yy=transform('EPSG:4326',r['crs'],[float(p['longitude']) for p in pts],[float(p['latitude']) for p in pts])
            residual=[];inside=[]
            for p,x,y in zip(pts,xx,yy):
                px,py=float(p['pixel_x']),float(p['pixel_y'])
                residual.append(math.hypot(x-(s['ULX']+px*s['XDIM']),y-(s['ULY']+py*s['YDIM'])))
                inside.append(0<=px<s['NCOLS'] and 0<=py<s['NROWS'] and native.buffer(15).covers(shape({'type':'Point','coordinates':[x,y]})))
            passed=all(inside) and max(residual)<=15
            r['registration_metadata_pass']=passed
            grid_checks.append({'product_id':key,'points':len(pts),'crs':r['crs'],
                'all_position_records_inside_parent_grid':all(inside),'max_coordinate_to_index_corner_residual_m':max(residual),
                'tolerance_m':15,'passed':passed,'scope':'coordinate/index metadata only; no imagery-label pixel registration or semantic quality claim'})
        elif r['geo']:
            g=shape(r['geo'])
            if g.is_valid and g.bounds[2]-g.bounds[0]<180:geo[key]=dense(g,.002)
        reasons=[];matches=[]
        if r['product_id'] in idmap:reasons.append('same_product');matches+=idmap[r['product_id']]
        if r['tile'] in tilemap:reasons.append('same_tile_any_date');matches+=tilemap[r['tile']]
        if key in geo:
            g=geo[key]
            if not g.is_valid or g.bounds[2]-g.bounds[0]>=180:r['spatial_unknown']=True
            else:
                crs=r['crs'] or f"EPSG:{32600+int(r['tile'][:2]) if r['tile'][2]>='N' else 32700+int(r['tile'][:2])}"
                for found in tree.query(envelope(g)):
                    j=int(found) if isinstance(found,numbers.Integral) else lookup[id(found)]
                    h=hh[j]
                    if hg[j].bounds[2]-hg[j].bounds[0]>=180:
                        r['spatial_unknown']=True;continue  # near Antarctic whole-longitude envelope: do not invent exact geometry
                    try:hit=buffered_intersects(g,dense(hg[j],.002),crs)
                    except Exception:r['spatial_unknown']=True;continue
                    if hit:reasons.append('both_1000m_buffers_intersect');matches.append(h)
        else:r['spatial_unknown']=True
        r['exclusion_reasons']=sorted(set(reasons))
        r['history_matches']=[{'product':p,'role':role} for p,role in sorted({(h['product'],h['role']) for h in matches})]
        if reasons:r['status']='excluded_overlap'
        elif r['metadata_complete'] and r['registration_metadata_pass'] and not r['spatial_unknown']:r['status']='candidate_metadata_pass'
        else:r['status']='pending_metadata'
        r['historical_access_status']='no use found in657 scoped metadata/code/plan files; unrecorded use and base pretraining unknown'
    edges=[]
    remaining=[r for r in rows if r['status']!='excluded_overlap']
    for i,a in enumerate(remaining):
        for b in remaining[i+1:]:
            why=[]
            if a['tile']==b['tile']:why.append('same_tile_any_date')
            if a['utc_day']==b['utc_day'] and a['orbit'] and a['orbit']==b['orbit']:why.append('same_UTC_day_and_orbit')
            if a['record_key'] in geo and b['record_key'] in geo:
                g1,g2=geo[a['record_key']],geo[b['record_key']]
                if envelope(g1).intersects(g2):
                    crs=a['crs'] or f"EPSG:{32600+int(a['tile'][:2]) if a['tile'][2]>='N' else 32700+int(a['tile'][:2])}"
                    if buffered_intersects(g1,g2,crs):why.append('both_1000m_buffers_intersect')
            if why:edges.append({'left':a['record_key'],'right':b['record_key'],'reasons':why})
    known_groups=[]
    for source in ['KappaSet7100327','PixBox5036991']:
        selected=[r for r in remaining if r['source']==source];ids=[r['record_key'] for r in selected]
        groups=components(ids,[(e['left'],e['right']) for e in edges])
        for members in groups:
            known_groups.append({'source':source,'group_id':hashlib.sha256('|'.join(members).encode()).hexdigest(),
                'members':members,'member_count':len(members),'metadata_representative_only':min(members),
                'connectivity_complete':all(r['metadata_complete'] and r['record_key'] in geo and not r['spatial_unknown'] and not r['parent_proxy'] for r in selected if r['record_key'] in members),
                'role_allocation':'none; not calibration/confirmation split'})
    index=[]
    for r in rows:index.append({k:r.get(k,'') for k in ['source','record_key','product_id','sensing_key','utc_day','tile','orbit',
        'location_evidence','footprint_precision','six_band_availability','processing_baseline','toa_basis',
        'historical_access_status','status','exclusion_reasons','history_matches','spatial_unknown']})
    counts={}
    for source in ['KappaSet7100327','PixBox5036991']:
        s=[r for r in rows if r['source']==source];gs=[g for g in known_groups if g['source']==source]
        counts[source]={'indexed_acquisition_or_product_records':len(s),'unique_products_verified':len(s) if source=='PixBox5036991' else None,
            'status_counts':dict(Counter(r['status'] for r in s)),'remaining_known_connectivity_components':len(gs),
            'remaining_complete_geometry_components':sum(g['connectivity_complete'] for g in gs),
            'date_range':[min(r['utc_day'] for r in s),max(r['utc_day'] for r in s)],'tiles':len({r['tile'] for r in s}),
            'independent_groups_certified':False,'shadow_group_support':'not read; not inferred from class names'}
    csvwrite(BASE/'product_location_index.csv',index)
    csvwrite(BASE/'overlap_exclusions.csv',[{k:r[k] for k in ['source','record_key','status','exclusion_reasons','history_matches']} for r in rows])
    csvwrite(BASE/'related_groups.csv',known_groups)
    write(BASE/'correlation_edges.json',edges)
    write(BASE/'coordinate_grid_checks.json',grid_checks)
    write(BASE/'metadata_audit.json',{'status':'completed_metadata_only','counts':counts,
        'Kappa_official_product_count':1038,'Kappa_archive_acquisition_keys':1079,'Kappa_identity_discrepancy_unresolved':True,
        'rules':{'buffer_each_m':1000,'contact':'planar distance<=2000m, including boundary touch',
            'projection':'candidate UTM; native edges densified<=250m; historical WGS84 envelopes densified<=.002degree',
            'grouping':'same tile OR same UTC day and known orbit OR buffered intersection; transitive components',
            'unknown_geometry':'pending, never assumed disjoint; Kappa component counts are incomplete-connectivity upper bounds'},
        'source_base_pretraining_independence':'unknown','registered_history_records':len(history),
        'history_roles':dict(Counter(h['role'] for h in history)),
        'new_training_fits_predictions_performance_evaluations':0,'raster_pixels_read':False,
        'raw_PixBox_CSV_bytes_tokenized_for_allowlisted_field_projection':True,
        'PixBox_label_file_obtained':True,'PixBox_label_values_semantically_parsed_counted_or_displayed':False,
        'imagery_or_mask_payload_downloaded':False,'new_role_split':False,
        'PixBox_full_scene_FPR_estimable':False,'PixBox_inclusion_probabilities_known':False,
        'runtime':{'shapely':shapely.__version__,'rasterio':rasterio.__version__},
        'input_SHA256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in INP.iterdir() if p.is_file()}})
    print(json.dumps(counts,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity-only',action='store_true');args=p.parse_args();main(args.identity_only)

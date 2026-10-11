"""ALCD metadata-only external-pool audit. No raster/pixel/model/network I/O.

Only existing JSON/CSV, small XML and vector shapefile metadata may be read.
Each raster is stat'ed, never opened. Results are exclusive-created in shared.
"""
import csv,hashlib,io,json,math,re,sys,zipfile
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict

REPO=Path('/home/scv/Cloud-Adapter-light')
SHARED=Path('/home/scv/shared')
DATA=SHARED/'data/sentinel2_cloud_mask_catalogue_4172871'
ALCD=SHARED/'data/sentinel2_alcd_1460961'
ROOT=SHARED/'phase86_20261011_run04'
PROV=REPO/'src/work_dirs/phase65a/source_provenance'
BIND=REPO/'src/research_plans/PHASE86_INPUT_BINDINGS_20261011.json'
BANDS=['B02','B03','B04','B08','B11','B12']
SOURCES=[]

def allowed(p):
    p=Path(p).resolve()
    if not any(p.is_relative_to(r) for r in [REPO,SHARED]):raise ValueError('Outside allowed roots '+str(p))
    return p

def sha(p):return hashlib.sha256(allowed(p).read_bytes()).hexdigest()

def read(p,role,expected=None):
    p=allowed(p);h=sha(p)
    if expected is not None and h!=expected:raise ValueError('Pinned metadata mismatch '+str(p))
    SOURCES.append(dict(path=str(p),sha256=h,bytes=p.stat().st_size,role=role))
    if p.suffix=='.json':return json.loads(p.read_text())
    if p.suffix=='.csv':
        with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
    return p.read_text()

def write(name,obj):
    with (ROOT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)

def csvwrite(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})

def canonical(p):
    p=Path(p.rstrip('/')).name.removesuffix('.SAFE')
    m=re.fullmatch(r'(S2[AB])_MSIL1C_(\d{8}T\d{6})_(N\d{4})_R(\d{3})_T(\d{2}[A-Z]{3})_(\d{8}T\d{6})',p)
    if not m:raise ValueError('Noncanonical product '+p)
    sat,t,baseline,orbit,tile,generation=m.groups()
    return dict(canonical_product=p,sensing_utc=datetime.strptime(t,'%Y%m%dT%H%M%S').isoformat()+'Z',utc_day=t[:8],relative_orbit=orbit,tile=tile,generation=generation,satellite=sat)

def original_match(old,tile,candidates):
    if '_OPER_PRD_' not in old:return [p for p in candidates if p.endswith('/'+old+'.SAFE/')]
    generation=old.split('_PDMC_')[1].split('_')[0]
    sensing=old.split('_V')[1].split('_')[0];orbit=re.search(r'_R\d{3}_',old).group(0)
    return [p for p in candidates if '_MSIL1C_'+sensing+'_' in p and orbit in p and '_T'+tile+'_' in p and p.endswith('_'+generation+'.SAFE/')]

def components(ids,edges):
    parent={x:x for x in ids}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for a,b in edges:
        x,y=find(a),find(b)
        if x!=y:parent[max(x,y)]=min(x,y)
    out=defaultdict(list)
    for x in sorted(ids):out[find(x)].append(x)
    return sorted(out.values())

def project_geometry(g,src,dst):
    from rasterio.warp import transform_geom
    from shapely.geometry import shape,mapping
    return shape(transform_geom(src,dst,mapping(g),precision=-1))

def buffered_intersects(g1,g2,crs):
    # Both1000m buffers including boundary touch, without circle approximation.
    return project_geometry(g1,'EPSG:4326',crs).distance(project_geometry(g2,'EPSG:4326',crs))<=2000.0

def geographic_envelope(bounds):
    from shapely.geometry import box,MultiPolygon
    left,bottom,right,top=bounds
    if not all(math.isfinite(x) for x in bounds) or not (-180<=left<=180 and -180<=right<=180 and -90<=bottom<top<=90):
        raise ValueError('Invalid geographic envelope')
    if left>right:return MultiPolygon([box(left,bottom,180,top),box(-180,bottom,right,top)])
    if right-left>=180:raise ValueError('Unresolved overwide envelope')
    return box(*bounds)

def main():
    # Existing project-local pyshp; do not install packages or change conda.
    sys.path.insert(0,str(REPO/'src/work_dirs/phase65a/geospatial_python'))
    import shapefile,shapely,rasterio
    from rasterio.crs import CRS
    from shapely import wkt
    from shapely.geometry import shape,box,mapping
    from shapely.strtree import STRtree
    ROOT.mkdir(exist_ok=False)
    b=read(BIND,'frozen local bindings')
    read(REPO/'src/research_plans/PHASE86_EXTERNAL_POOL_AUDIT_20261011.md','audit protocol',b['protocol_sha256'])
    def pinned(p,local,role):return read(p,role,b['local_inputs'][local]['sha256'])
    manifest=pinned(ALCD/'imagery/product_manifest.json','data/sentinel2_alcd_1460961/imagery/product_manifest.json','37 original product identity manifest')
    status=read(ALCD/'imagery/imagery_download_status.json','current remote verified object index')
    geom=pinned(ALCD/'imagery/geometry_audit.json','outputs/phase65/alcd_imagery/remote_geometry_audit.json','completed 37-scene six-band geometry report')
    ref=pinned(ALCD/'imagery/reference_disagreement_20261008.json','outputs/phase65/alcd_imagery/reference_disagreement_20261008.json','quality-only version comparison, not error rate')
    intake=read(ALCD/'run_logs/catalogue_audit.json','38 directory intake; LF/CRLF-only local byte difference independently resolved',b['intake_equivalence']['remote_sha256'])
    assert hashlib.sha256(json.dumps(intake,sort_keys=True,separators=(',',':')).encode()).hexdigest()==b['intake_equivalence']['semantic_sha256']
    oldover=pinned(ALCD/'run_logs/overlap_audit.json','outputs/phase65/alcd_download/overlap_audit.json','historical provisional ID-only overlap') if (ALCD/'run_logs/overlap_audit.json').exists() else None
    review=pinned(ALCD/'imagery/visual_review_v1/sample_manifest.json','outputs/phase65/alcd_visual_review_v1/sample_manifest.json','quality-review panels; no evidence of completed human judgments')
    assert manifest['resolved_scenes']==37 and geom['scenes_all_six_grids_match']==37
    fields=['size','md5_base64','sha256','generation','status'];bad=[]
    for name,v in b['local_object_index'].items():
        rv=status['files'].get(name,{})
        if any(rv.get(k)!=v[k] for k in fields):bad.append(dict(object=name,reason='index disagreement'))
        p=allowed(ALCD/'imagery'/rv.get('path','MISSING'))
        if not p.is_file() or p.stat().st_size!=v['size']:bad.append(dict(object=name,reason='remote stat disagreement'))
    if set(status['files'])!=set(b['local_object_index']):bad.append(dict(reason='object index membership mismatch'))
    write('integrity_check.json',dict(objects=len(b['local_object_index']),bytes=sum(v['size'] for v in b['local_object_index'].values()),mismatches=bad,local_stat=b['local_stat_check'],payload_rehashed=False,prior_geometry_sha256=sha(ALCD/'imagery/geometry_audit.json'),old_intake_byte_difference=b['intake_equivalence']))
    gaps=[];history=[];edges=[]
    if bad:gaps.append('Object index/stat mismatches: affected products remain pending; no overwrite or rehash attempted.')

    # Segment native straight edges before cross-projection: max250m segments.
    def dense(g,step=250):
        if g.geom_type=='Polygon':
            def ring(coords):
                out=[]
                for a,z in zip(coords[:-1],coords[1:]):
                    n=max(1,math.ceil(math.hypot(z[0]-a[0],z[1]-a[1])/step))
                    out.extend((a[0]+(z[0]-a[0])*i/n,a[1]+(z[1]-a[1])*i/n) for i in range(n))
                return out+[coords[-1]]
            from shapely.geometry import Polygon
            return Polygon(ring(list(g.exterior.coords)),[ring(list(r.coords)) for r in g.interiors])
        if g.geom_type=='MultiPolygon':
            from shapely.geometry import MultiPolygon
            return MultiPolygon([dense(x,step) for x in g.geoms])
        raise ValueError('Nonpolygon footprint')
    def to_wgs(g,crs):
        c=CRS.from_user_input(crs)
        if not c.is_projected or abs(c.linear_units_factor[1]-1)>1e-12:raise ValueError('Native projected metres required')
        g=project_geometry(dense(g),c,'EPSG:4326')
        if not g.is_valid or g.is_empty or g.bounds[2]-g.bounds[0]>180:raise ValueError('Invalid/antimeridian footprint; pending required')
        return g
    def add(product,tile,role,g,source,conservative=False):
        history.append(dict(product=product,tile=tile,role=role,geom=g,evidence=source,conservative_bounds=conservative))

    smeta=SHARED/'data/phase65_source_metadata'
    sm=read(smeta/'source_metadata_manifest.json','published Source train/val inventory')
    map_a=read(REPO/'src/work_dirs/phase65a/source_mapping/mapping_audit.json','actual Source PNG-to-product ordinal verification','c3c521a375c50f81ff0e58682621c7fcb72fc47472a7b37931c587036f12d0bb')
    assert map_a['status']=='verified_rgb_ordinal_mapping'
    mapped={(r['split'],r['png_ordinal']):r for r in map_a['mappings']}
    seen=set();source_products=set();source_roles=defaultdict(set)
    for split in ['train','val']:
        p=smeta/(split+'_metadata.csv');rows=read(p,'actual Source '+split,sm['splits'][split]['sha256'])
        assert len(rows)==map_a['splits'][split]['rows']
        for i,r in enumerate(rows):
            assert mapped[(split,i)]['product']==r['s2_id'] and mapped[(split,i)]['exact_rgb_match']
            c=canonical(r['s2_id']);source_products.add(c['canonical_product']);source_roles[c['canonical_product']].add(split);key=(r['s2_id'],r['proj_epsg'],r['proj_geometry'],split)
            if key not in seen:
                seen.add(key);add(c['canonical_product'],c['tile'],'Source_'+split,to_wgs(wkt.loads(r['proj_geometry']),int(r['proj_epsg'])),str(p))
    read(REPO/'src/configs/protocol/phase64_source_parent_rgb.py','Source actual training config')
    read(REPO/'src/work_dirs/phase64_day1/dataset_registry_audit.json','historical dataset registry, later SPARCS use covered separately')
    split=read(REPO/'src/work_dirs/phase65a/split_lock.json','original unchanged split metadata','a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b')
    roles={p:r['split'] for r in split['proposed_rows'] for p in r['products']}
    for phase in ['phase65a_explore_20261008','phase65b_explore_20261008','phase66_20261010/prepared','phase72_20261010/prepared']:
        p=DATA/phase/'manifest.json'
        if p.exists():read(p,'used Catalogue pool membership '+phase)
        else:gaps.append('Missing recorded used-pool manifest: '+str(p))
    for phase in ['phase66_20261010','phase72_20261010']:
        p=DATA/phase/'unseal_receipt.json'
        if p.exists():read(p,'already-used confirmation access metadata '+phase)
        else:gaps.append('Missing confirmation access receipt '+str(p))
    tags=read(DATA/'classification_tags.csv','Catalogue complete release IDs; unused products conservatively excluded too','46b88e07480f36a4d2fa27f01a1ee9987b1def7ad08f1eb721a88ef0b5288d25')
    # Read vector release only, not raster archive. Existing verified SHA binds it.
    zpath=DATA/'shapefiles.zip';zh=sha(zpath)
    SOURCES.append(dict(path=str(zpath),sha256=zh,bytes=zpath.stat().st_size,role='513 vector subscene footprints'))
    assert zh=='73ca5e675b6b885129f429988b14b37c06f309b51e991be122b8f01fce313d40'
    with zipfile.ZipFile(zpath) as z:
        names={Path(n).name:n for n in z.namelist()}
        for r in tags:
            product=r['scene'];c=canonical(product)
            try:
                parts={s:z.read(names[product+s]) for s in ['.shp','.shx','.dbf','.prj']}
                with shapefile.Reader(shp=io.BytesIO(parts['.shp']),shx=io.BytesIO(parts['.shx']),dbf=io.BytesIO(parts['.dbf'])) as reader:
                    assert len(reader.shapes())==1;g=shape(reader.shapes()[0].__geo_interface__)
                add(product,c['tile'],'Catalogue_'+roles.get(product,'unused_release_conservative'),to_wgs(g,parts['.prj'].decode()),str(zpath))
            except (KeyError,ValueError) as e:gaps.append('Catalogue footprint unresolved '+product+': '+str(e));add(product,c['tile'],'Catalogue_'+roles.get(product,'unused_release_conservative'),None,str(zpath))

    hist=read(PROV/'historical_overlap_v2.json','142 existing historical L8/SPARCS footprints; no raster headers reopened','5643bc9c3be55b8a9d467c6bed50d26f9cc4c702a745c9683e40e784c20a1ea5')
    assert not hist['unresolved'] and len(hist['historical_metadata'])==142
    hlookup={(r['domain'],r['scene']):r for r in hist['historical_metadata']}
    histused=set();sealed=set();manifest_sources=[]
    for p in sorted((REPO/'src/work_dirs').rglob('scene_disjoint_manifest.csv')):
        rows=read(p,'historical used/sealed IDs only; no associated pixels or predictions');manifest_sources.append(str(p))
        for r in rows:
            if 'new_split' not in r:continue
            domain='sparcs' if 'sparcs' in str(p).lower() else 'l8';key=(domain,r['scene'])
            if r['new_split'] in ['target_train','target_val']:histused.add(key)
            elif r['new_split']=='target_test':sealed.add(key)
    for key in sorted(histused|set(hlookup)):
        if key in hlookup:
            r=hlookup[key];bb=r['buffered_wgs84_bounds']
            # Existing report is an already100m-buffered geographic envelope.
            # Reusing its entire envelope and then another1000m is conservative.
            try:g=dense(geographic_envelope(bb),step=.002)
            except ValueError:g=None;gaps.append('Historical geographic envelope unresolved '+str(key))
            add(':'.join(key),'','historical_used',g,str(PROV/'historical_overlap_v2.json'),True)
        else:gaps.append('Used historical ID lacks existing footprint '+str(key));add(':'.join(key),'','historical_used',None,';'.join(manifest_sources))
    # Missing/unrecorded pretraining and manual reads cannot be certified here.
    limitations=['Backbone pretraining geographic lineage is unavailable; audit covers recorded project-development pools, not all external pretraining.',
      'No retrospective metadata scan can exclude undocumented manual/model use. No claim of never opened.',
      'Old sealed IDs only; their rasters/predictions/metrics were not opened. Sealed-only footprints are not reclassified as used-history.',
      'Full ALCD tile bounds and full513 Catalogue release are conservative exclusions; historical geographic envelopes already include100m.']
    if any(h['geom'] is None for h in history):gaps.append('At least one historical footprint unresolved; spatial nonoverlap cannot be complete.')

    # Exposure scan reads explicit manifests/indexes/locks, not score arrays or
    # metrics reports. No training console logs: they can include sealed metrics.
    scan={};scanerrors=[];hits=defaultdict(list)
    aliases={s['key']:set([s['parameters']['cloudy_product_name'],canonical(s['gcs_product_prefix'])['canonical_product'],s['key']]) for s in manifest['scenes']}
    model_roots=[DATA,REPO/'src/work_dirs']
    patterns=['*manifest*.json','*index*.json','*training_lock*.json','*input*lock*.json','unseal_receipt.json','*resolved_config*.py']
    scanpaths=set()
    for base in model_roots:
        for pat in patterns:scanpaths.update(base.rglob(pat))
    for p in sorted(scanpaths):
        if any(t in str(p).lower() for t in ['sealed','human_confirmation','visual_review','semantic_coverage','pixel_locator']):continue
        try:
            if p.stat().st_size>20_000_000:scanerrors.append(dict(path=str(p),reason='metadata file size guard'));continue
            text=allowed(p).read_text();scan[str(p)]=sha(p)
            for key,aa in aliases.items():
                matched=sorted(a for a in aa if a in text)
                if matched:
                    quality='source_provenance' in str(p) or 'source_mapping' in str(p)
                    if not quality:hits[key].append(dict(path=str(p),sha256=scan[str(p)],matched_aliases=matched,status='recorded_model_pool_or_cache_hit_requires_role_review'))
        except (OSError,ValueError,UnicodeError) as e:scanerrors.append(dict(path=str(p),reason=str(e)))
    if scanerrors:gaps.append('Exposure metadata scan errors; see exposure_scan.json')
    write('exposure_scan.json',dict(scanned_sha256=scan,errors=scanerrors,hits=hits,pixels_read=False,logs_not_read_reason='avoid sealed metric access',retrospective_manual_use='unknown',quality_evidence=[str(ALCD/'imagery/geometry_audit.json'),str(ALCD/'imagery/reference_disagreement_20261008.json'),str(ALCD/'imagery/visual_review_v1/sample_manifest.json')]))

    gs={r['key']:r for r in geom['scenes']};rs={r['key']:r for r in ref['scenes']}
    products=[];native={};wgs={};badnames={x.get('object') for x in bad}
    for s in manifest['scenes']:
        key=s['key'];old=s['parameters']['cloudy_product_name'];tile=s['parameters']['tile'];c=canonical(s['gcs_product_prefix']);g=gs[key]
        matches=original_match(old,tile,s['gcs_candidates']);identity_ok=matches==[s['gcs_product_prefix']]
        assert c['tile']==tile
        image_bounds={tuple(g['bands'][band]['bounds']) for band in BANDS}
        geometry_ok=g['alignment_status']=='all_six_grids_match' and image_bounds=={tuple(g['bounds'])} and g['resolution']==[60.0,60.0]
        product=c['canonical_product'];native[product]=box(*g['bounds']);wgs[product]=to_wgs(native[product],g['crs'])
        p=dict(**c,original_product=old,key=key,directories=[str(Path(n).parent.parent) for n in s['label_members']],identity_status='strict_original_match' if identity_ok else 'pending_ambiguous',
          crs=g['crs'],label_bounds=g['bounds'],image_bounds=list(next(iter(image_bounds))),wgs84_bounds=list(wgs[product].bounds),label_resolution_m=60,
          geometry_status='passed_existing_six_band_grid_audit' if geometry_ok else 'pending',semantic_review_status='paused_incomplete',
          original_label_sha256=g['label_sha256'],original_label_class_counts_existing=g['label_class_counts'],joint_valid_cells_existing=g['joint_valid_pixels_60m'],shadow_joint_valid_existing=g['shadow_pixels_joint_valid_60m'],
          revised_reference_present=key in rs,revised_label_sha256=rs.get(key,{}).get('revised_label_sha256'),
          development_exposure_status='unknown_recorded_model_pool_hit' if hits[key] else ('unknown_scan_errors' if scanerrors else 'no_recorded_model_development_use_conditional'),
          undocumented_use_status='unknown',quality_review_status='geometric_and_version_audit_and_panel_delivery_seen; human_judgments_not_certified',
          fit_or_selection_exposure='unknown' if hits[key] else 'not_found_in_scanned_records',performance_seen='unknown' if hits[key] else 'not_found_in_scanned_records',
          object_status='pending' if any(o['name'] in badnames for o in s['objects']) else 'verified_index_and_stat',
          evidence_geometry=str(ALCD/'imagery/geometry_audit.json'),evidence_identity=str(ALCD/'imagery/product_manifest.json'),evidence_exposure='exposure_scan.json',confirmation_ready=False)
        products.append(p)
        # Positive direct-use evidence overrides a negative retrospective scan.
        if product in source_roles:
            p['development_exposure_status']='recorded_Source_'+','.join(sorted(source_roles[product]))
            p['fit_or_selection_exposure']='recorded_Source_'+','.join(sorted(source_roles[product]))
            p['performance_seen']='recorded_Source_val' if 'val' in source_roles[product] else 'not_found_in_scanned_records'
        if product in roles:
            p['development_exposure_status']='recorded_Catalogue_'+roles[product]
            p['fit_or_selection_exposure']='recorded_fit' if roles[product]=='fit' else ('recorded_selection' if roles[product]=='development_val' else 'not_fit_selection_by_split')
            p['performance_seen']='recorded_already_used_confirmation' if 'confirmation' in roles[product] or roles[product]=='65c_final' else 'recorded_development' if roles[product]=='development_val' else 'not_found_in_scanned_records'
    by={p['canonical_product']:p for p in products}

    validhist=[h for h in history if h['geom'] is not None]
    tree=STRtree([h['geom'] for h in validhist]);indices={id(h['geom']):i for i,h in enumerate(validhist)}
    def search_envelope(g):
        left,bottom,right,top=g.bounds
        # Loose5km WGS84 search envelope only; final decisions are native UTM.
        latmargin=.06;lonmargin=min(180,.06/max(.001,math.cos(math.radians(min(89.99,max(abs(bottom),abs(top))+latmargin)))))
        return box(left-lonmargin,bottom-latmargin,right+lonmargin,top+latmargin)
    def nearby(g):
        found=tree.query(search_envelope(g))
        import numbers
        return [validhist[int(v) if isinstance(v,numbers.Integral) else indices[id(v)]] for v in found]
    def intersects(g1,g2,crs):
        # Buffer both footprints1000m, include touching boundaries. Resolution64
        # approximates circles; use exact distance<=2000 instead, no arc polygon.
        return buffered_intersects(g1,g2,crs)
    for p in products:
        product=p['canonical_product'];hitreasons=set()
        for h in history:
            why=[]
            if h['product']==product:why.append('same_product')
            if h['tile'] and h['tile']==p['tile']:why.append('same_tile_any_date')
            if why:
                hitreasons.update(why);edges.append(dict(left=product,right=h['product'],right_role=h['role'],edge_type='history_exclusion',reasons=why,evidence=h['evidence']))
        for h in nearby(wgs[product]):
            if intersects(wgs[product],h['geom'],p['crs']):
                hitreasons.add('buffered_footprint_intersection');edges.append(dict(left=product,right=h['product'],right_role=h['role'],edge_type='history_exclusion',reasons=['buffered_footprint_intersection'],evidence=h['evidence']))
        p['exclusion_reasons']=sorted(hitreasons)
        pending=not p['identity_status'].startswith('strict') or p['object_status']=='pending' or any(h['geom'] is None for h in history)
        p['spatial_status']='excluded_overlap' if hitreasons else ('pending_metadata' if pending else 'passed_recorded_project_pools')
    candidates=sorted(p['canonical_product'] for p in products if p['spatial_status']=='passed_recorded_project_pools')
    pairs=[]
    for i,a in enumerate(candidates):
        for z in candidates[i+1:]:
            aa,zz=by[a],by[z];why=[]
            if aa['tile']==zz['tile']:why.append('same_tile')
            if (aa['utc_day'],aa['relative_orbit'])==(zz['utc_day'],zz['relative_orbit']):why.append('same_UTC_day_and_relative_orbit')
            # Candidate pairs are small; distance is computed in left product UTM.
            # Avoid antipodal projected geometry; rough geographic proximity only
            # rejects pairs separated by>5degrees latitude, never an overlap.
            if search_envelope(wgs[a]).intersects(wgs[z]) and intersects(wgs[a],wgs[z],aa['crs']):why.append('buffered_footprint_intersection')
            if why:pairs.append((a,z));edges.append(dict(left=a,right=z,right_role='ALCD_candidate',edge_type='candidate_correlation',reasons=why,evidence='product_manifest.json + existing geometry_audit.json'))
    groups=[]
    for members in components(candidates,pairs):
        gid=hashlib.sha256(('phase86:'+ '|'.join(members)).encode()).hexdigest();representative=min(members)
        status='unknown_recorded_hit' if any(by[p]['development_exposure_status'].startswith('unknown') for p in members) else 'no_recorded_model_development_use_conditional'
        row=dict(group_id=gid,members=members,representative=representative,tiles=sorted({by[p]['tile'] for p in members}),spatial_status='passed_recorded_project_pools',development_exposure_status=status,
           geometry_status='passed_existing_six_band_grid_audit' if all(by[p]['geometry_status'].startswith('passed') for p in members) else 'pending',semantic_review_status='paused_incomplete',confirmation_ready=False,
           representative_shadow_joint_valid_existing=by[representative]['shadow_joint_valid_existing'],representative_joint_valid_existing=by[representative]['joint_valid_cells_existing'],support_counts_not_used_for_selection=True)
        groups.append(row)
        for product in members:by[product]['group_id']=gid;by[product]['fixed_representative']=representative
    csvwrite('products.csv',products);csvwrite('groups.csv',groups);csvwrite('overlap_edges.csv',edges)
    history_counts=Counter(h['role'] for h in history)
    write('history_sources.json',dict(sources=SOURCES,history_roles=dict(history_counts),unique_source_products=len(source_products),historical_used_ids=len(histused),sealed_ids_only=[list(x) for x in sorted(sealed)],
      missing_evidence=gaps,limitations=limitations,metadata_scan_files=len(scan),all_history_raster_pixels_read=False,training_predictions_read=False,
      historical_footprints=[dict(product=h['product'],tile=h['tile'],role=h['role'],evidence=h['evidence'],wgs84_bounds=list(h['geom'].bounds) if h['geom'] else None,conservative_envelope=h['conservative_bounds']) for h in history]))
    counts=dict(products=len(products),directories=sum(len(p['directories']) for p in products),tiles=len({p['tile'] for p in products}),
      spatial_excluded=sum(p['spatial_status']=='excluded_overlap' for p in products),spatial_pending=sum(p['spatial_status']=='pending_metadata' for p in products),spatial_candidate_products=len(candidates),candidate_related_groups=len(groups),confirmation_ready_groups=0)
    rules=dict(buffer_each_footprint_m=1000,boundary_contact='intersects, including touch; exact planar distance<=2000',metric_crs='ALCD native UTM of left/candidate product; WGS84 STRtree is only broadphase; native edges densified<=250m before reprojection',
      historical_bounds='reuse entire existing100m-buffered WGS84 envelope, then1000m in candidate UTM; wrapped longitude split at+/-180; densify geographic edges<=.002degree; conservative',correlation='same tile OR same UTC day and relative orbit OR both1000m-buffered footprints intersect; transitive components',representative='lexicographically smallest canonical product ID',
      full_release_exclusion='all513 Catalogue products conservatively included; unused membership not claimed as used',no_shadow_selection=True)
    write('candidate_lock.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),git_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),counts=counts,rules=rules,groups=groups,
      original_labels=dict(invalid=[0,1],Cloud=[2,3],Shadow=[4],Surface=[5,6,7],native_resolution_m=60,confidence_is_accuracy=False,revised_is_truth=False),
      missing_evidence=gaps,limitations=limitations,model_prediction_authorized=False,training_authorized=False,new_downloads=0,pixels_read=False,semantic_review_resumed=False,
      input_bindings_sha256=sha(BIND),source_metadata_sha256={x['path']:x['sha256'] for x in SOURCES}))
    unresolved=['# Phase86 未解决事项','', '1. 独立人工语义复核仍暂停、未完成；原60m半自动标签与公开修订版均不是已裁决真值。不能自动触发模型评价。',
      '2. 未记录的人工/模型使用无法由事后扫描排除；本轮只报告记录内未发现。人工复核面板交付不等于实际判读完成。',
      '3. 骨干预训练地理谱系不可得；空间通过仅对列出的项目历史池成立。',
      '4. 原sealed只读清单ID，不读像元、预测、指标；未取得其足迹不改称已使用历史。',
      '5. 下一协议须冻结外部标签版本、有效mask、尺度、模型清单、终点及证据规则；本轮零训练/推理/新影像下载。', '', '## 具体元数据缺项']
    unresolved+=['- '+g for g in gaps] or ['- 已列项目池内未发现待补足的足迹；上列未记录使用/预训练谱系/语义复核限制仍保留。']
    (ROOT/'unresolved.md').write_text('\n'.join(unresolved)+'\n',encoding='utf-8')
    lines=['# Phase86：ALCD外部候选池审计','', '## 1 数据结果','',json.dumps(counts,ensure_ascii=False), '',
      '37唯一产品、38目录、15tiles分别计数；当前完整397对象/六波段及XML状态优先于早期“影像缺失”记录。索引五字段及双机文件长度复核，不重读17GB影像。',
      '', '空间通过产品按同tile/同日同轨/缓冲交叠传递分组，代表仅按规范产品ID字典序选择。无影成员保留，既有支持数不参与筛选。',
      '', '|相关组代表|产品数|tile|旧审计代表有效云影格数|空间|开发暴露|语义|', '|---|---:|---|---:|---|---|---|']
    for g in groups:lines.append(f"|{g['representative']}|{len(g['members'])}|{','.join(g['tiles'])}|{g['representative_shadow_joint_valid_existing']}|通过已登记项目池|{g['development_exposure_status']}|暂停/未完成|")
    lines+=['', '## 2 结论探讨','', '空间与记录内暴露检查只能产生外部候选，不能宣称确认就绪或正式有效样本量。语义尚未独立裁决，确认就绪组数为0。本轮不运行任何模型，也不恢复人工判读。', '',
      'Source实际训练/开发映射、Catalogue原fit/development及已用确认池、历史L8/SPARCS元数据来源与SHA见history_sources.json。全513 Catalogue发布产品保守排除包含未使用产品，故可能过度排除。历史使用不详/足迹缺项见unresolved.md，不用不同tile替代无空间交叠。', '',
      '置信图不是正确概率；版本差异不是错误率；原生60m不变成高分辨率真值。Phase85分枝继续关闭；本审计不授权推理或新训练。', '', '开发暴露扫描见exposure_scan.json；products/groups/overlap_edges.csv保留每个产品、组成员/代表、理由和证据路径。']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write('runtime.json',dict(python=sys.version,shapely=shapely.__version__,rasterio=rasterio.__version__,GDAL=rasterio.__gdal_version__,PROJ=rasterio.__proj_version__,code_sha256=sha(Path(__file__)),no_network_calls=True,no_raster_open=True))
    write('artifact_hashes.json',{p.name:sha(p) for p in ROOT.iterdir() if p.is_file()})
    print(json.dumps(dict(status='audit_delivered_not_confirmation_ready',counts=counts,missing=gaps),ensure_ascii=False),flush=True)

if __name__=='__main__':main()

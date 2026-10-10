"""Metadata-only geometry intake: bounded HTTP, exact L1C identity, no raster reads."""
import base64,csv,datetime as dt,hashlib,json,re,struct,urllib.parse,urllib.request,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from acquire_phase65_alcd_imagery import BUCKET,select_original,digest as mirror_digest
from prepare_phase65a_explore import SPLIT_SHA,digest

LIMIT=20_000_000
BANDS=['B04','B03','B02','B08','B11','B12']


def utc():return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def select(split,tags):
    assert len([r for r in split['proposed_rows'] if r['split']=='fit'])==153
    fit=[r for r in split['proposed_rows'] if r['split']=='fit']
    candidates=[dict(product=r['representative'],group_id=r['group_id'],
        public_shadow_percent=float(tags[r['representative']]['shadow_percent']),
        selection_sha256=hashlib.sha256(('phase75:'+r['representative']).encode()).hexdigest())
        for r in fit if float(tags[r['representative']]['shadow_percent'])>0]
    candidates.sort(key=lambda r:r['selection_sha256'])
    return dict(rule="SHA256('phase75:'+full_product_id), ascending, first3 original fit representatives with public shadow_percent>0",
        original_fit_groups=153,eligible_products=len(candidates),selected=candidates[:3],no_model_results_used=True)


class BudgetClient:
    def __init__(self,root):self.root=root;self.used=0;self.logs=[]
    def get(self,url,destination):
        # Only this public original archive; no credentials, JP2, SAFE or DEM downloads.
        parsed=urllib.parse.urlparse(url)
        if parsed.scheme!='https' or parsed.netloc!='storage.googleapis.com':raise ValueError('Non-public-archive URL')
        if '.jp2' in urllib.parse.unquote(parsed.path).lower():raise ValueError('Raster URL forbidden')
        if destination.exists():raise FileExistsError(destination)
        row=dict(url=url,requested_utc=utc(),destination=str(destination.relative_to(self.root)),received_bytes=0)
        self.logs.append(row)
        try:
            request=urllib.request.Request(url,headers={'Accept-Encoding':'identity'})
            with urllib.request.urlopen(request,timeout=45) as response:
                if urllib.parse.urlparse(response.url).netloc!='storage.googleapis.com':raise ValueError('Archive redirect outside allowlist')
                row.update(status=response.status,response_headers={k.lower():v for k,v in response.headers.items()},final_url=response.url)
                length=response.headers.get('Content-Length')
                if length and int(length)>LIMIT-self.used:raise ValueError('Response exceeds remaining 20MB budget')
                parts=[]
                while True:
                    remaining=LIMIT-self.used
                    if remaining<=0:raise ValueError('20MB download budget exhausted')
                    block=response.read(min(65536,remaining))
                    if not block:break
                    self.used+=len(block);row['received_bytes']+=len(block);parts.append(block)
                b=b''.join(parts)
                if length and len(b)!=int(length):raise ValueError('Returned Content-Length mismatch')
            destination.write_bytes(b);row.update(completed_utc=utc(),sha256=hashlib.sha256(b).hexdigest())
            return b,row
        except Exception as e:
            row.update(error=type(e).__name__+': '+str(e),completed_utc=utc());raise
    def listing(self,prefix,destination,delimiter=None):
        query={'prefix':prefix,'maxResults':1000}
        if delimiter:query['delimiter']=delimiter
        url='https://storage.googleapis.com/storage/v1/b/'+BUCKET+'/o?'+urllib.parse.urlencode(query)
        raw,row=self.get(url,destination);result=json.loads(raw)
        if result.get('nextPageToken'):raise ValueError('Unexpected listing pagination; stop, no expansion')
        return result
    def xml(self,obj,destination):
        if not obj['name'].endswith('.xml'):raise ValueError('Only original XML objects allowed')
        if int(obj['size'])>LIMIT-self.used:raise ValueError('XML exceeds remaining budget')
        url='https://storage.googleapis.com/'+BUCKET+'/'+urllib.parse.quote(obj['name'],safe='/')+'?generation='+obj['generation']
        raw,row=self.get(url,destination)
        md5,sha=mirror_digest(destination)
        if len(raw)!=int(obj['size']) or md5!=obj['md5Hash']:raise ValueError('Original XML length/MD5 mismatch')
        returned_generation=row['response_headers'].get('x-goog-generation')
        if returned_generation and returned_generation!=obj['generation']:raise ValueError('Pinned generation mismatch')
        row.update(expected_size=int(obj['size']),server_md5_base64=obj['md5Hash'],generation=obj['generation'],
                   verified_sha256=sha,verification='length + server MD5 + pinned generation + SHA256')
        return raw


def tag(e):return e.tag.split('}')[-1]
def nodes(root,name):return [e for e in root.iter() if tag(e)==name]
def text(root,name):
    x=nodes(root,name);return x[0].text.strip() if x and x[0].text else None


def parse_xml(product_bytes,tile_bytes,product,prefix,objects):
    a,b=ET.fromstring(product_bytes),ET.fromstring(tile_bytes)
    uri=text(a,'PRODUCT_URI');baseline=text(a,'PROCESSING_BASELINE');tile=product.split('_')[5][1:]
    product_name=prefix.rstrip('/').rsplit('/',1)[-1].removesuffix('.SAFE')
    identity=dict(catalogue_product_id=product,archive_product_id=product_name,xml_product_uri=uri,
        product_sensing_start_utc=text(a,'DATATAKE_SENSING_START'),product_name_sensing_start_utc=dt.datetime.strptime(product.split('_')[2],'%Y%m%dT%H%M%S').replace(tzinfo=dt.timezone.utc).isoformat(),
        tile_sensing_time_utc=text(b,'SENSING_TIME'),processing_baseline=baseline,tile=tile,
        tile_identifier=text(b,'TILE_ID'),granule_ids=[e.attrib.get('granuleIdentifier') for e in nodes(a,'Granule')],
        archive_prefix=prefix,processing_generation_token=product.split('_')[-1])
    if product_name!=product or uri not in [product,product+'.SAFE']:raise ValueError('Exact original product URI mismatch')
    if baseline is None or baseline.replace('.','')!=product.split('_')[3][1:]:raise ValueError('Product baseline mismatch')
    if not identity['tile_identifier'] or '_T'+tile+'_' not in identity['tile_identifier']:raise ValueError('Tile identity mismatch')
    mean=nodes(b,'Mean_Sun_Angle');solar=dict(level='original granule/tile metadata mean; not per-pixel angles',
        azimuth_definition='clockwise from local North at the node; zenith from ellipsoid normal; degrees',
        definition_source='https://sentiwiki.copernicus.eu/web/s2-processing',
        grid_north_conversion='not performed; geographic local North is not silently treated as image-row North')
    if mean:
        for name,out in [('AZIMUTH_ANGLE','azimuth'),('ZENITH_ANGLE','zenith')]:
            n=nodes(mean[0],name)
            solar[out]=float(n[0].text) if n else None;solar[out+'_unit']=n[0].attrib.get('unit') if n else None
    solar['angle_grids']=[]
    for grid in nodes(b,'Sun_Angles_Grid'):
        for axis in grid:
            values=nodes(axis,'VALUES');lines=[e.text.split() for e in values if e.text]
            solar['angle_grids'].append(dict(axis=tag(axis),rows=len(lines),columns=sorted(set(map(len,lines))),
                row_step=text(axis,'ROW_STEP'),col_step=text(axis,'COL_STEP'),
                row_step_unit=nodes(axis,'ROW_STEP')[0].attrib.get('unit') if nodes(axis,'ROW_STEP') else None,
                col_step_unit=nodes(axis,'COL_STEP')[0].attrib.get('unit') if nodes(axis,'COL_STEP') else None,
                values_level='coarse angle nodes; no pixel interpolation executed'))
    solar['grid_present']=bool(solar['angle_grids'])
    solar_ok=all(solar.get(k) is not None and np.isfinite(solar[k]) for k in ['azimuth','zenith'])
    solar_ok=solar_ok and all(solar.get(k+'_unit','').lower() in ['deg','degree','degrees'] for k in ['azimuth','zenith'])
    if solar_ok and not (0<=solar['azimuth']<360 and 0<=solar['zenith']<=180):raise ValueError('Angle out of range')
    grid=dict(crs_name=text(b,'HORIZONTAL_CS_NAME'),crs_code=text(b,'HORIZONTAL_CS_CODE'),resolutions={})
    for size in nodes(b,'Size'):
        resolution=size.attrib['resolution'];position=[e for e in nodes(b,'Geoposition') if e.attrib.get('resolution')==resolution]
        if len(position)!=1:raise ValueError('Missing/ambiguous native resolution position')
        q=position[0];r={k:int(text(size,k)) for k in ['NROWS','NCOLS']}
        r.update({k:float(text(q,k)) for k in ['ULX','ULY','XDIM','YDIM']})
        r['bounds']=[r['ULX'],r['ULY']+r['YDIM']*r['NROWS'],r['ULX']+r['XDIM']*r['NCOLS'],r['ULY']]
        grid['resolutions'][resolution]=r
    grid['bands']={}
    files=[e.text.strip() for name in ['IMAGE_FILE','IMAGE_FILE_2A'] for e in nodes(a,name) if e.text]
    for band in BANDS:
        physical={band,'B'+str(int(band[1:]))}
        info=[e for e in nodes(a,'Spectral_Information') if e.attrib.get('physicalBand') in physical]
        resolution=text(info[0],'RESOLUTION') if len(info)==1 else None
        matches=[o['name'] for o in objects if '/IMG_DATA/' in o['name'] and o['name'].endswith('_'+band+'.jp2')]
        refs=[f for f in files if f.endswith('_'+band) or f.endswith('_'+band+'.jp2')]
        linked=len(matches)==len(refs)==1 and (prefix+refs[0]).removesuffix('.jp2')==matches[0].removesuffix('.jp2')
        if linked:
            granule=matches[0].split('/GRANULE/')[1].split('/')[0]
            if granule not in identity['granule_ids']:raise ValueError('Product XML to granule chain mismatch')
        grid['bands'][band]=dict(native_resolution_m=resolution,grid=grid['resolutions'].get(resolution),
            product_xml_references=refs,archive_objects=matches,identity_chain_verified=linked,raster_downloaded=False)
    grid_ok=bool(grid['crs_code']) and all(str(v) in grid['resolutions'] for v in [10,20,60])
    grid_ok=grid_ok and all(v['grid'] is not None and v['identity_chain_verified'] for v in grid['bands'].values())
    for resolution,q in grid['resolutions'].items():
        if q['XDIM']!=int(resolution) or q['YDIM']!=-int(resolution) or min(q['NROWS'],q['NCOLS'])<=0:raise ValueError('Unexpected native grid orientation/spacing')
    radiometry=dict(quantification_value=text(a,'QUANTIFICATION_VALUE'),special_values={},reflectance_offsets=[])
    for parent in a.iter():
        children={tag(e):e.text for e in parent}
        if 'SPECIAL_VALUE_TEXT' in children and 'SPECIAL_VALUE_INDEX' in children:radiometry['special_values'][children['SPECIAL_VALUE_TEXT']]=int(children['SPECIAL_VALUE_INDEX'])
    radiometry['reflectance_offsets']=[dict(value=e.text,attributes=e.attrib) for e in nodes(a,'RADIO_ADD_OFFSET')]
    radiometry['offset_status']='present' if radiometry['reflectance_offsets'] else 'absent_in_exact_original_XML; no guessed field inserted'
    return dict(identity=identity,solar=solar,source_grid=grid,radiometry=radiometry,
                solar_metadata_verified=bool(solar_ok),source_grid_verified=bool(grid_ok))


def cube_chain(data,product):
    with zipfile.ZipFile(data/'subscenes.zip') as z:
        member='subscenes/'+product+'.npy'
        with z.open(member) as f:
            version=np.lib.format.read_magic(f)
            shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f))
            end=f.tell()
        with z.open(member) as f:header=f.read(end)
    with zipfile.ZipFile(data/'shapefiles.zip') as z:
        prefix='shapefiles/'+product+'/'+product
        shp=z.read(prefix+'.shp');prj=z.read(prefix+'.prj');bounds=list(struct.unpack('<4d',shp[36:68]));wkt=prj.decode().strip()
    zone=re.search(r'UTM_Zone_(\d+)([NS])',wkt)
    epsg=('EPSG:'+str((32600 if zone[2]=='N' else 32700)+int(zone[1]))) if zone else None
    extent=[bounds[2]-bounds[0],bounds[3]-bounds[1]];size=[shape[1]*20,shape[0]*20]
    fits=bool(np.allclose(extent,size,atol=1e-6,rtol=0))
    # This is a documented-padding hypothesis ONLY. No affine is assigned or tested.
    excess=[extent[0]/20-shape[1],extent[1]/20-shape[0]]
    candidate=dict(status='untested_candidate_not_a_verified_affine',basis='README 1152 padded scene, intended1024, final1022; centered65 pixel inset is plausible, not proven',
        inset_pixels_if_symmetric=[v/2 for v in excess],
        north_up_upper_left_if_symmetric=[bounds[0]+excess[0]/2*20,bounds[3]-excess[1]/2*20],
        row_col_order_and_resampling_origin='not verified; candidate assumes north-up and symmetric crop')
    return dict(cube_member=member,shape=list(shape),dtype=str(dtype),fortran_order=fortran,npy_header_only=True,
        npy_header_sha256=hashlib.sha256(header).hexdigest(),official_pixel_size_m=20,cube_width_height_m=size,
        annotation_crs_wkt=wkt,annotation_epsg=epsg,annotation_bounds=bounds,annotation_extent_m=extent,
        annotation_shp_sha256=hashlib.sha256(shp).hexdigest(),annotation_prj_sha256=hashlib.sha256(prj).hexdigest(),
        bounds_match_1022_at20m=fits,annotation_extent_pixels_at20m=[v/20 for v in extent],
        official_readme_evidence=dict(pdf_sha256=digest(data/'README.pdf'),pages=[1,4,6,9],
            pixel_size='20m',non20m_resampling='bilinear',TOA='original integer /10000',
            annotation_window=1152,padding_per_side=64,intended_export=1024,final_export=1022,
            export_bug='border unfilled; final crop removes border',exact_affine_or_library_parameters='not specified'),
        candidate_mapping=candidate,cube_mapping_status='pending',cube_affine_assigned=False,
        missing=['Exact original-to-published cube crop offset and pixel-center convention',
                 'Cube row/column orientation relative to native tile grid',
                 'Actual bilinear resampling implementation, grid origin and crop/resample order',
                 'Pixel consistency validation of a candidate mapping (not authorized this phase)'])

"""Three preselected original-fit products, XML-only intake, no image registration."""
import csv,hashlib,json,traceback
from pathlib import Path
from datetime import datetime,timezone
from prepare_phase65a_explore import SPLIT_SHA,digest
from phase75_metadata import LIMIT,BUCKET,BANDS,BudgetClient,select,select_original,parse_xml,cube_chain,write,utc

REPO=Path('/home/scv/Cloud-Adapter-light')
DATA=Path('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871')
ROOT=DATA/'phase75_20261010'
PROTOCOL=REPO/'src/research_plans/PHASE75_GEOMETRY_INTAKE_20261010.md'
CODE=['run_phase75.py','phase75_metadata.py','test_phase75.py','run_phase75.sh',
      'acquire_phase65_alcd_imagery.py','prepare_phase65a_explore.py']
PDF_SHA='fe30cf507184ee3f3cf16e16bce271b0701763d00144b138e0b3edf075e50ce6'
TAGS_SHA='46b88e07480f36a4d2fa27f01a1ee9987b1def7ad08f1eb721a88ef0b5288d25'


def main():
    if datetime.now(timezone.utc)>=datetime(2026,10,19,16,tzinfo=timezone.utc):raise RuntimeError('Research budget expired')
    ROOT.mkdir(exist_ok=False)
    split_path=REPO/'src/work_dirs/phase65a/split_lock.json'
    assert digest(split_path)==SPLIT_SHA and digest(DATA/'README.pdf')==PDF_SHA and digest(DATA/'classification_tags.csv')==TAGS_SHA
    split=json.loads(split_path.read_text());tags={r['scene']:r for r in csv.DictReader((DATA/'classification_tags.csv').open())}
    selection=select(split,tags);write(ROOT/'selected_products.json',selection)
    write(ROOT/'input_lock.json',dict(created_utc=utc(),protocol_sha256=digest(PROTOCOL),original_split_sha256=SPLIT_SHA,
        official_pdf_sha256=PDF_SHA,official_tags_sha256=TAGS_SHA,selection_sha256=digest(ROOT/'selected_products.json'),
        code_sha256={n:digest(Path(__file__).with_name(n)) for n in CODE},public_archive=BUCKET,
        max_new_http_bytes=LIMIT,raster_downloads=0,new_network_forwards=0,new_fits=0,new_method_evaluations=0,
        no_confirmation_pixels=True,no_cube_affine_assignment=True))
    client=BudgetClient(ROOT);audits=[]
    for item in selection['selected']:
        product=item['product'];folder=ROOT/product;folder.mkdir();chain=cube_chain(DATA,product)
        row=dict(**item,cube_chain=chain,solar_metadata_verified=False,source_grid_verified=False,
            cube_mapping_status='pending',direction_geometry_ready=False,missing=[])
        try:
            tile=product.split('_')[5][1:];date=product.split('_')[2][:8]
            prefix='tiles/'+tile[:2]+'/'+tile[2]+'/'+tile[3:]+'/'+product[:3]+'_MSIL1C_'+date
            listing=client.listing(prefix,folder/'candidate_listing.json',delimiter='/')
            candidates=listing.get('prefixes',[]);matched=select_original(product,tile,candidates)
            row['exact_product_candidates']=matched
            if len(matched)!=1:raise ValueError('Original product absent or ambiguous; no substitute')
            prefix=matched[0];listing=client.listing(prefix,folder/'object_listing.json');objects=listing.get('items',[])
            product_xml=[o for o in objects if o['name']==prefix+'MTD_MSIL1C.xml']
            tile_xml=[o for o in objects if '/GRANULE/' in o['name'] and o['name'].endswith('/MTD_TL.xml')]
            if len(product_xml)!=len(tile_xml) or len(product_xml)!=1:raise ValueError('Product/tile XML absent or ambiguous')
            p=client.xml(product_xml[0],folder/'MTD_MSIL1C.xml');t=client.xml(tile_xml[0],folder/'MTD_TL.xml')
            row.update(parse_xml(p,t,product,prefix,objects))
            if not row['solar_metadata_verified']:row['missing'].append('Mean solar angle field/unit missing or unverified')
            if not row['source_grid_verified']:row['missing'].append('Native grid or requested band reference chain missing/unverified')
            row['xml_object_metadata']=dict(product=product_xml[0],tile=tile_xml[0])
            row['annotation_matches_native_crs']=chain['annotation_epsg']==row['source_grid']['crs_code']
            if not row['annotation_matches_native_crs']:
                row['cube_mapping_status']='mismatch';row['missing'].append('Annotation/native CRS conflict')
            row['missing']+=chain['missing']
        except Exception as e:
            row['missing'].append(type(e).__name__+': '+str(e));row['intake_error']=traceback.format_exc()
            print('Phase75 metadata incomplete',product,str(e),flush=True)
        audits.append(row);write(folder/'metadata_audit.json',row)
        print('Phase75 audited',len(audits),len(selection['selected']),product,
            'solar',row['solar_metadata_verified'],'grid',row['source_grid_verified'],'cube',row['cube_mapping_status'],flush=True)
    report=dict(status='metadata_intake_complete',products=len(audits),rows=audits,
        solar_metadata_verified=sum(r['solar_metadata_verified'] for r in audits),source_grid_verified=sum(r['source_grid_verified'] for r in audits),
        cube_mapping_verified=sum(r['cube_mapping_status']=='verified' for r in audits),
        direction_geometry_ready=sum(r['direction_geometry_ready'] for r in audits),
        public_archive_response_bytes=client.used,download_budget_bytes=LIMIT,new_original_rasters_downloaded=0,
        new_predictions=0,new_fits=0,confirmation_pixels_read=False,input_lock_sha256=digest(ROOT/'input_lock.json'),
        original_failure_conclusions_unchanged=True)
    assert client.used<=LIMIT and report['products']<=3
    write(ROOT/'metadata_audit.json',report);write(ROOT/'request_log.json',client.logs)
    flat=[]
    for r in audits:
        solar=r.get('solar',{});identity=r.get('identity',{});chain=r['cube_chain']
        flat.append(dict(product=r['product'],group_id=r['group_id'],tile=r['product'].split('_')[5],
            granule_id=';'.join(identity.get('granule_ids',[])),tile_sensing_time_utc=identity.get('tile_sensing_time_utc'),
            processing_baseline=identity.get('processing_baseline'),solar_azimuth_deg=solar.get('azimuth'),solar_zenith_deg=solar.get('zenith'),
            angle_grid_present=solar.get('grid_present'),source_crs=r.get('source_grid',{}).get('crs_code'),
            cube_shape='x'.join(map(str,chain['shape'])),annotation_extent_m='x'.join(map(str,chain['annotation_extent_m'])),
            cube_extent_m='x'.join(map(str,chain['cube_width_height_m'])),solar_metadata_verified=r['solar_metadata_verified'],
            source_grid_verified=r['source_grid_verified'],cube_mapping_status=r['cube_mapping_status'],direction_geometry_ready=False,
            missing='; '.join(r['missing'])))
    with (ROOT/'metadata_audit.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]) if flat else ['product','solar_metadata_verified','source_grid_verified','cube_mapping_status','direction_geometry_ready']);w.writeheader();w.writerows(flat)
    write(ROOT/'minimum_pixel_validation_plan.json',dict(status='plan_only_not_executed',products=[r['product'] for r in audits],
        original_bands_minimum=['B04 native10m','B11 native20m'],rationale='one resampled10m band plus one unchanged20m band can distinguish crop/grid origin from interpolation; add other bands only in a separately authorized protocol',
        candidate_mapping='Original UTM tile grid; compare footprint direct extent (1152) versus symmetric65-pixel20m inset (1022); neither is accepted without original-pixel verification',
        steps=['Freeze exact product, original band hashes and native grid before reading pixels',
            'Use image bands only, no truth labels or model output, to compare candidate correspondence',
            'Compare unresampled20m B11 radiometry/valid mask at declared crop offsets; quantify exact agreement, absolute errors and spatial residuals',
            'Check B04 bilinear pixel-center convention, row/column orientation and crop/resample order against native10m data',
            'Use prespecified distributed interior/edge image-only samples; reject an ambiguous mapping instead of picking lowest residual without a frozen rule',
            'If mapping resolves, separately document geographic-North to UTM-grid/image direction conversion; no shadow projection or DEM acquisition in Phase75'],
        not_yet_frozen=['Pixel agreement tolerances based on stored precision','Candidate enumeration and tie/ambiguity rule','Distributed verification locations and minimum support'],
        labels_used_for_registration=False,new_raster_downloads_this_phase=0))
    write(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):digest(p) for p in ROOT.rglob('*') if p.is_file()})
    print('Phase75 complete',json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()

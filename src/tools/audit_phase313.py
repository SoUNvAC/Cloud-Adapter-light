"""One metadata/header inventory and bounded primary-literature acquisition. No pixel reads."""
import ast
from datetime import datetime,timezone
import os
import resource
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
from phase313_common import *

URLS=[
 ('ESA_ATBD','https://sentinels.copernicus.eu/documents/247904/446933/Sentinel-2-Level-2A-Algorithm-Theoretical-Basis-Document-ATBD.pdf'),
 ('COPERNICUS_PROCESSOR','https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/api/s2msi.projected_geometry_processor.l2a_scene_classification_processor.html'),
 ('FMASK2015','https://gerslab.cahnr.uconn.edu/wp-content/uploads/sites/2514/2021/06/Improvement-and-expansion-of-the-Fmask-algorithm-cloud-cloud-shadow-and-snow-detection-for-Landsats-4%E2%80%937-8-and-Sentinel-2-images.pdf')]
EVIDENCE=[];ACCESSES=[];INPUTS={};NETWORK=[];TOTAL=0;START=time.monotonic();MINMEM=10**20
def utc():return datetime.now(timezone.utc).isoformat()
def evidence(p,kind='metadata'):
    p=Path(p);assert p.is_file(),str(p)
    assert p.stat().st_size<12000000,str(p)
    h=sha(p);INPUTS[str(p)]=h
    EVIDENCE.append(dict(path=str(p),kind=kind,sha256=h,bytes=p.stat().st_size,read_utc=utc()))
    ACCESSES.append(dict(path=str(p),mode='metadata bytes; never array payload',utc=utc()))
    return p
def obj(p):return load(evidence(p))
def header(stream):
    prefix=stream.read(8);assert prefix[:6]==b'\x93NUMPY'
    nsize=2 if prefix[6]==1 else 4
    n=struct.unpack('<H' if nsize==2 else '<I',stream.read(nsize))[0];assert n<65536
    raw=stream.read(n);v=ast.literal_eval(raw.decode('latin1').strip())
    return dict(shape=list(v['shape']),dtype=v['descr'],fortran=v['fortran_order'],header_bytes=8+nsize+n,header_sha256=hashlib.sha256(prefix+struct.pack('<H' if nsize==2 else '<I',n)+raw).hexdigest())
def npy(p):
    p=Path(p)
    if not p.exists():return dict(status='missing',path=str(p))
    with p.open('rb') as f:h=header(f)
    ACCESSES.append(dict(path=str(p),mode='NPY header only',utc=utc(),**h))
    return dict(status='ready',path=str(p),**h)
def npz(p,names):
    p=Path(p)
    if not p.exists():return dict(status='missing',path=str(p))
    out={}
    with zipfile.ZipFile(p) as z:
        for name in names:
            if name+'.npy' in z.namelist():
                with z.open(name+'.npy') as f:out[name]=header(f)
    ACCESSES.append(dict(path=str(p),mode='NPZ central directory and selected NPY headers only; no reference arrays opened',utc=utc(),headers=out))
    return dict(status='ready' if all(n in out for n in names) else 'missing',path=str(p),headers=out)
def guard():
    global MINMEM
    m={k:int(v.split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())}
    MINMEM=min(MINMEM,m['MemAvailable'])
    assert m['MemAvailable']>=4000000000,'host free RAM stop line'
    assert shutil.disk_usage(ROOT).free>=4000000000,'disk stop line'
    assert time.monotonic()-START<3600,'one hour stop line'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*1024**3
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())*4+20000000<=300000000
def watch():
    while True:
        try:guard()
        except BaseException:
            write(ROOT/'RESOURCE_STOP.json',dict(utc=utc(),traceback=traceback.format_exc()));os._exit(74)
        time.sleep(1)
def fetch(name,url):
    global TOTAL
    import requests
    original=url
    for hop in range(6):
        guard();entry=dict(object=name,url=url,original_url=original,utc=utc(),redirect_hop=hop,response_body_bytes=0)
        try:
            with requests.get(url,stream=True,timeout=(15,35),allow_redirects=False) as response:
                entry.update(status=response.status_code,content_type=response.headers.get('Content-Type'),content_length=response.headers.get('Content-Length'),location=response.headers.get('Location'),last_modified=response.headers.get('Last-Modified'),etag=response.headers.get('ETag'))
                left=20000000-TOTAL
                if response.headers.get('Content-Length','').isdigit() and int(response.headers['Content-Length'])>left:
                    entry['failure']='declared response exceeds remaining 20MB budget; body not read';NETWORK.append(entry);return
                data=bytearray()
                for chunk in response.iter_content(chunk_size=16384):
                    assert TOTAL+len(chunk)<=20000000,'response budget stop';TOTAL+=len(chunk);entry['response_body_bytes']+=len(chunk);data.extend(chunk);guard()
                suffix='.pdf' if bytes(data[:4])==b'%PDF' else '.html'
                p=ROOT/'literature'/f'{name}_{hop}{suffix}';p.parent.mkdir(exist_ok=True);p.write_bytes(data)
                entry.update(path=str(p.relative_to(ROOT)),sha256=sha(p),body_complete=True)
                NETWORK.append(entry)
                if response.is_redirect:
                    from urllib.parse import urljoin
                    url=urljoin(url,response.headers['Location']);assert url.startswith('https://');continue
                if response.status_code!=200:entry['failure']='HTTP non-200; no source claims inferred'
                return
        except Exception:
            entry['failure']=traceback.format_exc();NETWORK.append(entry);return
    NETWORK[-1]['failure']='redirect limit; no source accepted'

def inventory():
    fit=obj(PL/'FIT_IMAGES.json')['rows'];dev=obj(PL/'EVAL_IMAGES.json')['rows'];assert len(fit)==100 and len(dev)==32
    assert len({v['product'] for v in fit+dev})==132
    assert INPUTS[str(PL/'FIT_IMAGES.json')]=='1a328d6e5b7d88fea3aa3dda89205d79ab1562c4fac76361fcc4e31219e26ea5'
    assert INPUTS[str(PL/'EVAL_IMAGES.json')]=='7bc7fd23b684f9ab8e7072cab119b1412a56971a2592ac791f580375f6e696a4'
    pseudo=obj(PL/'PSEUDO_LOCK.json');evaluation=obj(PL/'EVALUATION_LOCK.json')
    for n in ['EXECUTION_LOCK.json','ENDPOINT_LOCK.json','BACKUP_FILES.json']:obj(PL/n)
    assert pseudo['definition']=='Q90-PL=B@0.90'
    pp={v['product']:v for v in pseudo['rows']};ee={(v['product'],v['arm']):v for v in evaluation['rows']}
    p75=BASE/'phase75_20261010';old=obj(p75/'reparse01/metadata_audit.json')['rows'];mapping={v['product']:v for v in old}
    for path in sorted(p75.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.csv','.xml'):
            evidence(path,'old75 metadata/XML read-only')
    # Exact original product identity from actual preserved XML; never fetch XML.
    identities={}
    for product,v in mapping.items():
        def fields(p):
            tree=ET.parse(p);out={}
            for e in tree.iter():out.setdefault(e.tag.split('}')[-1],[]).append((e.text or '').strip())
            return out
        prod=fields(p75/product/'MTD_MSIL1C.xml');tile=fields(p75/product/'MTD_TL.xml')
        assert prod['PRODUCT_URI'][0]==product+'.SAFE'
        original_tree=ET.parse(p75/product/'MTD_MSIL1C.xml')
        granules=[e.attrib.get('granuleIdentifier') for e in original_tree.iter() if e.tag.split('}')[-1]=='Granule']
        assert tile['TILE_ID'][0] in granules
        assert tile['TILE_ID'][0]==v['identity']['tile_identifier']
        assert prod['PROCESSING_BASELINE'][0]==v['identity']['processing_baseline']
        identities[product]=dict(product_uri=prod['PRODUCT_URI'][0],tile_id=tile['TILE_ID'][0],processing_baseline=prod['PROCESSING_BASELINE'][0],solar=v['solar'],source_grid=v['source_grid'],identity=v['identity'],viewing_angle_grids_present=bool(tile.get('Viewing_Incidence_Angles_Grids')))
    write(ROOT/'EXACT_XML_IDENTITIES.json',identities)
    # Search existing metadata names only; never enter/read sealed pixel pools.
    xmls=[]
    for d in BASE.iterdir():
        if d.is_dir() and d.name.startswith('phase'):
            for path in d.rglob('*.xml'):xmls.append(str(path))
    write(ROOT/'EXISTING_XML_INDEX.json',dict(paths=xmls,scope=str(BASE),no_xml_downloads=True,no_pixel_pool_reads=True))
    for path in [Path('/home/scv/shared/phase309_20261025')/n for n in ['RESULT.md','SPATIAL_PROVENANCE.csv','SHAPEFILE_METADATA.json','INPUT_SHA256.json','SOURCE_RECORD.json','NETWORK_EVIDENCE.json','evidence/README_fulltext.txt']]:
        evidence(path,'Phase309 existing document provenance')
    for phase in ['phase70_20261010','phase74_20261010']:
        for n in ['input_lock.json','policy_lock.json','metadata_readiness.json','neighbor_audit.json','fit_audit.json']:
            path=BASE/phase/n
            if path.is_file():evidence(path,'historical metadata only')
    for path in sorted((S/'prior_inputs').glob('*')):evidence(path,'local original report byte-copy synchronized by git')
    rows=[];arrays=[]
    archive=Path(fit[0]['image_archive'])
    with zipfile.ZipFile(archive) as z:
        for pool,items in [('fit',fit),('development',dev)]:
            for item in items:
                guard();product=item['product'];tokens=product.split('_');v=mapping.get(product);ident=identities.get(product)
                assert Path(item['image_archive'])==archive
                with z.open(item['image_member']) as f:ih=header(f)
                assert ih['shape']==[1022,1022,13] and ih['dtype']=='<f4'
                ACCESSES.append(dict(path=str(archive)+'::'+item['image_member'],mode='NPY header only',utc=utc(),**ih))
                if pool=='fit':
                    p=pp[product];sp=Path(p['scores_path']);sv=PL/'teacher'/f'{product}.npz'
                    sh=npy(sp);vh=npz(sv,['pred','pseudo','maximum','valid'])
                    sourcehash=p['scores_sha256'];validhash=p['sha256'];studentpath=None;studenthash=None;studentstatus='missing: full-fit student cache not produced'
                else:
                    p=ee[(product,'Source')];sp=Path(p['cache_path']);sv=sp
                    sh=npz(sp,['scores','pred','valid']);vh=sh;sourcehash=p['sha256'];validhash=sourcehash
                    q=ee[(product,'LoRA')];studentpath=PL/'evaluation'/f'LoRA_{product}.npz';studenthash=q['sha256'];qh=npz(studentpath,['scores','pred','valid']);studentstatus=qh['status'];arrays.append(dict(product=product,kind='Q90_PL_student',**qh))
                arrays.extend([dict(product=product,kind='Source_scores',**sh),dict(product=product,kind='Source_valid',**vh)])
                rows.append(dict(product=product,group_id=item['group_id'],pool=pool,image_archive=str(archive),image_member=item['image_member'],image_member_sha256=item['image_member_sha256'],image_sha_evidence='inherited pinned Phase304 image manifest; not payload rehashed',image_header=ih,RGB_indices=[3,2,1],necessary_band_indices=[3,2,1,7,11,12],bands='B04 B03 B02 B08 B11 B12',Source_checkpoint_sha256=SOURCE,Source_scores_path=str(sp),Source_scores_sha256=sourcehash,Source_scores_status=sh['status'],Source_valid_path=str(sv),Source_valid_sha256=validhash,Source_valid_status=vh['status'],Q90_PL_prediction_scores_path=str(studentpath) if studentpath else None,Q90_PL_cache_sha256=studenthash,Q90_PL_cache_status=studentstatus,Q90_pseudo_path=str(sv) if pool=='fit' else None,cache_sha_evidence='inherited pinned historical locks; headers/existence checked now; no pixel SHA rehash',nine_windows='512x512; starts row,col in {0,256,510} squared; FP32 sum/count mean semantic scores; class order Surface Cloud Shadow',tile=tokens[5][1:],sensing_token=tokens[2],baseline_token=tokens[3],generation_token=tokens[6],exact_original_XML_status='ready' if ident else 'missing',exact_granule=ident['tile_id'] if ident else None,exact_sensing_time=ident['identity']['tile_sensing_time_utc'] if ident else None,exact_processing_baseline=ident['processing_baseline'] if ident else None,XML_source=str(p75/product) if ident else None,solar_azimuth_deg=v['solar']['azimuth'] if v else None,solar_zenith_deg=v['solar']['zenith'] if v else None,solar_level='original tile mean degrees; coarse 23x23 angle grid available; no interpolation' if v else None,solar_status='approximate-only' if v else 'missing',source_CRS=v['source_grid']['crs_code'] if v else None,source_grid=v['source_grid']['resolutions'] if v else None,source_grid_status='ready' if v else 'missing',cube_origin=None,cube_rowcol_orientation=None,cube_pixel_center=None,cube_mapping_status='missing',mapping_evidence='Phase75 pending; Phase309 documents 1152->1024->1022; symmetric65 inset conditional only',cloud_height_m=None,cloud_height_prior=None,height_status='missing',viewing_angle_status='approximate-only' if ident and ident['viewing_angle_grids_present'] else 'missing',terrain_status='missing',context_status='missing',context_reason='published 1022 crop lacks external cloud context; 1152 is documentation extent, not an available registered context raster',old75_selected=bool(v),old_selection_bias='public positive Shadow proportion among153fit then fixedhash first3; not representative of132' if v else None,ready_status='missing',reason='unverified cube transform and height/context prevent directional pairing; no inferred affine or negative from unpaired Shadow'))
    assert len(rows)==132
    table(ROOT/'INPUT_READINESS.csv',rows);write(ROOT/'ARRAY_HEADER_EVIDENCE.json',arrays)
    intersection=[dict(product=r['product'],pool=r['pool'],group_id=r['group_id'],selection_bias=r['old_selection_bias']) for r in rows if r['old75_selected']]
    table(ROOT/'OLD75_INTERSECTION.csv',intersection)
    write(ROOT/'INVENTORY_ACCEPTANCE.json',dict(products=len(rows),fit=100,development=32,old75_intersection=intersection,ready=sum(r['ready_status']=='ready' for r in rows),solar_metadata=sum(r['solar_status']!='missing' for r in rows),source_grid=sum(r['source_grid_status']=='ready' for r in rows),cube_verified=0,height_available=0,external_context_verified=0,source_scores_existing=sum(r['Source_scores_status']=='ready' for r in rows),student_caches_existing=sum(r['Q90_PL_cache_status']=='ready' for r in rows),no_array_payload_read=True,no_reference_arrays_read=True))

def main():
    assert sys.platform=='linux' and sys.executable.startswith('/home/scv/miniconda3/envs/cloud-lite-pt210/')
    assert not ROOT.exists(),'no repeated audit or root overwrite'
    ROOT.mkdir();resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));guard();threading.Thread(target=watch,daemon=True).start()
    write(ROOT/'STATUS.json',dict(status='running',utc=utc()))
    try:
        code={str(p.relative_to(R)):sha(p) for p in (R/'src/tools').glob('*phase313*') if p.is_file()}
        write(ROOT/'EXECUTION_LOCK.json',dict(code=code,plan_sha256=sha(R/'src/research_plans/phase313.md'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),python=sys.executable,authorization='User execute313 plus persistent git/results synchronization; no new training authorization',scope='metadata headers primary literature only; no historical entrypoints',utc=utc()))
        inventory()
        for name,url in URLS:fetch(name,url)
        # Reuse isolated existing PDF reader; never install or alter environment.
        sys.path.insert(0,'/home/scv/shared/phase309_20261025/toolenv')
        import pypdf
        for n in NETWORK:
            if n.get('status')==200 and n.get('path','').endswith('.pdf'):
                p=ROOT/n['path'];reader=pypdf.PdfReader(p)
                for i,page in enumerate(reader.pages):
                    t=page.extract_text() or '';dest=ROOT/'literature'/f'{n["object"]}_page{i+1:03}.txt';dest.write_text(t,encoding='utf-8')
                n['pages']=len(reader.pages);n['reader_version']=pypdf.__version__
        guard()
        write(ROOT/'STATUS.json',dict(status='inventory_complete_pending_editorial',utc=utc(),wall_seconds=time.monotonic()-START,peak_cpu_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,minimum_available_memory=MINMEM,train=0,fit=0,forwards=0,projection=0,raster_downloads=0,DEM_downloads=0,network_response_body_bytes=TOTAL,reading_objects=len(URLS)))
    except BaseException:
        write(ROOT/'FAILURE.json',dict(utc=utc(),traceback=traceback.format_exc(),action='preserve; no blind repeat'));write(ROOT/'STATUS.json',dict(status='failed',utc=utc()));raise
    finally:
        write(ROOT/'INPUT_SHA256.json',INPUTS);write(ROOT/'NETWORK.json',dict(response_body_bytes=TOTAL,limit=20000000,objects=len(URLS),requests=NETWORK));table(ROOT/'EVIDENCE_INDEX.csv',EVIDENCE);write(ROOT/'ACCESS.json',ACCESSES)
    print(json.dumps(load(ROOT/'INVENTORY_ACCEPTANCE.json')));print(json.dumps(load(ROOT/'STATUS.json')))
if __name__=='__main__':main()

"""One-pass Kappa identity/semantics/provenance audit. Metadata only, no HTTP."""
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
from pathlib import PurePosixPath
import re
import resource
import subprocess
from phase316_common import *

INPUTS={}
def pin(p,role):
    p=Path(p);assert p.is_file(),p
    assert p.resolve().is_relative_to(R) or p.resolve().is_relative_to(Path('/home/scv/shared'))
    h=sha(p);INPUTS[str(p)]=dict(sha256=h,bytes=p.stat().st_size,role=role)
    return p
def js(p,role):return load(pin(p,role))
def cs(p,role):return read(pin(p,role))
def manual_examples(patches):
    examples=[]
    for name,split,tile,when,x,y in [
        ('kappaset/test/T52JGR_20200806T011729_tile_4_8.nc','test','52JGR','20200806T011729',4,8),
        ('kappaset/test/T35QPG_20200416T083559_tile_17_12.nc','test','35QPG','20200416T083559',17,12),
        ('kappaset/test/T16UEA_20201027T164409_tile_16_17.nc','test','16UEA','20201027T164409',16,17)]:
        r=next(p for p in patches if p['archive_path']==name)
        assert r['tile']==tile and r['sensing_utc_key']==when and int(r['index_x'])==x and int(r['index_y'])==y
        assert int(r['declared_pixel_offset_x'])==x*512 and int(r['declared_pixel_offset_y'])==y*512
        examples.append(dict(member=name,split=split,tile=tile,time=when,index_x=x,index_y=y,offset_x=x*512,offset_y=y*512,interpretation='index arithmetic only; axes/origin/geolocation not certified'))
    write(ROOT/'MANUAL_NAME_EXAMPLES.json',examples)

def main():
    assert not (ROOT/'AUDIT_COMPLETE.json').exists() and not (ROOT/'IDENTITY_RECONCILIATION.csv').exists()
    resource_check();start=time.monotonic()
    # Explicitly deny raw imagery, labels, predictions and model deserialization.
    weight=R/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'
    def io_guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0]));s=str(p).lower()
        if p.suffix.lower() in ['.nc','.npy','.npz','.tif','.tiff','.jp2','.png','.pth','.pt'] and p!=weight:raise PermissionError('Phase316 metadata-only: '+s)
        if p.name.lower() in ['kappaset.zip','masks.zip','images.zip','alt_masks.zip']:raise PermissionError(s)
    sys.addaudithook(io_guard)
    oldaudit=js(OLD/'metadata_audit.json','Phase102 frozen counts/input SHA')
    for name,h in oldaudit['input_SHA256'].items():assert sha(OLD/'inputs'/name)==h,name
    patches=cs(OLD/'inputs/kappa_patch_positions.csv','original9251patch name/index projection')
    oldex={r['record_key']:r for r in cs(OLD/'overlap_exclusions.csv','original exclusions retained byte-exact') if r['source']=='KappaSet7100327'}
    loc={r['record_key']:r for r in cs(OLD/'product_location_index.csv','old per-key provenance/status') if r['source']=='KappaSet7100327'}
    candidates=js(OLD/'inputs/candidate_products_metadata.json','old acquisition metadata including one reprocessed proxy')
    hist=js(OLD/'inputs/history_positions_only.json','old9717position-only registry')
    groups=cs(OLD/'related_groups.csv','old incomplete connectivity components; not statistical independence')
    sources=js(OLD/'inputs/source_records.json','pinned release/version/license statements')
    oldaccess=js(OLD/'inputs/source_access_sha256.json','old acquisition SHA/size bindings')
    ledger=js(OLD/'inputs/access_ledger.json','old HTTP ledger; no new requests')
    js(OLD/'inputs/exposure_scan.json','old657file history scope')
    pin(OLD/'README.md','old audit limits');pin(OLD/'SOURCE_FACTS.md','old source interpretation/version warning')
    pin(R/'src/research_results/phase315_20261026/CONFIRMATION_POOL_ADMISSION.csv','Phase315 pending row, not proof no evidence')
    reg=js(E/'record_7100327.json','original cached official Zenodo API7100327 revision6')
    archive=js(E/'kappaset_archive_index.json','previously acquired ZIP central-directory name/size metadata')
    for p in sorted(E.iterdir()):
        pin(p,'existing official README or author-code hint; no new acquisition')
        entry=next(x for x in oldaccess['sources'] if Path(x['local_path']).name==p.name)
        assert sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes'],p
    assert sha(E/'record_7100327.json')==next(x for x in sources if x['requested_record']==7100327)['record_sha256']
    pdf=E/'kappaset_readme.pdf';assert pdf.stat().st_size==231988 and hashlib.md5(pdf.read_bytes()).hexdigest()=='98bf16a2d90cba386b197133a24f402f'
    assert reg['id']==7100327 and reg['metadata']['version']=='1'
    assert next(f for f in reg['files'] if f['key']=='kappaset.zip')['checksum']=='md5:3f435c61efbb511c13a353a322787e42'
    manual_examples(patches)
    nc=[r for r in archive if r['path'].endswith('.nc')];assert len(nc)==len(patches)==9251
    assert Counter(r['path'] for r in nc)==Counter(r['archive_path'] for r in patches)
    parsed=[];bykey=defaultdict(list);bysplit=defaultdict(list)
    for p in patches:
        name=p['archive_path'];m=re.fullmatch(r'kappaset/([^/]+)/T(\d{2}[A-Z]{3})_(\d{8}T\d{6})_tile_(\d+)_(\d+)\.nc',name);assert m,name
        split,tile,t,x,y=m.groups();datetime.strptime(t,'%Y%m%dT%H%M%S');key=f'Kappa:T{tile}:{t}'
        assert p['record_key']==key and p['tile']==tile and int(p['index_x'])==int(x) and int(p['index_y'])==int(y)
        parsed.append(dict(member=name,release_directory=split,tile=tile,sensing_utc=t,index_x=int(x),index_y=int(y),record_key=key,original_full_product='unknown',identity_level='tile/time only; satellite/baseline/orbit/generation not recovered'))
        bykey[key].append(name);bysplit[split].append(key)
    size_byname={r['path']:r for r in nc}
    for p in parsed:p.update(zip_compressed_bytes=size_byname[p['member']]['compressed'],netcdf_uncompressed_bytes=size_byname[p['member']]['uncompressed'])
    assert len(bykey)==1079 and set(bykey)==set(oldex)
    month_keys=set(k for split,v in bysplit.items() if split!='test' for k in v);test_keys=set(bysplit['test'])
    duplicate_paths=sum(n-1 for n in Counter(p['member'] for p in parsed).values())
    split_rows=[dict(directory=k,patches=len(v),unique_tile_time_keys=len(set(v)),role='published directory; month is not a newly assigned training split' if k!='test' else 'published test directory; not project-level untouched guarantee') for k,v in sorted(bysplit.items())]
    table(ROOT/'RELEASE_SPLITS.csv',split_rows);table(ROOT/'PATCH_MEMBER_MAP.csv',parsed)
    identities=[
        dict(object='official_original_products',what_counted='README/API declared Sentinel2L1C products',source='7100327 Version1 README p1/API revision6',recomputed='unknown exact original products',difference='official1038; tile/time keys are a different identity level',unresolved='complete SAFE/product identifiers absent; do not deduplicate to force1038'),
        dict(object='archive_NetCDF_files',what_counted='all .nc entries in existing frozen central directory',source='kappaset_archive_index.json',recomputed=len(nc),difference='9251 exactly matches README',unresolved='payload not downloaded/opened'),
        dict(object='duplicate_full_member_paths',what_counted='9251 full archive paths',source='PATCH_MEMBER_MAP',recomputed=duplicate_paths,difference='same basename in different directories counted separately',unresolved='byte-identical payload duplication unknown'),
        dict(object='all_tile_time_keys',what_counted='tile plus full timestamp',source='9251filename parse; original1079-key registry',recomputed=len(bykey),difference=f'{len(bykey)-1038:+d} relative to official product count',unresolved='key is not full product identity'),
        dict(object='non_test_month_directory_keys',what_counted='union of12month directories',source='RELEASE_SPLITS',recomputed=len(month_keys),difference=f'test keys={len(test_keys)}; month/test shared keys={len(month_keys&test_keys)}',unresolved='directory split may explain some count levels; do not infer official count excludes test without explicit statement'),
        dict(object='test_only_keys',what_counted='keys in published test and absent from month directories',source='RELEASE_SPLITS',recomputed=len(test_keys-month_keys),difference='see full key membership map',unresolved='not proof of originalproduct identity/independence'),
        dict(object='unique_tile_day_keys',what_counted='tile plus UTC day, diagnostic alternative only',source='PATCH_MEMBER_MAP',recomputed=len({(p['tile'],p['sensing_utc'][:8]) for p in parsed}),difference='no forced same-day collapse',unresolved='same tile/day not exact product'),
        dict(object='original_full_products_confirmed',what_counted='full original S2 SAFE identifiers in name/index evidence',source='original metadata product_id fields',recomputed=0,difference=f'{len(bykey)}keys remain unresolved at fullproduct level',unresolved='one reprocessed proxy is explicitly excluded from exact-original identity'),
        dict(object='unique_tiles',what_counted='MGRS tile strings',source='PATCH_MEMBER_MAP',recomputed=len({p['tile'] for p in parsed}),difference='not independent sample/group count',unresolved='no new geometry certification')]
    table(ROOT/'IDENTITY_RECONCILIATION.csv',identities)
    # Bind the actual Source checkpoint to the original saved config and published train/val map.
    assert sha(pin(weight,'SHA only; never deserialized'))==SOURCE
    manifest=js(Path('/home/scv/shared/data/phase65_source_metadata/source_metadata_manifest.json'),'published CloudSEN12 train/val record revision')
    mapping=js(R/'src/work_dirs/phase65a/source_mapping/mapping_audit.json','existing exact RGB PNGordinal-to-product proof, no pixels reopened')
    assert sha(R/'src/work_dirs/phase65a/source_mapping/mapping_audit.json')=='c3c521a375c50f81ff0e58682621c7fcb72fc47472a7b37931c587036f12d0bb'
    assert mapping['status']=='verified_rgb_ordinal_mapping'
    src_config=pin(R/'src/work_dirs/phase64_source_parent/seed64/phase64_source_parent_rgb.py','actual saved Source training config')
    cfg=src_config.read_text();assert 'Phase64CloudSENParentDataset' in cfg and 'img_dir/train' in cfg and 'img_dir/val' in cfg
    mapped={(r['split'],r['png_ordinal']):r for r in mapping['mappings']};source_by_tile=defaultdict(list);source_by_key=defaultdict(list);source_rows=[];source_counts={}
    for split in ['train','val']:
        p=Path('/home/scv/shared/data/phase65_source_metadata')/(split+'_metadata.csv');rows=cs(p,'actual Source '+split+' product and geometry metadata')
        assert sha(p)==manifest['splits'][split]['sha256']==mapping['splits'][split]['metadata_sha256']
        assert len(rows)==mapping['splits'][split]['rows'];source_counts[split]=len(rows)
        for i,r in enumerate(rows):
            assert mapped[(split,i)]['product']==r['s2_id'] and mapped[(split,i)]['exact_rgb_match']
            m=re.fullmatch(r'S2[AB]_MSIL1C_(\d{8}T\d{6})_N\d+_R\d+_T(\d{2}[A-Z]{3})_\d{8}T\d{6}',r['s2_id']);assert m
            t,tile=m.groups();item=dict(product=r['s2_id'],role='Source_'+split,roi_id=r['roi_id'],metadata_index=r['index'],epsg=r['proj_epsg'],geometry=r['proj_geometry'],evidence=str(p))
            source_by_tile[tile].append(item);source_by_key[f'Kappa:T{tile}:{t}'].append(item);source_rows.append(item)
    table(ROOT/'SOURCE_TRAIN_VAL_BINDING.csv',source_rows)
    for p in [R/'src/configs/protocol/phase64_source_parent_rgb.py',R/'src/cloud_adapter/datasets/phase64_parent.py',R/'src/tools/phase302_images.py',R/'src/tools/execute_phase302.py']:
        pin(p,'currentSource ontology/input implementation as text only')
    # Historical matches are carried forward, not silently reclassified as current Source training overlap.
    oldhist=defaultdict(list)
    for h in hist['records']:oldhist[h['tile']].append(h)
    out=[];role_counts=Counter();status_counts=Counter();same_acq=0
    groupmap={k:g['group_id'] for g in groups if g['source']=='KappaSet7100327' for k in json.loads(g['members'])}
    for key,names in sorted(bykey.items()):
        tile=key.split(':')[1][1:];old=oldex[key];matches=json.loads(old['history_matches']);current=source_by_tile[tile];exact=source_by_key[key];same_acq+=bool(exact)
        current_roles=sorted({r['role'] for r in current});history_roles=sorted({r['role'] for r in matches if not r['role'].startswith('Source_')})
        status='有证据排除（保留旧保守排除）' if old['status']=='excluded_overlap' else '仍有关键未知'
        # Same-tile risk is a conservative exclusion; neither exact pixel overlap nor exact identity is inferred.
        role_counts.update(current_roles);status_counts[status]+=1
        out.append(dict(record_key=key,tile=tile,release_directories=sorted({PurePosixPath(n).parts[1] for n in names}),patch_count=len(names),member_map='PATCH_MEMBER_MAP.csv#'+key,original_status=old['status'],original_reasons=json.loads(old['exclusion_reasons']),original_history_matches=matches,current_Source_same_tile_rows=len(current),current_Source_roles=current_roles,current_Source_tile_time_rows=len(exact),current_Source_tile_time_products=sorted({r['product'] for r in exact}),current_Source_evidence='SOURCE_TRAIN_VAL_BINDING.csv; published metadata and existing exactRGB ordinal proof',historical_non_Source_roles=history_roles,historical_use_status='只见元数据；范围外unknown',historical_use_evidence='Phase102 metadata intake + Phase315 admission; bounded Phase316 text scan; no new-pixel use established',spatial_precision='whole-tile name match only; exact Kappa footprint unknown; same tile is possible overlap, not proven same pixel',old_incomplete_component=groupmap.get(key,'excluded/not assigned'),current_status=status,change_from_old='none; source/non-source reasons separated, no exclusion removed',full_product='unknown',Shadow_positive_support='unknown',ready_for_small_sample=False))
    assert sum(r['original_status']=='excluded_overlap' for r in out)==180 and sum(r['original_status']=='pending_metadata' for r in out)==899
    table(ROOT/'OVERLAP_AND_HISTORY.csv',out)
    # Bounded reporting search, no workdirs or performance payloads.
    scanned=[];hits=[]
    for root in [R/'src/tools',R/'src/research_plans',R/'src/research_results']:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.suffix not in ['.py','.md','.json'] or p.stat().st_size>1500000 or 'phase316' in str(p).lower() or 'sealed' in p.name.lower():continue
            if root.name=='research_results' and not any(x in p.name.lower() for x in ['admission','provenance','exposure','history','source_record','metadata','readiness','source_facts']):continue
            text=p.read_text(encoding='utf-8-sig',errors='replace');h=sha(p);scanned.append(dict(path=p.relative_to(R).as_posix(),sha256=h))
            if re.search(r'kappaset|7100327|kappazeta',text,re.I):hits.append(dict(path=p.relative_to(R).as_posix(),sha256=h,matched_lines=[i for i,line in enumerate(text.splitlines(),1) if re.search(r'kappaset|7100327|kappazeta',line,re.I)],interpretation='text mention is not proof of pixel use or non-use; retain prior metadata-only finding'))
    write(ROOT/'HISTORY_SEARCH.json',dict(scope='tools/plans .py/.md/.json <=1.5MB; results only admission/provenance/exposure/history/source_record/metadata/readiness/source_facts names; excludePhase316/sealednames/workdirs/pixel/prediction/metrics',files=scanned,hits=hits,limits='unrecordedmanualuse/externalpretraining/otherfiles unknown; no hit is not proof unseen'))
    # Source-anchored schema, all original values retained; no actual arrays were inspected.
    fields=[]
    def field(item,value,rule,status,risk,source):fields.append(dict(item=item,official_or_code_value=value,Source_rule_or_future_requirement=rule,evidence_status=status,difference_risk=risk,source=source))
    for code,label,mapped_class in [(0,'UNDEFINED','ignore255'),(1,'CLEAR','Surface0'),(2,'CLOUD SHADOW','Shadow2'),(3,'SEMI TRANSPARENT CLOUD','Cloud1'),(4,'CLOUD','Cloud1'),(5,'MISSING','ignore255')]:
        field('Label code '+str(code),label,mapped_class,'文档已知；数组未验','保留原码；0/5绝不作Surface；旧5095024的0/5含义相反','7100327 README p1; original PDF MD5 verified')
    field('annotation','CVAT/Segments.ai; active-learning semi-automatic labelling; challenging patches selected','parent-class merge only; not independent physical gold truth','文档已知','thin-cloud subtype/boundary differences and selected-patch bias remain','README p1/API description')
    field('dimensions/resolution','512x512,13bands resampled10m','do not silently change current20m Catalogue512patch protocol','文档已知；数组未验','512native10m=5.12km; originalCatalogue512at20m=10.24km; physical view differs','README p1; phase302_images.py')
    field('variables/order','B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12 Label','select named B04/B03/B02 RGB; never assume on-disk variable enumeration','文档已知；数组未验','must check case/dimensions and sameindex alignment','README p1')
    field('test_extra_variables','FMC SS2C IDPX MAJAC SCL in test, MAJAC limited','not Label; never substitute automatic baselines as truth','文档已知；数组未验','additional maps differ in availability/meaning','README p1')
    field('resampling','Sinc IIR windowed with Blackman to10m','no new resampling chosen this phase','文档已知；数组未验','upsampled20/60m bands not independent10m measurements','README p1')
    field('dtype','unknown release dtype; cached author writer uses NC_FLOAT or NC_UBYTE by image.main_depth','inspect actual variable datatype before conversion','作者代码线索；发布数组未知','current authorcode snapshot not exact release exporter','cmvsm_vsm_lib_raster_netcdf_interface.cpp lines55-100')
    field('scale_factor/add_offset','release values unknown; author writer uses custom scaling_factor attribute','freeze physical conversion only after inspecting raw values/attributes; do not default divide10000','作者代码线索；发布数组未知','custom scaling_factor is not CF scale_factor; double autoscaling risk; add_offset not certified','same author file lines104-108; no release exporter commit binding')
    field('units/valid/fill','TOA physical meaning from L1C; actual DN/reflectance,units,_FillValue,missing_value,valid_range unknown','maskinvalid jointlyRGB+Label0/5 after documented conversion','未知','currentadapter accepts finite TOA including>1, rejects0 and65535/10000; cannot transplant invalid constants blindly','README L1C; phase302_images.py')
    field('Source numerical pipeline','B04/B03/B02 reflectanceRGB *255 -> BGR -> preprocessorRGB; mean[123.675,116.28,103.53],std[58.395,57.12,57.375]','match numericalmeaning not datasetname; no clipping currentpipeline','代码已知；新数组未验','unresolvedKappa storage/offset prevents frozen complete reader','phase302_images.py; execute_phase302.py preprocess')
    field('coordinate/index','README explicitly says currentsubtiles no georeferencing; filename indices multiplied512','require originalparent product/grid, origin,axis orientation and Label/bands shared dims','文档已知；实际字段未知','indices alone do not give CRS/footprint; cloudheight/solarangle not required for thisaudit','README p1 and figure2; old proxy not originalgrid')
    field('actual Shadow positive support','unknown','do not infer positives from classnames; future officialpersamplecounts or separatelyauthorizedsampleinspection','未知','no positivepixel/groupcount or geometry-certifiedindependentgroups available','noLabelpayloadopened thisphase')
    field('license','CC-BY-4.0; open release; creators in API','credit creators, title/version/DOI/license; indicate remapping/changes and no endorsement','原API版本声明已核对；非新授权书','publicreadable not unrestricted; authorcode Apache2.0 separate from datalicense','cached record7100327 metadata.license/creators; writercode header')
    table(ROOT/'INPUT_LABEL_MAPPING.csv',fields)
    write(ROOT/'FUTURE_READER_STATUS.json',dict(status='incomplete_not_frozen_not_implemented',named_RGB=['B04','B03','B02'],label_mapping={'0':'ignore255','1':'Surface0','2':'Shadow2','3':'Cloud1','4':'Cloud1','5':'ignore255'},missing=['actualdtype/units/fill','scaling_factor vs scale_factor/add_offset and exporterrevision','shared axes/dimensions/index orientation','originalproduct/grid/spatialintersection'],no_resampling_or_normalization_selected=True))
    cost=[]
    def action(condition,fact,missing,files,estimate,stop):cost.append(dict(condition=condition,known_fact=fact,missing=missing,minimum_action_exact_files=files,estimated_requests_bytes_disk_CPU=estimate,acceptance_stop=stop,can_enter_small_sample=False))
    action('identity',f'9251entries/{len(bykey)}tile-time keys; monthkeys{len(month_keys)},testkeys{len(test_keys)},shared{len(month_keys&test_keys)}','full originalproducts and official1038 scope','authoritative mapping from every PATCH_MEMBER_MAP member to originalSAFE+releaseREADMEclarification','not present in allowed materials; acquisition sizeunknown; no contact authorized','stopif cannot bind exactidentity; never force1038')
    action('currentSource overlap',f'original180exclusions retained; {sum(bool(r["current_Source_roles"]) for r in out)}keys same tile asactualSource train/val; {same_acq}same tile/time','Kappa actual footprints; exactpixel overlapunknown','originalparentgrid/origin/axes or authoritative wholeSentineltile footprint for every proposed key','metadatafetchcostunknown; no cloudheight/solarangle required','must be spatially disjoint from all known training/selection/history footprints with inherited2km rule; intersectingtile with unknownpatch staysunknown')
    action('pixel/reader semantics','class mapping documented; RGB namesknown; no actualarraychecked','dtype/units/fill/attributes/shared dimensions and support','only after admission: named.ncheader/attributeinspection; labelpixels require separate authorization',f'per.nc compressed min/max {min(r["compressed"] for r in nc)}/{max(r["compressed"] for r in nc)}bytes; expanded min/max {min(r["uncompressed"] for r in nc)}/{max(r["uncompressed"] for r in nc)}bytes from olddirectory; requestcount/partialretrievalfeasibilityunknown','do not implement readeruntil fieldsknown; no promise of tiny selectiveaccess')
    action('bulk retrieval fallback','ZIP56186308652bytes; publishedMD5 only; archive not present/acquired', 'reliable selective member access/export binding', 'https://zenodo.org/api/records/7100327/files/kappaset.zip/content',f'if only fullZIP possible:download56186308652B plus uncompressedallNC{sum(r["uncompressed"] for r in nc)}B disk; extractionCPUunknown; no newdownload performed','far beyond current300MB budget; stop and report realcost, not auto-download')
    action('historicaluse/basepretraining','onlymetadatamentions established within statedscope','unrecorded use and DINO pretraining coverage unknown','HISTORY_SEARCH plus original657scan and Source train/val binding','0HTTP now; retrospective missingrecords acquisitioncostunknown','absence of evidence not unseenproof; no poolrenaming')
    action('Shadow support','actual positivepixels/groups unknown','positive support in genuinely disjointcandidategroups','official per-sample classstatistics if available; otherwise later separatelyauthorized fixedsampleinspection','not known to exist; metadata/labelfetchcostunknown','not ready for directShadowconfirmation; no candidatepasses current mandatorygaps')
    table(ROOT/'ADMISSION_AND_COST.csv',cost)
    table(ROOT/'CANDIDATE_LIST.csv',[],['record_key','members','basis','ready_for_small_sample'])
    (ROOT/'ZERO_CANDIDATES.md').write_text('0候选：180键保留旧保守排除，其余899键仍缺精确产品/足迹和完整输入存储规则；本轮不把只无同tile记录当作空间独立。0个已经满足按身份及非重合条件进入小样本读取的候选；这不等于证明数据集不可能使用。实际Shadow支持unknown，不自动进入下一轮同类核查。\n',encoding='utf-8')
    # Save the exact small source metadata and a scope ledger. No model/performance payload is copied.
    inputs_dir=ROOT/'inputs';inputs_dir.mkdir()
    for i,(path,v) in enumerate(sorted(INPUTS.items())):
        if path==str(weight):continue
        target=inputs_dir/(str(i).zfill(3)+'_'+Path(path).name);shutil.copyfile(path,target);v['backup_relative']=target.relative_to(ROOT).as_posix()
    write(ROOT/'INPUT_SHA256.json',INPUTS)
    official=[]
    for f in E.iterdir():
        old=next(x for x in oldaccess['sources'] if Path(x['local_path']).name==f.name)
        events=[v for v in ledger if v.get('file')==f.name]
        official.append(dict(file=f.name,sha256=sha(f),bytes=f.stat().st_size,prior_path=old['local_path'],prior_events=events,note='Phase102 cache reused; old recorded acquisitiontime if present; Phase316 no HTTP'))
    write(ROOT/'EVIDENCE_INDEX.json',dict(reused_files=official,release=dict(id=7100327,version=reg['metadata']['version'],revision=reg.get('revision'),created=reg.get('created'),modified=reg.get('modified'),license=reg['metadata']['license'],creators=reg['metadata']['creators']),checkpoint=dict(path=str(weight),sha256=SOURCE),source_manifest=manifest,cached_author_code_not_exporter_binding=True))
    write(ROOT/'NETWORK_LEDGER.json',dict(new_official_requests=0,new_response_body_bytes=0,redirects=0,failures=0,reason='existing originalREADME/API/authorcode and archiveindex sufficient to establish gaps; no refetch or newZIP range',code_sync_events=[dict(action='first gzs gitpull',result='GnuTLS handshake non-properlyterminated beforediscovery; safe sync retry succeeded'),dict(action='first audit-code gitpush40239ba',result='Connection closed by198.18.1.7port22; no remoteaudit execution; sync retried')],budget_requests=3,budget_body_bytes=3000000,scientific_data_downloads=0))
    stats=dict(status='completed_metadata_only',patches=len(patches),unique_paths=len({p['member'] for p in parsed}),keys=len(bykey),month_keys=len(month_keys),test_keys=len(test_keys),month_test_shared_keys=len(month_keys&test_keys),full_products_verified=0,keys_full_product_unknown=len(bykey),official_products=1038,excluded_keys=180,pending_keys=899,original_exclusions_unchanged=True,current_Source_same_tile_keys=sum(bool(r['current_Source_roles']) for r in out),current_Source_same_tile_time_keys=same_acq,Source_rows=source_counts,current_Source_role_key_counts=dict(role_counts),eligible_candidates=0,Shadow_positive_support='unknown',new_requests=0,new_image_label_members=0,neural_forwards=0,training_updates=0,cpu_seconds=time.monotonic()-start,resources=resource_check())
    write(ROOT/'AUDIT_COMPLETE.json',stats)
    print(json.dumps(stats,ensure_ascii=False))
if __name__=='__main__':main()

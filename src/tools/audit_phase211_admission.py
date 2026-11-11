"""Phase211: text/geometry metadata only. Never open raster/score/weight bytes."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import re
import struct
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/phase211/phase211_20261020'
INPUTS = {}
OPENED = set()
FORBIDDEN = {'.png', '.jpg', '.jpeg', '.jp2', '.tif', '.tiff', '.dat', '.npy', '.npz', '.pth', '.pt', '.safetensors'}


def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        p = Path(args[0]).resolve()
        if p.suffix.lower() in FORBIDDEN:
            raise RuntimeError(f'Phase211 forbids raster/score/weight byte access: {p}')
        if p.suffix.lower() == '.zip' and p.name != 'shapefiles.zip':
            raise RuntimeError(f'Only geometry ZIP is permitted: {p}')
        OPENED.add(str(p))


sys.addaudithook(guard)


def read(rel):
    p = ROOT / rel
    b = p.read_bytes()
    INPUTS[rel] = {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
    return b


def js(rel):
    return json.loads(read(rel).decode('utf-8-sig'))


def rows(rel):
    return list(csv.DictReader(read(rel).decode('utf-8-sig').splitlines()))


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def writecsv(name, data):
    assert data
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


def identity(p):
    t = re.search(r'_T(\d{2}[A-Z]{3})(?:_|$)', p)
    d = re.search(r'(20\d{6})T\d{6}', p)
    o = re.search(r'_R(\d{3})(?:_|$)', p)
    return (t.group(1) if t else '', d.group(1) if d else '', o.group(1) if o else '')


def native_bbox(wkt):
    xy = re.findall(r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)', wkt)
    pts = [(float(x), float(y)) for x, y in xy]
    return [min(x for x, y in pts), min(y for x, y in pts), max(x for x, y in pts), max(y for x, y in pts)]


def distance(a, b):
    return math.hypot(max(a[0]-b[2], b[0]-a[2], 0), max(a[1]-b[3], b[1]-a[3], 0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fetch = js('outputs/phase211/phase211_20261020/SOURCE_FETCH_MANIFEST.json')
    records = fetch['records']
    sources = []
    for r in records:
        if r.get('status') == 'ok':
            b = read('outputs/phase211/phase211_20261020/sources/' + r['name'])
            assert hashlib.sha256(b).hexdigest() == r['sha256']
            sources.append(r)
    assert sum(r['bytes'] for r in sources) <= 20_000_000
    train = rows('outputs/phase65/source_metadata/acquired/train_metadata.csv')
    val = rows('outputs/phase65/source_metadata/acquired/val_metadata.csv')
    test = rows('outputs/phase211/phase211_20261020/sources/test_metadata.csv')
    mapping = js('outputs/phase65/work_dirs/phase65a/source_mapping/mapping_audit.json')
    for split, data in [('train', train), ('val', val)]:
        mm = sorted((x for x in mapping['mappings'] if x['split'] == split), key=lambda x: x['png_ordinal'])
        assert len(mm) == len(data)
        assert all(x['exact_rgb_match'] and x['product'] == y['s2_id'] and x['original_index'] == y['index'] for x, y in zip(mm, data))
    hist = js('src/research_results/phase102_20261013/inputs/history_positions_only.json')['records']
    rules = js('src/research_results/phase109_20261013/OVERLAP_RULES.json')
    fit = js('outputs/phase208/phase208_20261020/delivery/12201_FIT_INPUT_LOCK.json')['bindings']
    cal = js('outputs/phase209/phase209_20261020/CALIBRATION_PLAN.json')['rows']
    ev = js('outputs/phase208/phase208_20261020/delivery/EVALUATION_INPUT_PLAN.json')['rows']
    fitp = {x['product'] for x in fit}; fitg = {x['group_id'] for x in fit}
    calp = {x['product_id'] for x in cal}; calg = {x['group']['group_id'] for x in cal}
    devp = {x['product_id'] for x in ev if x['pool'] == 'Catalogue'}
    pixp = {x['product_id'] for x in ev if x['pool'] == 'PixBox'}
    assert len(fitg) == 100 and len(calg) == 53 and len(devp) == 32 and len(pixp) == 16
    assert not fitg & calg
    cache_inventory = []
    for model in ['R4', 'M', 'B']:
        base = ROOT / ('outputs/phase205/phase205_20261019/delivery' if model == 'R4' else 'outputs/phase208/phase208_20261020/delivery') / 'infer' / model / 'scores'
        for x in ev:
            p = base / x['pool'] / (x['product_id']+'.npz' if x['pool']=='Catalogue' else x['product_id']+'/UNLABELLED_POINT_SCORES.json')
            assert p.is_file(), f'Missing previously sealed cache {p}'
            cache_inventory.append({'model':model,'pool':x['pool'],'product':x['product_id'],'path':str(p.relative_to(ROOT)),'bytes_stat_only':p.stat().st_size,'bytes_opened':False})
    assert len(cache_inventory)==144
    writecsv('EXISTING_CACHE_INVENTORY.csv',cache_inventory)
    groups = js('outputs/phase65/source_metadata/scene_groups.json')['groups']
    gid = {p: x['group_id'] for x in groups for p in x['members']}
    tags = rows('data/sentinel2_cloud_mask_catalogue_4172871/classification_tags.csv')
    # Read only the geometry package; never masks/subscenes or rasters.
    rel = 'data/sentinel2_cloud_mask_catalogue_4172871/shapefiles.zip'
    read(rel)  # hash geometry-only input
    geo = {}
    with zipfile.ZipFile(ROOT / rel) as z:
        for n in z.namelist():
            if not n.endswith('.shp'):
                continue
            product = Path(n).stem
            shp = z.read(n)
            prj = z.read(n[:-4] + '.prj').decode('ascii')
            utm = re.search(r'UTM_Zone_(\d+)([NS])', prj)
            assert utm and struct.unpack_from('<i', shp, 108)[0] == 5
            # Polygon envelopes are conservative; bbox proximity is not exact polygon overlap.
            geo[product] = {'epsg': str((32600 if utm[2] == 'N' else 32700) + int(utm[1])), 'bbox': list(struct.unpack_from('<4d', shp, 112))}
    assert len(geo) == 513
    mapdata = js('outputs/phase211/phase211_20261020/sources/v4_map_data.geojson')
    v4map = {}
    for f in mapdata['features']:
        props = f['properties']; name = props['Name']
        if name in v4map:
            assert props == v4map[name], 'Conflicting duplicate map properties'
        v4map[name] = props
    highkey = 'CloudSEN12 high L1C+L2A'
    tc = collections.Counter(identity(x['s2_id'])[0] for x in train)
    assert sum(x[highkey] for x in v4map.values()) == 16980
    assert all(v4map[t][highkey] == 2*n for t, n in tc.items())

    inventory = []
    splitstats = {}
    for split, data in [('train', train), ('val', val), ('test', test)]:
        names = []
        for kind in ['img_dir', 'ann_dir']:
            folder = ROOT / 'data/cloudsen12_high_l1c' / kind / split
            files = sorted(folder.glob('*.png'), key=lambda p: int(p.stem))
            assert [p.name for p in files] == [f'{i}.png' for i in range(len(data))]
            names.append([p.name for p in files])
            inventory += [{'split': split, 'kind': kind, 'name': p.name, 'bytes_stat_only': p.stat().st_size} for p in files]
        assert names[0] == names[1]
        tiles = {identity(x['s2_id'])[0] for x in data}
        splitstats[split] = {'chips': len(data), 'unique_products': len({x['s2_id'] for x in data}), 'ROIs': len({x['roi_id'] for x in data}), 'tiles': len(tiles), 'v4_any_training_tiles': sum(v4map.get(t, {}).get('Training data count', 0)>0 for t in tiles), 'v4_high_training_tiles': sum(v4map.get(t, {}).get(highkey, 0)>0 for t in tiles), 'chips_on_v4_any_training_tile': sum(v4map.get(identity(x['s2_id'])[0], {}).get('Training data count', 0)>0 for x in data), 'local_mapping': 'historical_exact_RGB_match' if split != 'test' else 'pending_not_verified_by_row_order', 'label_support': 'unknown_no_pixels_read'}
    writecsv('LOCAL_FILE_INVENTORY.csv', inventory)

    # Known components: same product/tile/day+orbit, plus native-CRS 2km envelope proximity.
    # Cross-CRS spatial relations remain pending. These are NOT certified independent groups.
    cs = [('train', i, x) for i, x in enumerate(train)] + [('val', i, x) for i, x in enumerate(val)] + [('test', i, x) for i, x in enumerate(test)]
    parent = list(range(len(cs)))
    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a, b):
        parent[find(b)] = find(a)
    keys = {}; buckets = collections.defaultdict(list); boxes = []
    for i, (split, ordinal, x) in enumerate(cs):
        tile, day, orbit = identity(x['s2_id']); bb = native_bbox(x['proj_geometry']); boxes.append(bb)
        for key in [('product', x['s2_id']), ('tile', tile), ('day_orbit', day, orbit)]:
            if key in keys: union(i, keys[key])
            keys[key] = i
        # 20km bucket broad phase; each footprint inserted into all touched cells.
        cells = [(x['proj_epsg'], xx, yy) for xx in range(math.floor((bb[0]-2000)/20000), math.floor((bb[2]+2000)/20000)+1) for yy in range(math.floor((bb[1]-2000)/20000), math.floor((bb[3]+2000)/20000)+1)]
        checked = set()
        for cell in cells:
            for j in buckets[cell]:
                if j not in checked and distance(bb, boxes[j]) <= 2000: union(i, j)
                checked.add(j)
        for cell in cells: buckets[cell].append(i)
    members = collections.defaultdict(list)
    for i, (s, o, x) in enumerate(cs): members[find(i)].append(f'{s}:{o}:{x["s2_id"]}')
    ids = {k: hashlib.sha256('|'.join(sorted(v)).encode()).hexdigest() for k, v in members.items()}
    groupids = {(s, o): ids[find(i)] for i, (s, o, x) in enumerate(cs)}
    for split in splitstats:
        splitstats[split]['known_component_count_upper_bound'] = len({groupids[(s,o)] for s,o,x in cs if s == split})
        splitstats[split]['cross_crs_group_status'] = 'pending; never count these as certified independent groups'

    def overlap(product, tile, day, orbit, epsg='', bb=None):
        hits = []; pending = 0
        for p, g in geo.items():
            t, d, o = identity(p)
            why = []
            if product == p: why.append('same_product')
            if tile and tile == t: why.append('same_tile')
            if day and orbit and day == d and orbit == o: why.append('same_UTC_day_orbit')
            if bb is not None and epsg == g['epsg'] and distance(bb, g['bbox']) <= 2000:
                why.append('native_2km_envelope_contact_potential')
            if bb is None or epsg != g['epsg']: pending += 1
            if why: hits.append((p, '+'.join(why)))
        def hitpool(pp, gg):
            return '|'.join(sorted({p + ':' + w for p,w in hits if p in pp or gid.get(p) in gg}))
        return {'fit100_matches': hitpool(fitp, fitg), 'cal53_matches': hitpool(calp, calg), 'Catalogue_all_matches': '|'.join(p+':'+w for p,w in hits), 'cross_CRS_or_missing_geometry_pairs_pending': pending, 'spatial_conclusion': 'known/potential matches listed; cross-CRS pending; no independence claim'}

    manifest = []; overlaps = []
    def add(cid, dataset, split, product, tile, date, footprint, group, group_status, mapping_status, v4, old, inputs, classes, context, admission, reason, epsg='', bb=None):
        t,d,o = identity(product); d = d or date.replace('-','')[:8]
        ov = overlap(product,tile,d,o,epsg,bb)
        mp = v4map.get(tile, {})
        row = {'candidate_id':cid, 'dataset':dataset, 'official_split':split, 'product':product, 'tile':tile, 'sensing_time':date, 'footprint':footprint, 'related_group_id':group, 'group_status':group_status, 'local_identity_mapping':mapping_status, 'V4_exact_sample_training':v4, 'V4_tile_any_training_count':mp.get('Training data count','not_in_map_not_holdout_proof'), 'V4_tile_CloudSEN12_High_count':mp.get(highkey,'not_in_map_not_holdout_proof'), 'V4_selection_holdout':'unknown; public map is not sample/revision-bound split manifest', 'old_project_Source_use':old, 'input_status':inputs, 'label_mapping_ignore':classes, 'actual_valid_shadow_support':'unknown_not_audited_this_phase', 'context':context, 'fit100_relation':'known_or_potential_overlap' if ov['fit100_matches'] else 'no_identity/native_match_detected_cross_CRS_pending', 'cal53_relation':'known_or_potential_overlap' if ov['cal53_matches'] else 'no_identity/native_match_detected_cross_CRS_pending', 'admission':admission, 'reason':reason}
        manifest.append(row); overlaps.append({'candidate_id':cid, 'product':product, 'metadata_mapping':mapping_status, **ov})

    for split, data in [('val', val), ('test', test)]:
        for i,x in enumerate(data):
            p=x['s2_id']; tile,day,orbit=identity(p)
            add(f'CloudSEN12:{split}:{i}', 'CloudSEN12-high@d97728dd61858a21c7d6bc61385687d328433cba', split,p,tile,x['s2_date'],f'EPSG:{x["proj_epsg"]};{x["proj_geometry"]}',groupids[(split,i)],'known component upper bound; cross-CRS pending',splitstats[split]['local_mapping'],'unknown; tile aggregates consistent with published train, not exact-sample evidence','Source selection val/phase64' if split=='val' else 'historical test configured/access exposure cannot be excluded; executed extent unresolved','local uint8 minmax RGB B04/B03/B02 only; raw B8A and physical TOA unavailable','0 Surface;1/2 Cloud;3 Shadow;configured ignore255;512 padding validity unknown','509x509 at10m=5.09km;512 stored;1000/300 context not available','C','missing physical B8A input/context; V4 holdout unknown'+('; test PNG identity unverified' if split=='test' else ''),x['proj_epsg'],native_bbox(x['proj_geometry']))

    rolemap = collections.defaultdict(set)
    for h in hist: rolemap[h['product']].add(h['role'])
    tagid = 'scene' if 'scene' in tags[0] else 'product'
    for x in tags:
        p=x[tagid].removesuffix('.SAFE'); tile,day,orbit=identity(p)
        if p not in geo:
            p=next(q for q in geo if q in p or p in q);tile,day,orbit=identity(p)
        g=gid[p]; valid=str(x['shadows_marked'])=='1'
        in_fit=g in fitg; in_cal=g in calg; cache=p in devp
        admission='B' if cache and valid and not in_fit and not in_cal else 'C'
        reason='cached target-domain history only; not source retention' if admission=='B' else ('fit100/cal53 related group overlap' if in_fit or in_cal else 'cross-CRS geometry/complete input support pending outside existing evaluated pool; no disjointness certification')
        if not valid: reason='author shadows_marked=0; reliable Shadow reference unavailable'
        add('Catalogue:'+p,'Zenodo4172871',str(x.get('dataset','author split not research split')),p,tile,day,f'EPSG:{geo[p]["epsg"]};bbox={geo[p]["bbox"]}',g,'inherited fixed Catalogue group','official archive identity','unknown','|'.join(sorted(rolemap[p])),'13-band physical TOA float32;B04/B03/B8A;20m;existing input locks','Surface/Cloud/Shadow onehot;common support only locked for existing pools','1022x1022 at20m;four1000/300windows',admission,reason,geo[p]['epsg'],geo[p]['bbox'])
    pix = [x for x in js('src/research_results/phase102_20261013/inputs/candidate_products_metadata.json') if x['source']=='PixBox5036991']
    pixgroup = {x['product_id']:x['product']['group_id'] for x in ev if x['pool']=='PixBox'}
    pixhistory = {x['product_id']:x for x in rows('src/research_results/phase102_20261013/product_location_index.csv') if x['source']=='PixBox5036991'}
    for x in pix:
        p=x['product_id']; cache=p in pixp
        h=pixhistory[p]
        assert (h['status']=='candidate_metadata_pass') == cache
        add('PixBox:'+p,'Zenodo5036991','author point validation; historical external pool',p,x['tile'],x['sensing_key'],json.dumps(x.get('geo'),separators=(',',':')),pixgroup.get(p,'pending'),'inherited group for cached16; otherwise pending','metadata product identity; native raster not local','unknown','historical external evaluation; not CloudSEN12 Source training evidence','R4/M/B cached scores for16 only; native B04/B03/B8A not present locally','point class join previously locked; no new reference values read','original full-tile windows cached; no local fresh inference input','B' if cache else 'C','cached external-transfer description only; not source holdout' if cache else 'Phase102 excluded_overlap: '+h['exclusion_reasons'])
        overlaps[-1]['spatial_conclusion'] += '; inherited Phase102=' + h['status'] + '; history_matches=' + h['history_matches']
    alcd=js('data/sentinel2_alcd_1460961/imagery/product_manifest.json')['scenes']
    for x in alcd:
        q=x['parameters']; p=q['cloudy_product_name'];tile=q['tile']
        add('ALCD:'+x['key'],'Zenodo1460961','author reference collection',p,tile,q['cloudy_date'],'metadata original tile; precise crop footprint pending',tile,'tile only; broader correlation pending','official used_parameters original product','unknown','historical label/version review; not part of208fit','B02/B03/B04/B08/B11/B12 only; B8A missing;B08 not interchangeable','2/3Cloud;4Shadow;5/6/7Surface;ignore0/1;RF-derived reference','label60m;Omni20m input/context not frozen','C','missing fixed B8A; label semantics/context/overlap incomplete')
    assert len(manifest)==535+975+513+29+37
    writecsv('CANDIDATE_MANIFEST.csv',manifest);writecsv('OVERLAP_AUDIT.csv',overlaps)
    stats={'cloudsen':splitstats,'manifest_rows':len(manifest),'admission':dict(collections.Counter(x['admission'] for x in manifest)),'dataset_admission':{ds:dict(collections.Counter(x['admission'] for x in manifest if x['dataset']==ds)) for ds in sorted({x['dataset'] for x in manifest})},'fit_groups':len(fitg),'cal_groups':len(calg),'Catalogue_dev_products':len(devp),'PixBox_cached_products':len(pixp),'PixBox_cached_groups':len(set(pixgroup.values())),'new_successful_source_bytes':sum(r['bytes'] for r in sources),'V4_map_features':len(mapdata['features']),'V4_map_tiles':len(v4map),'duplicate_map_tiles_identical_properties':['01VCE','01WCQ'],'V4_high_map_sum':16980,'V4_high_train_per_tile_2x_consistent':True,'pixels_read':False,'neural_forward_calls':0,'training':0,'fitting':0,'new_performance_calculation':0,'notes':['Group count is known-component upper bound across train+val+test connectivity, not certified independent support.','Cross-CRS spatial checks explicitly pending; cannot promote any source holdout.','Public training map is current and not bound to pinned V4 weight revision.']}
    save('AUDIT_COUNTS.json',stats)
    timing=js('outputs/phase208/phase208_20261020/BENCHMARK_RATIOS.json')['equal_product_and_repeat_mean_GPU_ms']
    costs=[]
    for split,n in [('val',535),('test',975)]:
        costs.append({'candidate':'CloudSEN12 '+split+' (NOT ADMITTED)','new_download':'required;raw3bands alone '+str(n*512*512*2*3)+' bytes;surrounding context additional unknown','R4_calls_nominal':n,'M_calls_nominal':n,'B_calls_nominal':n,'conditional_calls_each':f'{n}..{9*n}; constructor/warmup extra unknown','R4_GPU_seconds_surrogate':n*timing['R4']/1000,'M_GPU_seconds_surrogate':n*timing['M']/1000,'B_GPU_seconds_surrogate':n*timing['B']/1000,'basis':'one512 window/chip nominal;high-nodata auto patch reduction may yield up to9;no pixels inspected;2081000-window surrogate NOT measured512 speed','context':'5.09km native;strict20km1000at20m not executable','answers':'only after inputs/protocol fixed: historical/source-like retention description','cannot_claim':'strict V4 holdout/independence/actual support or speed'})
    for name,n in [('Catalogue development',32),('PixBox historical',16)]:
        costs.append({'candidate':name,'new_download':'none for existing sealed score reuse only','R4_calls_nominal':0,'M_calls_nominal':0,'B_calls_nominal':0,'conditional_calls_each':'0 only existing scores; no new evaluation executed','R4_GPU_seconds_surrogate':0,'M_GPU_seconds_surrogate':0,'B_GPU_seconds_surrogate':0,'basis':f'208 R4/M/B cached outputs for{n}products;score bytes unopened','context':'original cached protocol','answers':'already available target/external transfer description','cannot_claim':'source holdout retention or new independent evidence'})
    writecsv('FUTURE_COST_ESTIMATE.csv',costs)
    save('IDENTITY_LOCK.json',{'V4':js('outputs/phase208/phase208_20261020/delivery/V4_WEIGHT_LOCK.json'),'M_B':js('outputs/phase208/phase208_20261020/delivery/TRAINING_SEAL.json'),'comparison':'R4 vs M vs B; E4 is not preadaptation member baseline'})
    # Hash text evidence and protocol implementation, never data pixels or model bytes.
    for rel in ['src/tools/audit_phase211_admission.py','src/tools/fetch_phase211_sources.py','src/dataset/cloudsen12_high.py','src/tools/convert_datasets/create_l1c_l2a.py','src/configs/protocol/phase64_source_parent_rgb.py','src/configs/protocol/phase22_clean_v8_l1c.py','src/configs/_base_/datasets/cloudsen12_high_l1c.py','src/research_plans/PHASE211_OMNI_RETENTION_SET_ADMISSION_20261020.md','src/research_plans/V4_IDENTITY_AND_SCOPE_CHECK_20261019.md','src/research_plans/OMNICLOUDMASK_THREE_PAPERS_FULLTEXT_REVIEW_20261017.md','src/research_results/phase109_20261013/sources/omnicloudmask__cloud_mask.py','src/research_results/phase210_20261020/README.md','outputs/phase210/phase210_20261020/THRESHOLD_REFERENCE.json','outputs/phase208/phase208_20261020/DELIVERY_COMPLETE.json','src/research_results/phase102_20261013/metadata_audit.json']:
        read(rel)
    save('INPUT_CODE_SOURCE_HASHES.json',INPUTS)
    save('METADATA_IO_RECEIPT.json',{'guard':'Python audit hook rejects raster/array/weight opens and all ZIP except geometry-only shapefiles.zip','opened_file_paths':sorted(OPENED),'PNG_operation':'glob/name/stat only, no image or label byte reads','network_forward_calls':0,'training_fitting':0,'performance_calculation':0,'cross_CRS_spatial_status':'pending, no certification of disjointness','small_metadata_successful_bytes':stats['new_successful_source_bytes'],'failed_HTTP_wire_bytes':'not measured; failed map API404 retained in fetch manifest; no large downloads'})
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

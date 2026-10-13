"""Fixed PixBox reference definition; no models, imagery, fitting or metrics."""
import csv,hashlib,json,math
from collections import Counter,defaultdict
from decimal import Decimal,InvalidOperation
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'src/research_results/phase103_20261013'
DATA=ROOT/'data/pixbox_s2_cmix_5036991/phase103_20261013'
REQUIRED=('CLOUD_CHARACTERISTICS_ID','SHADOW_ID','SURFACE_ID')
VALID={'CLOUD_CHARACTERISTICS_ID':set(range(13)),'SHADOW_ID':set(range(5)),'SURFACE_ID':set(range(36))}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def csvread(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvwrite(p,rows,fields=None):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def code(v):
    if v is None or not str(v).strip():return None
    try:
        d=Decimal(str(v))
        return int(d) if d.is_finite() and d==d.to_integral_value() else None
    except (InvalidOperation,ValueError,OverflowError):return None

def classify(attrs):
    values={k:code(attrs.get(k)) for k in REQUIRED};reasons=[]
    for k in REQUIRED:
        if attrs.get(k) is None or not str(attrs[k]).strip():reasons.append('missing_'+k)
        elif values[k] not in VALID[k]:reasons.append('unknown_code_'+k)
    if reasons:return 'ignore',reasons,values
    C,S,U=[values[k] for k in REQUIRED]
    if U in {9,15,16}:reasons.append('spatial_cloud_mixture_SURFACE_ID')
    if S==4:reasons.append('shadow_above_cloud')
    if reasons:return 'ignore',reasons,values
    if C==0:reasons.append('C0_not_clear')
    if S==0:reasons.append('S0_not_shadow_absence')
    if C==8:reasons.append('spatially_mixed_cloud')
    if C==10:reasons.append('fog')
    if C==11:reasons.append('haze')
    if C==12:reasons.append('cloud_border')
    if reasons:return 'ignore',reasons,values
    if C in {1,7} and S==3:return 'positive_ground_cloud_shadow',[],values
    if C in {1,7} and S in {1,2}:return 'negative_surface',[],values
    if C in {2,3,4,5,6,9} and S==1:return 'negative_cloud',[],values
    if C in {2,3,4,5,6,9} and S in {2,3}:reasons.append('cloud_surface_shadow_conflict')
    else:reasons.append('unsupported_attribute_combination')
    return 'ignore',reasons,values

def summarize(locations,cells):
    counts=Counter(x['reference'] for x in locations)
    valid=[x for x in locations if x['reference']!='ignore' and not x['position_pending']]
    v=Counter(x['reference'] for x in valid)
    c_support={}
    for c in [1,7]:
        selected=[x for x in valid if x['C']==c]
        c_support[str(c)]={'effective_positions':len(selected),
            'positive':sum(x['reference']=='positive_ground_cloud_shadow' for x in selected),
            'surface_negative':sum(x['reference']=='negative_surface' for x in selected)}
    all_reasons=Counter(reason for x in locations for reason in x['ignore_reasons'])
    primary_reasons=Counter(x['ignore_reasons'][0] for x in locations if x['ignore_reasons'])
    cluster=Counter(x['cluster_1km'] for x in locations if not x['position_pending'])
    support={'raw_points':sum(len(x['point_ids']) for x in locations),'deduplicated_positions':len(locations),
        'duplicates_merged_extra_rows':sum(len(x['point_ids'])-1 for x in locations if not x['duplicate_attribute_conflict']),
        'duplicate_attribute_conflict_positions':sum(x['duplicate_attribute_conflict'] for x in locations),
        'duplicate_attribute_conflict_rows':sum(len(x['point_ids']) for x in locations if x['duplicate_attribute_conflict']),
        'valid_positive':v['positive_ground_cloud_shadow'],'valid_surface_negative':v['negative_surface'],
        'valid_cloud_negative':v['negative_cloud'],
        'terrain_tree_building_shadow_negative_subset':sum(x['reference']=='negative_surface' and x['S']==2 for x in valid),
        'valid_negative_total':v['negative_surface']+v['negative_cloud'],
        'ignored_positions':sum(x['reference']=='ignore' or x['position_pending'] for x in locations),
        'ignore_all_reasons_nonexclusive':dict(all_reasons),'ignore_primary_reason_exclusive':dict(primary_reasons),
        'position_pending':sum(x['position_pending'] for x in locations),'C1_C7_effective_support':c_support,
        '20m_cells':len(cells),'20m_positive_negative_conflict_cells':sum(x['positive_negative_conflict'] for x in cells),
        '20m_multisample_cells':sum(x['deduplicated_positions']>1 for x in cells),
        'max_positions_per20m_cell':max([x['deduplicated_positions'] for x in cells] or [0]),
        'max_raw_points_per20m_cell':max([x['raw_point_count'] for x in cells] or [0]),
        'unique_fixed_1km_spatial_cells':len(cluster),
        'max_positions_per_fixed_1km_spatial_cell':max(cluster.values(),default=0),
        'largest_fixed_1km_cell_position_fraction':max(cluster.values(),default=0)/sum(cluster.values()) if cluster else None,
        'can_compute_sampled_point_AP':v['positive_ground_cloud_shadow']>0 and v['negative_surface']+v['negative_cloud']>0}
    assert support['valid_positive']+support['valid_negative_total']+support['ignored_positions']==len(locations)
    return support

def main():
    import io,zipfile,shutil
    lock=read(OUT/'prelabel_lock.json')
    assert lock['label_values_accessed_before_lock'] is False and lock['semantic_gate']=='no_material_conflict_found_for_fixed_conservative_task'
    for name,h in lock['code_sha256'].items():assert sha(ROOT/name)==h,(name,'prelabel code mismatch')
    for name,h in lock['input_sha256'].items():assert sha(ROOT/name)==h,(name,'prelabel metadata mismatch')
    assert sha(OUT/'reference_mapping.json')==lock['mapping_sha256']
    DATA.mkdir(parents=True,exist_ok=True)
    archive=ROOT/'outputs/phase102/phase102_20261013/acquired/PixBox-S2-CMIX.zip'
    assert sha(archive)==lock['source_zip_sha256']
    original=DATA/'original_PixBox-S2-CMIX.zip'
    if not original.exists():shutil.copyfile(archive,original)
    assert sha(original)==lock['source_zip_sha256']
    positions={p['point_id']:p for p in csvread(DATA/'coordinate_metadata.csv')}
    members={m['product_link']:m for m in lock['members']}
    selected=[];opaque_skipped=0
    with zipfile.ZipFile(original) as z:
        member=next(n for n in z.namelist() if n.endswith('.csv'))
        with z.open(member) as stream:
            header=stream.readline().decode('utf-8-sig').strip().split(',')
            assert header[:2]==['ID','PRODUCT_ID']
            category_fields=header[6:]
            with (DATA/'allowed_original_rows.csv').open('xb') as filtered:
                filtered.write((','.join(header)+'\n').encode())
                for raw in stream:
                    if not raw.strip():continue
                    # Inspect only ID/product-link bytes before whitelist; excluded remainder stays opaque.
                    first=raw.split(b',',2)
                    if len(first)<3:raise ValueError('Malformed row before product whitelist')
                    product_link=first[1].decode('ascii').strip()
                    if product_link not in members:opaque_skipped+=1;continue
                    values=next(csv.reader([raw.decode('utf-8-sig').strip()]))
                    assert len(values)==len(header)
                    attrs=dict(zip(header,values));pid=attrs['ID'];p=positions[pid]
                    assert attrs['PRODUCT_ID']==p['product_link']
                    for a,b in [('PIXEL_X','pixel_x'),('PIXEL_Y','pixel_y'),('LATITUDE','latitude'),('LONGITUDE','longitude')]:assert attrs[a]==p[b]
                    reference,reasons,parsed=classify(attrs)
                    selected.append({'point_id':pid,'product_id':members[product_link]['product_id'],
                        'group_id':members[product_link]['group_id'],'attrs':attrs,
                        'category_tuple':tuple(attrs[k].strip() for k in category_fields),
                        'reference':reference,'ignore_reasons':reasons,'C':parsed[REQUIRED[0]],'S':parsed[REQUIRED[1]],'U':parsed[REQUIRED[2]],
                        'position':p})
                    filtered.write(raw.rstrip(b'\r\n')+b'\n')
    assert len(selected)==len(positions) and {x['point_id'] for x in selected}==set(positions)
    duplicate_groups=defaultdict(list)
    for x in selected:
        p=x['position'];key=(x['product_id'],str(code(p['pixel_x'])) if code(p['pixel_x']) is not None else p['pixel_x'],str(code(p['pixel_y'])) if code(p['pixel_y']) is not None else p['pixel_y'])
        duplicate_groups[key].append(x)
    locations=[];duplicates=[]
    for key,items in sorted(duplicate_groups.items()):
        first=items[0];p=first['position'];conflict=len({x['category_tuple'] for x in items})>1
        pending=any(x['position']['position_pending']=='True' for x in items)
        reasons=['duplicate_attribute_conflict'] if conflict else list(first['ignore_reasons'])
        ref='ignore' if conflict else first['reference']
        if pending:reasons.append('position_pending');ref='ignore'
        row={'product_id':key[0],'group_id':first['group_id'],'pixel_x':key[1],'pixel_y':key[2],
            'point_ids':sorted(x['point_id'] for x in items),'reference':ref,'ignore_reasons':reasons,
            'C':first['C'] if not conflict else None,'S':first['S'] if not conflict else None,'U':first['U'] if not conflict else None,
            'duplicate_attribute_conflict':conflict,'position_pending':pending,
            'col10':p['col10'],'row10':p['row10'],'col20':p['col20'],'row20':p['row20'],
            'cluster_1km':p['cluster_1km'],
            'original_attributes':[x['attrs'] for x in items]}
        locations.append(row)
        if len(items)>1:duplicates.append(row)
    cell_groups=defaultdict(list)
    for x in locations:
        if not x['position_pending']:cell_groups[(x['product_id'],x['col20'],x['row20'])].append(x)
    cells=[]
    for key,items in sorted(cell_groups.items()):
        combo=Counter(x['reference'] for x in items)
        cells.append({'product_id':key[0],'group_id':items[0]['group_id'],'col20':key[1],'row20':key[2],
            'deduplicated_positions':len(items),'raw_point_count':sum(len(x['point_ids']) for x in items),
            'label_combination':dict(combo),'positive_negative_conflict':combo['positive_ground_cloud_shadow']>0 and combo['negative_surface']+combo['negative_cloud']>0,
            'point_ids':sorted(pid for x in items for pid in x['point_ids']),
            '20m_truth_assigned':False})
    write(DATA/'parsed_allowed_rows.json',selected)
    write(DATA/'deduplicated_locations.json',locations)
    write(DATA/'duplicate_conflicts_full.json',duplicates)
    write(DATA/'20m_cell_combinations_full.json',cells)
    product_support=[];group_support=[]
    for m in lock['members']:
        ls=[x for x in locations if x['product_id']==m['product_id']];cs=[x for x in cells if x['product_id']==m['product_id']]
        product_support.append({'product_id':m['product_id'],'group_id':m['group_id'],**summarize(ls,cs)})
    for gid in sorted({m['group_id'] for m in lock['members']}):
        ls=[x for x in locations if x['group_id']==gid];cs=[x for x in cells if x['group_id']==gid]
        group_support.append({'group_id':gid,'products':sorted(m['product_id'] for m in lock['members'] if m['group_id']==gid),**summarize(ls,cs)})
    csvwrite(OUT/'per_product_support.csv',product_support);csvwrite(OUT/'per_group_support.csv',group_support)
    csvwrite(OUT/'duplicates_and_position_conflicts.csv',[
        {'product_id':r['product_id'],'group_id':r['group_id'],
         **{k:r[k] for k in ['raw_points','deduplicated_positions','duplicates_merged_extra_rows','duplicate_attribute_conflict_positions','duplicate_attribute_conflict_rows','position_pending','20m_cells','20m_positive_negative_conflict_cells','20m_multisample_cells','max_positions_per20m_cell']}} for r in product_support])
    report={'status':'completed_fixed_reference_support_only','total_support':summarize(locations,cells),
        'candidate_products':16,'candidate_groups':10,'AP_computable_groups':sum(r['can_compute_sampled_point_AP'] for r in group_support),
        'AP_or_recall_or_FPR_computed':False,'labels_are_published_reference_not_new_verified_truth':True,
        '20m_labels_expanded_or_majority_voted':False,'new_calibration_split':False,
        'new_training_fits_predictions_imagery_downloads':0,'code_mapping_membership_changed_after_label_access':False,
        'prelabel_lock_sha256':sha(OUT/'prelabel_lock.json'),
        'controlled_data_SHA256':{p.name:sha(p) for p in DATA.iterdir() if p.is_file()},
        'access_record':{'labels_interpreted_products':sorted(m['product_id'] for m in lock['members']),
            'allowed_rows_semantically_parsed':len(selected),'excluded_rows_remainder_opaque':opaque_skipped,
            'physical_ZIP_stream_contains_all29_products':True,'excluded13_category_values_interpreted':False,
            'raw_and_derived_class_tables_in_git':False,'historical_Phase102_original_copy_retained_unmodified':True,
            'Phase103_labels_transferred_to_remote':False},
        'limits':['selected-point reference only','unknown sampling weights','no dense mIoU or scene FPR support','group count not pixel count','C1 operational absence interpretation distinct from explicit C7 Clear']}
    write(OUT/'support_report.json',report)
    print(json.dumps({'status':report['status'],'raw_allowed_points':len(selected),'deduplicated_positions':len(locations),
                      'positive':report['total_support']['valid_positive'],'negative':report['total_support']['valid_negative_total'],
                      'AP_computable_groups':report['AP_computable_groups']}))

if __name__=='__main__':main()

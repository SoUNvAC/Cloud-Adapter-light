"""Independent stdlib counts on the local controlled16-product CSV only."""
import csv,hashlib,json
from collections import Counter,defaultdict
from decimal import Decimal
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'src/research_results/phase103_20261013'
DATA=ROOT/'data/pixbox_s2_cmix_5036991/phase103_20261013'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def integer(text):
    try:
        v=Decimal(text);return int(v) if v.is_finite() and v==v.to_integral_value() else None
    except Exception:return None

def main():
    lock=read(BASE/'prelabel_lock.json');orig=rows(DATA/'allowed_original_rows.csv')
    allow={m['product_link']:m for m in lock['members']};positions={x['point_id']:x for x in rows(DATA/'coordinate_metadata.csv')}
    # Explicit disjoint truth table independent of the main classify() function.
    table={(c,s):'positive_ground_cloud_shadow' for c in (1,7) for s in (3,)}
    table.update({(c,s):'negative_surface' for c in (1,7) for s in (1,2)})
    table.update({(c,1):'negative_cloud' for c in (2,3,4,5,6,9)})
    projected={};raw_groups=defaultdict(list)
    for r in orig:
        assert r['PRODUCT_ID'] in allow
        c,s,u=[integer(r[x]) for x in ['CLOUD_CHARACTERISTICS_ID','SHADOW_ID','SURFACE_ID']]
        valid=c in range(13) and s in range(5) and u in range(36) if c is not None and s is not None and u is not None else False
        ref=table.get((c,s),'ignore') if valid and u not in (9,15,16) and s!=4 else 'ignore'
        projected[r['ID']]=ref
        key=(allow[r['PRODUCT_ID']]['product_id'],str(integer(r['PIXEL_X'])) if integer(r['PIXEL_X']) is not None else r['PIXEL_X'],str(integer(r['PIXEL_Y'])) if integer(r['PIXEL_Y']) is not None else r['PIXEL_Y']);raw_groups[key].append(r)
    parsed=read(DATA/'parsed_allowed_rows.json');assert len(parsed)==len(orig)==len(positions)
    assert all(projected[r['point_id']]==r['reference'] for r in parsed)
    loc=read(DATA/'deduplicated_locations.json');actual={(x['product_id'],x['pixel_x'],x['pixel_y']):x for x in loc}
    assert set(actual)==set(raw_groups)
    for key,rs in raw_groups.items():
        # Compare all categorical attributes, not only the target three.
        attrs=set(tuple(r[k].strip() for k in list(r)[6:]) for r in rs)
        conflict=len(attrs)>1;pending=any(positions[r['ID']]['position_pending']=='True' for r in rs)
        ref='ignore' if conflict or pending else projected[rs[0]['ID']]
        a=actual[key];assert a['duplicate_attribute_conflict']==conflict and a['position_pending']==pending and a['reference']==ref
        assert a['point_ids']==sorted(r['ID'] for r in rs)
    group=rows(BASE/'per_group_support.csv');product=rows(BASE/'per_product_support.csv')
    for report,field in [(group,'group_id'),(product,'product_id')]:
        for r in report:
            subset=[x for x in loc if x[field]==r[field]];n=Counter(x['reference'] for x in subset)
            assert int(r['raw_points'])==sum(len(x['point_ids']) for x in subset)
            assert int(r['deduplicated_positions'])==len(subset)
            assert int(r['valid_positive'])==n['positive_ground_cloud_shadow']
            assert int(r['valid_surface_negative'])==n['negative_surface'] and int(r['valid_cloud_negative'])==n['negative_cloud']
            assert int(r['ignored_positions'])==n['ignore']
            assert int(r['terrain_tree_building_shadow_negative_subset'])==sum(x['reference']=='negative_surface' and x['S']==2 for x in subset)
            ignore=Counter(reason for x in subset for reason in x['ignore_reasons'])
            assert json.loads(r['ignore_all_reasons_nonexclusive'])==dict(ignore)
            for c,summary in json.loads(r['C1_C7_effective_support']).items():
                eligible=[x for x in subset if x['C']==int(c) and x['reference']!='ignore']
                assert summary['effective_positions']==len(eligible)
                assert summary['positive']==sum(x['reference']=='positive_ground_cloud_shadow' for x in eligible)
                assert summary['surface_negative']==sum(x['reference']=='negative_surface' for x in eligible)
    # Independent20m grouping and conflict presence, with no cell truth assignment.
    cells=defaultdict(list)
    for x in loc:
        if not x['position_pending']:cells[(x['product_id'],x['col20'],x['row20'])].append(x)
    saved=read(DATA/'20m_cell_combinations_full.json');assert len(saved)==len(cells)
    for x in saved:
        key=(x['product_id'],x['col20'],x['row20']);combo=Counter(r['reference'] for r in cells[key])
        assert x['label_combination']==dict(combo)
        assert x['positive_negative_conflict']==bool(combo['positive_ground_cloud_shadow'] and combo['negative_surface']+combo['negative_cloud'])
        assert not x['20m_truth_assigned']
    report=read(BASE/'support_report.json')
    assert report['AP_computable_groups']==sum(int(r['valid_positive'])>0 and int(r['valid_negative_total'])>0 for r in group)
    for name,h in report['controlled_data_SHA256'].items():assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==h
    receipt={'status':'passed','independent_truth_table_rows':len(orig),'deduplicated_positions':len(loc),
        'products':len(product),'groups':len(group),'20m_cells_checked':len(saved),
        'all_group_product_counts_match':True,'C1_C7_counts_match':True,'duplicate_attributes_checked_all_categories':True,
        'all_controlled_data_SHA256_match':True,'excluded13_labels_interpreted':False,'raw_derived_labels_transferred':False,
        'model_scores_loaded_or_metrics_computed':False,'independent_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (BASE/'CPU_INDEPENDENT_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()

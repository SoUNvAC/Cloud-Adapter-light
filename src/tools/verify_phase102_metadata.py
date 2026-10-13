"""Independent CPU checks of fixed metadata identities, components and exclusion reasons."""
import csv,hashlib,json
from collections import defaultdict,deque
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'src/research_results/phase102_20261013'

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    candidates=load(BASE/'inputs/candidate_products_metadata.json')
    history=load(BASE/'inputs/history_positions_only.json')['records']
    index=rows(BASE/'product_location_index.csv');by={r['record_key']:r for r in candidates}
    assert len(index)==len(by)==1108
    tiles={h['tile'] for h in history};ids={h['product'] for h in history}
    for r in index:
        reasons=json.loads(r['exclusion_reasons']);c=by[r['record_key']]
        assert ('same_tile_any_date' in reasons)==(c['tile'] in tiles)
        assert ('same_product' in reasons)==(c['product_id'] in ids)
        assert bool(reasons)==(r['status']=='excluded_overlap')
        if r['status']=='candidate_metadata_pass':
            assert c['metadata_complete'] and c['grid'] and r['source']=='PixBox5036991'
            assert r['spatial_unknown']=='False'
    # BFS components independently verify union-find output; unknown geometry is not completed.
    edges=load(BASE/'correlation_edges.json');groups=rows(BASE/'related_groups.csv')
    for source in ['KappaSet7100327','PixBox5036991']:
        remaining={r['record_key'] for r in index if r['source']==source and r['status']!='excluded_overlap'}
        adj={x:set() for x in remaining}
        for e in edges:
            if e['left'] in adj and e['right'] in adj:adj[e['left']].add(e['right']);adj[e['right']].add(e['left'])
        expected=[];visited=set()
        for x in sorted(remaining):
            if x in visited:continue
            q=deque([x]);members=set()
            while q:
                v=q.popleft()
                if v in visited:continue
                visited.add(v);members.add(v);q.extend(adj[v]-visited)
            expected.append(sorted(members))
        actual=sorted(sorted(json.loads(g['members'])) for g in groups if g['source']==source)
        assert sorted(expected)==actual
    access=load(BASE/'inputs/source_access_sha256.json')
    conservative_total=access['received_body_bytes_recorded']+access['aborted_batch_conservative_body_upper_bound']+access['browser_text_reserve_bytes']
    assert conservative_total<access['limit_bytes']
    positions=rows(BASE/'inputs/pixbox_positions_only.csv')
    assert len(positions)==17351
    assert set(positions[0])=={'point_id','product_link','pixel_x','pixel_y','latitude','longitude'}
    # Repeat exact2km-distance contact checks with Phase86 vector transforms, no raster I/O.
    from audit_phase86 import buffered_intersects,project_geometry
    from shapely.geometry import box
    a=project_geometry(box(499000,5000000,500000,5001000),'EPSG:32632','EPSG:4326')
    near=project_geometry(box(501999.9,5000000,502999.9,5001000),'EPSG:32632','EPSG:4326')
    far=project_geometry(box(502000.1,5000000,503000.1,5001000),'EPSG:32632','EPSG:4326')
    assert buffered_intersects(a,near,'EPSG:32632') and not buffered_intersects(a,far,'EPSG:32632')
    checks=load(BASE/'coordinate_grid_checks.json');assert len(checks)==29
    audit=load(BASE/'metadata_audit.json')
    for name,h in audit['input_SHA256'].items():assert hashlib.sha256((BASE/'inputs'/name).read_bytes()).hexdigest()==h
    receipt={'status':'passed','identity_records':1108,'Kappa_keys_not_certified_product_count':1079,
        'PixBox_products':29,'coordinate_records_allowlist_only':17351,
        'history_identity_tile_exclusion_verified':True,'BFS_components_match':True,
        'fixed_1000m_each_buffer_contact_test_passed':True,'all_input_hashes_match':True,
        'existing_history_footprint_geometry_lineage':'Phase86 positions-only metadata; no historical raster reopened',
        'coordinate_grid_checks':len(checks),'coordinate_pass_count':sum(c['passed'] for c in checks),
        'conservative_metadata_transfer_budget_bytes':conservative_total,
        'training_fits_prediction_performance_evaluation':0,'raster_or_class_values_interpreted':False}
    (BASE/'CPU_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()

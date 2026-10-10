"""Preserve original intake; fix PSD long-ID linkage using the SAME verified XML."""
import csv,json
from pathlib import Path
from phase75_metadata import parse_xml,write,utc
from prepare_phase65a_explore import digest
from run_phase75 import ROOT,PROTOCOL


def main():
    original=json.loads((ROOT/'metadata_audit.json').read_text())
    original_hashes=json.loads((ROOT/'artifact_hashes.json').read_text())
    for n,h in original_hashes.items():assert digest(ROOT/n)==h,n
    out=ROOT/'reparse01';out.mkdir(exist_ok=False)
    write(out/'input_lock.json',dict(created_utc=utc(),reason='PSD long granuleIdentifier/TILE_ID differs from compact IMAGE_FILE directory; verify explicit identity links',
        original_artifact_hashes_sha256=digest(ROOT/'artifact_hashes.json'),original_input_lock_sha256=digest(ROOT/'input_lock.json'),
        code_sha256={n:digest(Path(__file__).with_name(n)) for n in ['phase75_metadata.py','reparse_phase75.py','test_phase75.py']},
        protocol_sha256=digest(PROTOCOL),new_http_requests=0,new_raster_downloads=0,no_original_report_overwrite=True))
    rows=[]
    for before in original['rows']:
        folder=ROOT/before['product'];objects=json.loads((folder/'object_listing.json').read_text())['items']
        prefix=before['exact_product_candidates'][0]
        parsed=parse_xml((folder/'MTD_MSIL1C.xml').read_bytes(),(folder/'MTD_TL.xml').read_bytes(),before['product'],prefix,objects)
        row={k:before[k] for k in ['product','group_id','public_shadow_percent','selection_sha256','cube_chain','exact_product_candidates']}
        row.update(parsed);row.update(cube_mapping_status='pending',direction_geometry_ready=False,missing=row['cube_chain']['missing'],
            annotation_matches_native_crs=row['cube_chain']['annotation_epsg']==parsed['source_grid']['crs_code'],
            xml_object_metadata=dict(product=next(o for o in objects if o['name']==prefix+'MTD_MSIL1C.xml'),
                tile=next(o for o in objects if o['name'].endswith('/MTD_TL.xml'))))
        assert row['annotation_matches_native_crs']
        rows.append(row)
    report={**original,'status':'metadata_reparsed_without_new_downloads','rows':rows,
        'solar_metadata_verified':sum(r['solar_metadata_verified'] for r in rows),
        'source_grid_verified':sum(r['source_grid_verified'] for r in rows),
        'reparse_input_lock_sha256':digest(out/'input_lock.json'),'additional_HTTP_bytes':0}
    write(out/'metadata_audit.json',report)
    flat=[dict(product=r['product'],group_id=r['group_id'],solar_azimuth_deg=r['solar'].get('azimuth'),solar_zenith_deg=r['solar'].get('zenith'),
        source_crs=r['source_grid']['crs_code'],solar_metadata_verified=r['solar_metadata_verified'],source_grid_verified=r['source_grid_verified'],
        cube_mapping_status=r['cube_mapping_status'],direction_geometry_ready=False,missing='; '.join(r['missing'])) for r in rows]
    with (out/'metadata_audit.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    write(out/'artifact_hashes.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))

if __name__=='__main__':main()

"""Phase110: verify original B8A metadata and frozen identity, no raster/model I/O."""
import argparse,base64,hashlib,json,pathlib,re,time,xml.etree.ElementTree as ET

def main(root):
    start=time.process_time();repo=pathlib.Path(__file__).resolve().parents[2]
    frozen_path=repo/'src/research_results/phase104_20261013/object_manifest.json'
    old=json.loads(frozen_path.read_text(encoding='utf-8'));expected={x['product_id']:x for x in old['products']}
    new=json.loads((root/'B8A_OBJECTS.json').read_text(encoding='utf-8'));assert new['count']==16 and len(new['products'])==16
    assert {x['product_id'] for x in new['products']}==set(expected)
    assert len({x['name'] for x in new['products']})==16
    results=[]
    for x in new['products']:
        pid=x['product_id'];m=expected[pid];p=root/'sources'/(pid+'_B8A_metadata.json');raw=p.read_bytes();meta=json.loads(raw)
        assert hashlib.sha256(raw).hexdigest()==x['object_metadata_sha256']
        assert meta['name']==x['name'] and meta['generation']==x['generation']
        assert int(meta['size'])==x['bytes'] and meta['md5Hash']==x['md5_base64']
        assert len(base64.b64decode(x['md5_base64']))==16
        assert meta['crc32c']==x['crc32c_base64'] and len(base64.b64decode(x['crc32c_base64']))==4
        assert pid+'.SAFE/' in x['name'] and x['name'].endswith('_B8A.jp2')
        assert x['processing_baseline'].replace('.','')==re.search(r'_N(\d{4})_',pid).group(1)
        assert x['native_resolution_m']==20 and x['xml_band_id']=='8' and x['grid20']==m['grid20'] and x['crs']==m['crs']
        results.append({'product_id':pid,'original_identity':True,'declared_checksum_only':True,'grid20_and_baseline':True})
    assert sum(x['bytes'] for x in new['products'])==new['declared_total_bytes']==521604536
    bindings=json.loads((root/'XML_BINDINGS.json').read_text())
    for x in bindings:
        pid=x['product_id'];m=expected[pid]
        for path_key,sha_key,expected_key in [('xml_path','xml_sha256','product_xml_sha256'),('tile_prefix_path','tile_prefix_sha256','original_XML_prefix_sha256'),('listing_path','listing_sha256','gcs_listing_sha256')]:
            p=repo/x[path_key];assert hashlib.sha256(p.read_bytes()).hexdigest()==x[sha_key]==m[expected_key]
        product=ET.fromstring((repo/x['xml_path']).read_bytes());values=lambda tag:[e.text for e in product.iter() if e.tag.split('}')[-1]==tag]
        assert values('PRODUCT_URI')==[pid+'.SAFE'] and x['B8A_image_file'] in values('IMAGE_FILE')
        b=[e for e in product.iter() if e.tag.split('}')[-1]=='Spectral_Information' and e.attrib.get('physicalBand')=='B8A'];assert len(b)==1
        assert [e.text for e in b[0].iter() if e.tag.split('}')[-1]=='RESOLUTION']==['20']
        geo=ET.fromstring(re.search(r'<Tile_Geocoding\b.*?</Tile_Geocoding>',(repo/x['tile_prefix_path']).read_text(),re.S).group())
        size=geo.find("Size[@resolution='20']");pos=geo.find("Geoposition[@resolution='20']")
        assert all(float(e.text)==m['grid20'][e.tag] for parent in [size,pos] for e in parent)
    report={'phase':110,'status':'metadata_only_verified','objects':16,'declared_B8A_bytes':521604536,'unchanged_phase104_manifest_sha256':hashlib.sha256(frozen_path.read_bytes()).hexdigest(),'products':results,'XML_bindings_verified':16,'raster_reads':0,'weight_loads':0,'forwards':0,'performance_evaluations':0,'checksum_content_verified':False,'validator_process_cpu_seconds':time.process_time()-start}
    (root/'CPU_METADATA_VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='products'}))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,default=pathlib.Path('outputs/phase110/phase110_20261014'));main(ap.parse_args().root)

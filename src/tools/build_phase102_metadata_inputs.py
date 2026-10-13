"""Project public metadata into a label-value-free Phase102 CPU audit input."""
import csv,hashlib,json,re,shutil,xml.etree.ElementTree as E
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
LOCAL=ROOT/'outputs/phase102/phase102_20261013'
ACQ=LOCAL/'acquired'
DEST=ROOT/'src/research_results/phase102_20261013'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvwrite(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    inp=DEST/'inputs';inp.mkdir(parents=True,exist_ok=True)
    files=[]
    for p in sorted(ACQ.iterdir()):
        if p.is_file():files.append({'local_path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p),
            'scope':'bundle obtained with labels; local only; not copied to git or remote' if p.suffix=='.zip' else 'metadata/description/index/XML or projected coordinates only'})
    meta=[]
    for rec in [7100327,5036991,5095024,11121064]:
        d=read(ACQ/f'record_{rec}.json');m=d['metadata']
        meta.append({'requested_record':rec,'resolved_record':d['id'],'doi':m.get('doi'),
                     'title':m['title'],'license':m.get('license'),'version':m.get('version'),
                     'related_identifiers':m.get('related_identifiers'),
                     'files':[{'key':f['key'],'bytes':f['size'],'published_checksum':f['checksum']} for f in d['files']],
                     'role':'candidate' if rec in [7100327,5036991] else 'lineage_only_not_additional_pool',
                     'record_sha256':sha(ACQ/f'record_{rec}.json')})
    write(inp/'source_records.json',meta)
    patch=[];acquisitions={}
    for row in read(ACQ/'kappaset_archive_index.json'):
        if not row['path'].endswith('.nc'):continue
        m=re.search(r'T(\d{2}[A-Z]{3})_(\d{8}T\d{6})_tile_(\d+)_(\d+)\.nc$',row['path'])
        if not m:raise ValueError('Unresolved Kappa filename '+row['path'])
        tile,t,x,y=m.groups();key=f'Kappa:T{tile}:{t}'
        patch.append({'source':'KappaSet7100327','record_key':key,'archive_path':row['path'],
                      'tile':tile,'sensing_utc_key':t,'index_x':x,'index_y':y,
                      'declared_pixel_offset_x':int(x)*512,'declared_pixel_offset_y':int(y)*512,
                      'basis':'official README: indices multiplied by512; axes/origin/original product grid not verified',
                      'precise_footprint':'pending_metadata; NetCDF not obtained'})
        acquisitions[key]={'source':'KappaSet7100327','record_key':key,'product_id':'unknown_original_product',
             'tile':tile,'sensing_key':t,'utc_day':t[:8],'orbit':'',
             'location_evidence':'archive filename + official README offset indices; parent grid unresolved',
             'footprint_precision':'MGRS tile/acquisition key only; no precise sub-tile polygon',
             'six_band_availability':'official declares B02/B03/B04/B08/B11/B12 in NetCDF; actual arrays unopened',
             'processing_baseline':'unknown','toa_basis':'unknown actual NetCDF scaling/offset; do not assume DN/10000',
             'geo':None,'crs':'','grid':None,'metadata_complete':False,'parent_proxy':False}
    # A feasibility-only CDSE parent lookup is retained as a proxy, never original-grid proof.
    probe=ACQ/'cdse_kappa_probe.json'
    if probe.exists():
        for v in read(probe).get('value',[]):
            m=re.search(r'_MSIL1C_(\d{8}T\d{6})_N\d+_R(\d+)_T(\d{2}[A-Z]{3})_',v['Name'])
            if m:
                t,orbit,tile=m.groups();key=f'Kappa:T{tile}:{t}'
                if key in acquisitions:acquisitions[key].update(geo=v['GeoFootprint'],orbit=orbit,parent_proxy=True,
                    location_evidence='CDSE reprocessed same-acquisition parent proxy; original baseline/grid still unresolved')
    csvwrite(inp/'kappa_patch_positions.csv',patch)
    names=re.findall(r'(\d+);(S2[AB]_MSIL1C_\S+)\.SAFE',
                     (ACQ/'pixbox_description.txt').read_text(encoding='utf-8'))
    products=list(acquisitions.values())
    for i,(link,n) in enumerate(names):
        match=re.fullmatch(r'S2[AB]_MSIL1C_(\d{8}T\d{6})_(N\d+)_R(\d+)_T(\d{2}[A-Z]{3})_\d{8}T\d{6}',n)
        assert match;t,baseline,orbit,tile=match.groups()
        item={'source':'PixBox5036991','record_key':n,'product_id':n,'product_link':link,'tile':tile,
              'sensing_key':t,'utc_day':t[:8],'orbit':orbit,
              'location_evidence':'official product-link mapping + WGS84/PIXEL_X/Y only',
              'footprint_precision':'sample coordinates known; original product grid pending',
              'six_band_availability':'pending','processing_baseline':baseline,'toa_basis':'pending XML',
              'geo':None,'crs':'','grid':None,'metadata_complete':False,'parent_proxy':False}
        listing=ACQ/f'gcs_pixbox_{i:02}.json';tilexml=ACQ/f'pixbox_{i:02}_tile_prefix.xml';productxml=ACQ/f'pixbox_{i:02}_product.xml'
        if listing.exists():
            objects=read(listing).get('items',[])
            bands={re.search(r'_(B\d{2})\.jp2$',x['name'])[1]:{'bytes':int(x['size']),'md5_base64':x['md5Hash'],'generation':x['generation']} for x in objects if re.search(r'_(B02|B03|B04|B08|B11|B12)\.jp2$',x['name'])}
            item['six_band_availability']='six objects indexed; content NOT downloaded' if len(bands)==6 else 'pending missing objects'
            item['band_object_metadata']=bands
        if tilexml.exists() and productxml.exists():
            text=tilexml.read_text(encoding='utf-8');g=re.search(r'<Tile_Geocoding[^>]*>.*?</Tile_Geocoding>',text,re.S)
            assert g,'Geocoding element not wholly inside bounded prefix'
            grid=E.fromstring(g[0]);item['crs']=grid.findtext('HORIZONTAL_CS_CODE')
            gp=grid.find("Geoposition[@resolution='10']");size=grid.find("Size[@resolution='10']")
            item['grid']={x:float(gp.findtext(x)) for x in ['ULX','ULY','XDIM','YDIM']}
            item['grid'].update({x:int(size.findtext(x)) for x in ['NROWS','NCOLS']})
            product=E.parse(productxml).getroot();vals={e.tag.split('}')[-1]:(e.text or '').strip() for e in product.iter()}
            assert vals['PRODUCT_URI']==n+'.SAFE'
            item['processing_baseline']=vals['PROCESSING_BASELINE'];item['quantification']=float(vals['QUANTIFICATION_VALUE'])
            item['offset_elements']=[(e.tag.split('}')[-1],e.text) for e in product.iter() if 'OFFSET' in e.tag]
            item['special_values']={s.findtext('SPECIAL_VALUE_TEXT'):s.findtext('SPECIAL_VALUE_INDEX') for s in product.iter() if s.tag.split('}')[-1]=='Special_Values'}
            item['toa_basis']='original L1C XML quantification; no RADIO_ADD_OFFSET element in these old-baseline metadata' if not item['offset_elements'] else 'see recorded offsets'
            item['location_evidence']='exact-original SAFE prefix + PRODUCT_URI match + XML UTM grid and WGS84 sample metadata'
            item['footprint_precision']='original full109.8km parent tile, conservatively excludes entire tile; point positions retained'
            item['metadata_complete']=len(item.get('band_object_metadata',{}))==6
        products.append(item)
    write(inp/'candidate_products_metadata.json',products)
    shutil.copyfile(ACQ/'pixbox_positions_only.csv',inp/'pixbox_positions_only.csv')
    for p in (LOCAL/'inputs').iterdir():shutil.copyfile(p,inp/p.name)
    ledger=read(ACQ/'access_ledger.json');write(inp/'access_ledger.json',ledger)
    write(inp/'source_access_sha256.json',{'sources':files,'received_body_bytes_recorded':sum(e['body_bytes_received'] for e in ledger),
       'aborted_batch_conservative_body_upper_bound':111*80001,
       'browser_text_reserve_bytes':1000000,'limit_bytes':20000000,
       'note':'Aborted OR-batch requests are conservatively charged all111 planned request caps in addition to ledger; browser page payloads not independently metered.',
       'label_access':{'PixBox_archive_obtained':True,'csv_member_present':True,'CSV_header_read':True,
                       'projected_fields':['ID','PRODUCT_ID','PIXEL_X','PIXEL_Y','LATITUDE','LONGITUDE'],
                       'category_value_conversion_display_counts_or_selection':False,
                       'raw_csv_extracted_to_file':False,'archive_transferred_to_remote':False,
                       'Kappa_NetCDF_or_masks_obtained':False}})
    print(json.dumps({'patch_records':len(patch),'kappa_acquisition_keys':len(acquisitions),'pixbox_products':len(names),
                      'pixbox_original_metadata_complete':sum(x['metadata_complete'] for x in products if x['source'].startswith('PixBox')),
                      'recorded_transfer':sum(e['body_bytes_received'] for e in ledger)}))

if __name__=='__main__':main()

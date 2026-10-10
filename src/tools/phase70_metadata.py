"""Existing selected-product sidecars only; annotation CRS is not a cube affine."""
import csv,hashlib,struct,zipfile
from prepare_phase65a_explore import digest


def audit(data,prep,selected,root,write):
    rows=[]
    with zipfile.ZipFile(data/'shapefiles.zip') as z:
        names=set(z.namelist())
        for product,r in selected.items():
            prefix='shapefiles/'+product+'/'+product
            evidence={};annotation={}
            for extension in ['.prj','.shp','.dbf']:
                path=prefix+extension
                if path not in names:continue
                # No annotation attributes, label fractions or label pixels used here.
                with z.open(path) as f:b=f.read() if extension=='.prj' else f.read(4096)
                evidence[extension]=dict(member=path,header_sha256=hashlib.sha256(b).hexdigest())
                if extension=='.prj':annotation['crs_wkt']=b.decode('utf-8').strip()
                elif extension=='.shp':annotation['xy_bounds']=list(struct.unpack('<4d',b[36:68]))
                elif extension=='.dbf':
                    end=int.from_bytes(b[8:10],'little')
                    annotation['dbf_fields']=[b[i:i+11].split(bytes([0]))[0].decode('ascii') for i in range(32,min(end-1,len(b)-31),32)]
            fields=dict(product_id='verified',solar_azimuth='missing',solar_zenith='missing',cube_crs='missing',
                cube_pixel_size='missing',cube_raster_orientation='missing',cube_georeferencing_correspondence='missing',
                cube_product_correspondence='verified',annotation_crs='verified' if annotation.get('crs_wkt') else 'missing')
            row=dict(product=product,group_id=r['group_id'],status='not_ready',fields=fields,annotation=annotation,evidence=evidence,
                cube_source='subscenes/'+product+'.npy',cube_member_sha256=r['member_sha256']['image'],
                correspondence_source='prepared manifest product and official ZIP member name/SHA',
                missing_reason='Existing NPY cube has no CRS/affine or solar angles; annotation sidecar CRS/bounds alone cannot establish cube grid or angles')
            rows.append(row)
    report=dict(status='not_ready',products=len(rows),rows=rows,ready_products=0,
        inspected_sources=['prepared manifest.json','subscenes selected product NPY metadata','selected shapefiles .prj/.shp/.dbf headers','classification_tags.csv field names'],
        annotation_georeferencing_not_assumed_cube_georeferencing=True,new_downloads=0,projection_or_geometry_fits=0,
        cloud_height='Unknown; any future search range requires a new protocol; no projection performed')
    write(root/'metadata_readiness.json',report)
    flat=[dict(product=r['product'],group_id=r['group_id'],status=r['status'],**r['fields'],
        sources=';'.join(e['member'] for e in r['evidence'].values()),cube_source=r['cube_source'],reason=r['missing_reason']) for r in rows]
    with (root/'metadata_readiness.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    return report

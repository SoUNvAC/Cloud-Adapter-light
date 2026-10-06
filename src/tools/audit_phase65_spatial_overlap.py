"""Source/new-target footprint audit using already-verified metadata only.

Requires existing rasterio/Shapely plus pyshp installed in --dependency-root.
Buffers native UTM geometries by 100m before geographic transformation, so
adjacent tiles/projection zones are checked too. No target image/mask pixels read.
This does NOT freeze splits or certify historical target lineage.
"""
import argparse
import csv
import io
import json
import numbers
import sys
import zipfile
from pathlib import Path, PurePosixPath
from phase65a_freeze import allowed, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-mapping', type=Path, required=True)
    p.add_argument('--catalogue-root', type=Path, required=True)
    p.add_argument('--dependency-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(allowed(a.dependency_root)))
    import shapefile
    from rasterio.crs import CRS
    from rasterio.warp import transform_geom
    from shapely.geometry import shape, mapping
    from shapely import wkt
    from shapely.strtree import STRtree
    source_root, root, output = map(allowed, (a.source_root, a.catalogue_root, a.output))
    if output.exists():
        raise ValueError('Existing spatial audit; inspect before rerunning')
    source_mapping = json.loads(allowed(a.source_mapping).read_text())
    if source_mapping['status'] != 'verified_rgb_ordinal_mapping':
        raise ValueError('Actual source PNG/raw ordinal mapping is not verified')
    manifest = json.loads(allowed(source_root / 'source_metadata_manifest.json').read_text())
    status = json.loads(allowed(root / 'download_status.json').read_text())
    published = status['verified_files']['shapefiles.zip']
    archive_path = allowed(root / 'shapefiles.zip')
    if published['zip_crc'] != 'passed' or digest(archive_path) != published['sha256']:
        raise ValueError('Shapefile package not verified')

    def geographic(geometry, crs):
        crs = CRS.from_user_input(crs)
        epsg = crs.to_epsg()
        if not epsg or not (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760):
            raise ValueError('Expected native metre-based UTM CRS')
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError('Invalid native footprint')
        result = shape(transform_geom(crs, 'EPSG:4326', mapping(geometry.buffer(100)),
                                      antimeridian_cutting=True))
        if result.is_empty or not result.is_valid:
            raise ValueError('Invalid transformed footprint')
        if result.bounds[2] - result.bounds[0] > 180:
            raise ValueError('Unresolved antimeridian footprint; fail closed')
        return result

    geometries = {}
    for split in ('train', 'val'):
        path = allowed(source_root / f'{split}_metadata.csv')
        if digest(path) != manifest['splits'][split]['sha256']:
            raise ValueError('Source CSV hash mismatch')
        with path.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                key = (row['proj_epsg'], row['proj_geometry'])
                if key not in geometries:
                    geometries[key] = geographic(wkt.loads(row['proj_geometry']), f'EPSG:{row["proj_epsg"]}')
    source = list(geometries.values())
    tree = STRtree(source)
    with allowed(root / 'classification_tags.csv').open(newline='', encoding='utf-8') as stream:
        products = [row['scene'] for row in csv.DictReader(stream) if row['shadows_marked'] == '1']
    findings = []
    with zipfile.ZipFile(archive_path) as archive:
        members = {PurePosixPath(n).name: n for n in archive.namelist() if not n.endswith('/')}
        for product in products:
            # The verified release has one shapefile per product. No extraction.
            components = {suffix: archive.read(members[product + suffix])
                          for suffix in ('.shp', '.shx', '.dbf', '.prj')}
            with shapefile.Reader(shp=io.BytesIO(components['.shp']),
                                  shx=io.BytesIO(components['.shx']),
                                  dbf=io.BytesIO(components['.dbf'])) as reader:
                shapes = reader.shapes()
                if len(shapes) != 1:
                    raise ValueError('Expected exactly one subscene footprint')
                target = geographic(shape(shapes[0].__geo_interface__), components['.prj'].decode())
            hits = tree.query(target)
            count = sum(target.intersects(source[int(hit)] if isinstance(hit, numbers.Integral) else hit)
                        for hit in hits)
            findings.append(dict(product=product, source_footprint_overlap_count=int(count),
                                 source_footprint_overlap=bool(count),
                                 buffered_wgs84_bounds=list(target.bounds)))
    report = dict(status='source_spatial_audited_split_not_locked', native_buffer_m=100,
                  source_unique_footprints=len(source), target_shadow_valid_products=len(findings),
                  source_spatial_overlap_products=sum(row['source_footprint_overlap'] for row in findings),
                  source_mapping_sha256=digest(a.source_mapping), shapefiles_sha256=digest(archive_path),
                  target_tags_sha256=digest(root / 'classification_tags.csv'), findings=findings,
                  target_pixels_read=False, training_authorized=False,
                  pending=['historical_target_footprints', 'related_product_group_split_lock',
                           'complete_acquisition', 'array_and_loader_audit'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'findings'}))


if __name__ == '__main__':
    main()

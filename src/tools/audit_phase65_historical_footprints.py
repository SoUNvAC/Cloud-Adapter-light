"""Conservative historical L8/SPARCS geographic exclusion, using TIFF headers.

Only target_train/target_val paths are opened; sealed test rows supply IDs only.
Entire historical raster bounds +100m are used, not selected pixel windows. New
target bounds are already buffered in the verified source spatial audit. This
can overexclude but does not choose scenes using model outcomes or label pixels.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from phase65a_freeze import allowed, digest


def development_rows(path):
    used, sealed = [], set()
    with allowed(path).open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['new_split'] in ('target_train', 'target_val'):
                used.append(row)
            elif row['new_split'] == 'target_test':
                sealed.add(row['scene'])
            else:
                raise ValueError('Unknown historical split')
    return used, sealed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--l8-manifest', type=Path, required=True)
    p.add_argument('--l8-raw-root', type=Path, required=True)
    p.add_argument('--sparcs-manifest', type=Path, required=True)
    p.add_argument('--target-spatial-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import rasterio
    from rasterio.warp import transform_bounds
    from shapely.geometry import box
    target = json.loads(allowed(a.target_spatial_audit).read_text())
    if target['status'] != 'source_spatial_audited_split_not_locked':
        raise ValueError('Source spatial metadata audit not complete')
    l8, l8_sealed = development_rows(a.l8_manifest)
    sparcs, sparcs_sealed = development_rows(a.sparcs_manifest)
    raw = allowed(a.l8_raw_root)
    paths = {}
    for row in l8:
        scene = row['scene']
        path = allowed(raw / 'l8biome' / row['biome'] / scene / f'{scene}.TIF')
        paths[('l8', scene)] = path
    for row in sparcs:
        path = allowed(row['multispectral_path'])
        key = ('sparcs', row['scene'])
        if key in paths and paths[key] != path:
            raise ValueError('Multiple SPARCS raster paths for one historical scene')
        paths[key] = path
    footprints, unresolved = [], []
    for (domain, scene), path in sorted(paths.items()):
        if scene in (l8_sealed if domain == 'l8' else sparcs_sealed):
            raise ValueError('Historical split leakage; test raster forbidden')
        try:
            with rasterio.open(path) as dataset:
                if not dataset.crs or not dataset.crs.is_projected:
                    raise ValueError('Known projected CRS required for native metre buffer')
                epsg = dataset.crs.to_epsg()
                units, metre_factor = dataset.crs.linear_units_factor
                if not math.isfinite(metre_factor) or metre_factor <= 0:
                    raise ValueError('Unknown projected-unit scale')
                native_buffer = 100 / metre_factor
                b = dataset.bounds
                bounds = transform_bounds(dataset.crs, 'EPSG:4326',
                                          b.left - native_buffer, b.bottom - native_buffer,
                                          b.right + native_buffer, b.top + native_buffer, densify_pts=101)
                if not all(math.isfinite(v) for v in bounds) or not -90 <= bounds[1] <= bounds[3] <= 90:
                    raise ValueError('Invalid geographic bounds after reprojection')
                metadata = dict(domain=domain, scene=scene, raster_path=str(path),
                                native_epsg=epsg, native_crs_wkt=dataset.crs.to_wkt(),
                                native_units=units, native_unit_metre_factor=metre_factor,
                                raster_shape=[dataset.height, dataset.width],
                                buffered_wgs84_bounds=list(bounds))
            left, bottom, right, top = bounds
            rectangles = [box(left, bottom, right, top)] if right >= left else [
                box(left, bottom, 180, top), box(-180, bottom, right, top)]
            footprints.append((metadata, rectangles))
        except (OSError, ValueError, rasterio.errors.RasterioError) as error:
            unresolved.append(dict(domain=domain, scene=scene, error=str(error)))
    findings = []
    for row in target['findings']:
        rectangle = box(*row['buffered_wgs84_bounds'])
        overlapping = [f'{meta["domain"]}:{meta["scene"]}' for meta, rectangles in footprints
                       if any(rectangle.intersects(g) for g in rectangles)]
        findings.append(dict(product=row['product'], historical_overlap=bool(overlapping),
                             overlapping_used_scenes=overlapping))
    report = dict(status='historical_headers_audited_split_not_locked' if not unresolved else
                  'blocked_unresolved_historical_footprints',
                  historical_used_scenes=len(paths), resolved_scenes=len(footprints),
                  unresolved=unresolved, historical_overlap_products=sum(r['historical_overlap'] for r in findings),
                  native_buffer_m=100, raster_pixels_read=False, test_pixels_read=False,
                  sealed_scene_ids_only=True, training_authorized=False,
                  input_sha256={str(allowed(path)): digest(path) for path in
                                (a.l8_manifest, a.sparcs_manifest, a.target_spatial_audit)},
                  historical_metadata=[meta for meta, _ in footprints], findings=findings,
                  pending=['related_product_group_split_lock', 'complete_acquisition',
                           'array_and_loader_audit'])
    output = allowed(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ('findings', 'historical_metadata')}))


if __name__ == '__main__':
    main()

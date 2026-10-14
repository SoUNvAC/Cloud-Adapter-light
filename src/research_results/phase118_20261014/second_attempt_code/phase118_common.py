"""Frozen no-label input grids and author-compatible scale scheduling."""
import csv
import hashlib
import json
import threading
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import binary_dilation

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/home/scv/shared/phase118_20261014')
BASE = Path('/home/scv/shared/phase111_ocm_runtime')
RAW = Path('/home/scv/shared/data/pixbox_s2_cmix_5036991/phase104_20261013')
COORD = Path('/home/scv/shared/phase103_positions_20261013/coordinate_metadata.csv')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p)+'.writing.'+str(threading.get_ident()))
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


def ram():
    return {s.split(':')[0]: int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
            if s.startswith(('MemTotal:', 'MemAvailable:'))}


def coordinates():
    assert sha(COORD) == 'c2f029a349096738935d9239585c67ca6dfc5a4c71017f5e8634af982b86ce68'
    with COORD.open(encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        assert not set(reader.fieldnames) & {'reference', 'class', 'label', 'prediction'}
        rows = list(reader)
    assert len(rows) == 9500 and len({r['point_id'] for r in rows}) == 9500
    return rows


def image(product, setting, points):
    """Bilinear uint16 author radiometry; conservative native invalid support."""
    pid = product['product_id']
    original = load(RAW/'download_status.json')['files']
    b8 = load(REPO/'src/research_results/phase112_20261014/DOWNLOAD_STATUS.json')['files']
    if pid in b8:
        b8_path, b8_sha = Path(b8[pid]['path']), b8[pid]['sha256']
    else:
        entry = next(e for e in load(REPO/'src/research_results/phase111_20261014/ASSET_VERIFICATION.json')['files']
                     if e['path'] == str(BASE/'assets/B8A.jp2'))
        b8_path, b8_sha = BASE/'assets/B8A.jp2', entry['sha256']
    target_grid = product['grid10' if setting == 'H10' else 'grid20']
    shape = (target_grid['NROWS'], target_grid['NCOLS'])
    arrays, validity, bindings = [], [], []
    for band in ('B04', 'B03', 'B8A'):
        p = b8_path if band == 'B8A' else RAW/'imagery'/pid/(band+'.jp2')
        expected = b8_sha if band == 'B8A' else original[pid+'/'+band+'.jp2']['sha256']
        assert sha(p) == expected
        native_grid = product['grid20' if band == 'B8A' else 'grid10']
        with rasterio.open(p) as ds:
            assert ds.count == 1 and ds.dtypes == ('uint16',)
            assert ds.crs.to_string() == product['crs']
            assert (ds.height, ds.width) == (native_grid['NROWS'], native_grid['NCOLS'])
            assert tuple(ds.transform)[:6] == (native_grid['XDIM'], 0, native_grid['ULX'], 0, native_grid['YDIM'], native_grid['ULY'])
            native = ds.read(1)
            valid_native = (native != 0) & (native != 65535)
            if ds.shape == shape:
                a, v = native, valid_native
            elif setting == 'C20':
                a = ds.read(1, out_shape=shape, resampling=Resampling.bilinear)
                v = (~binary_dilation(~valid_native, structure=np.ones((3, 3), bool))).reshape(shape[0], 2, shape[1], 2).all((1, 3))
            else:
                assert band == 'B8A' and shape == (10980, 10980)
                a = ds.read(1, out_shape=shape, resampling=Resampling.bilinear)
                v = np.repeat(np.repeat(~binary_dilation(~valid_native, structure=np.ones((3, 3), bool)), 2, 0), 2, 1)
                # Independent aligned bilinear interpolation at the fixed original10m cells.
                yy = np.array([int(q['row10']) for q in points]); xx = np.array([int(q['col10']) for q in points])
                fy, fx = yy/2-.25, xx/2-.25
                y0, x0 = np.floor(fy).astype(int), np.floor(fx).astype(int)
                wy, wx = fy-y0, fx-x0
                y1, x1 = (y0+1).clip(0, ds.height-1), (x0+1).clip(0, ds.width-1)
                y0, x0 = y0.clip(0, ds.height-1), x0.clip(0, ds.width-1)
                calc = native[y0,x0]*(1-wy)*(1-wx)+native[y1,x0]*wy*(1-wx)+native[y0,x1]*(1-wy)*wx+native[y1,x1]*wy*wx
                eligible = v[yy,xx]
                error = float(np.max(np.abs(a[yy,xx][eligible].astype(float)-calc[eligible]))) if eligible.any() else 0
                assert error <= 1, 'bilinear native DN quantization mismatch'
                bindings.append({'band': 'B8A_independent_bilinear_point_check', 'max_DN_error': error,
                                 'rounding_bound_DN': 1, 'original20m_observation_not_new10m_information': True})
            a = a.copy()
            a[~v] = 0
            arrays.append(a); validity.append(v)
            bindings.append({'band': band, 'path': str(p), 'sha256': expected, 'native_grid': native_grid,
                             'target_grid': target_grid, 'bilinear_uint16': ds.shape != shape})
            del native, valid_native
    for q in points:
        assert q['crs'] == product['crs']
        assert int(q['col20']) == int(q['col10'])//2 and int(q['row20']) == int(q['row10'])//2
        x, y = float(q['native_x_m']), float(q['native_y_m'])
        for resolution in (10, 20):
            g = product[f'grid{resolution}']
            assert int(np.floor((x-g['ULX'])/g['XDIM'])) == int(q[f'col{resolution}'])
            assert int(np.floor((y-g['ULY'])/g['YDIM'])) == int(q[f'row{resolution}'])
    return np.stack(arrays), np.logical_and.reduce(validity), bindings

"""Frozen three-fit-cube engineering correspondence audit; no labels or models."""
import argparse
import base64
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import urllib.parse
import urllib.request
import zipfile
import numpy as np

N = 1022
CENTERS = [32, 128, 256, 512, 768, 896, 989]
OFFSETS = [(r, c) for r in [64, 65, 66] for c in [64, 65, 66]] + [(0, 0)]
# Maps cube (r,c) to north-up crop (u,v): u=a*r+b*c+d; v=e*r+f*c+h.
D4 = [(1,0,0,0,1,0), (0,1,0,-1,0,N-1),
      (-1,0,N-1,0,-1,N-1), (0,-1,N-1,1,0,0),
      (1,0,0,0,-1,N-1), (0,-1,N-1,-1,0,N-1),
      (-1,0,N-1,0,1,0), (0,1,0,1,0,0)]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def csvwrite(p, rows):
    if not rows: return
    with Path(p).open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def orient(arr, k):
    r, c = np.indices((N, N)); a,b,d,e,f,h = D4[k]
    return arr[a*r+b*c+d, e*r+f*c+h]


def metrics(cube, ref):
    cv = np.isfinite(cube) & (cube != 0)
    rv = np.isfinite(ref)
    joint = cv & rv
    out = dict(pixels=int(cube.size), cube_valid=int(cv.sum()), native_valid=int(rv.sum()),
               joint_valid=int(joint.sum()), joint_fraction=float(joint.mean()),
               validity_agreement=float((cv == rv).mean()),
               cube_only_valid=int((cv & ~rv).sum()), native_only_valid=int((rv & ~cv).sum()))
    keys = ['median_abs', 'p99_abs', 'rmse', 'correlation', 'signed_mean', 'signed_p01', 'signed_p99', 'max_abs']
    if not joint.any(): return dict(out, **{k: None for k in keys})
    x = cube[joint].astype('float64'); y = ref[joint].astype('float64'); z = x-y
    out.update(median_abs=float(np.median(abs(z))), p99_abs=float(np.quantile(abs(z), .99)),
               rmse=float(np.sqrt(np.mean(z*z))), signed_mean=float(z.mean()),
               signed_p01=float(np.quantile(z, .01)), signed_p99=float(np.quantile(z, .99)), max_abs=float(abs(z).max()),
               correlation=float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else None)
    return out


def error_ok(m, band):
    med, p99 = (5e-6, 2e-5) if band == 'B11' else (5e-5, 2e-4)
    return m['median_abs'] is not None and m['median_abs'] <= med and m['p99_abs'] <= p99


def sample(cube, ref, band, candidate, partitions=("selection", "heldout")):
    rows = []; parts = {'selection': [], 'heldout': []}
    for i, r in enumerate(CENTERS):
        for j, c in enumerate(CENTERS):
            part = 'selection' if (i+j)%2 == 0 else 'heldout'
            if part not in partitions: continue
            x = cube[r-8:r+9,c-8:c+9]; y = ref[r-8:r+9,c-8:c+9]
            m = metrics(x, y); usable = m['joint_fraction'] >= .8
            row = dict(candidate=candidate, band=band, partition=part, row=r, col=c,
                       usable=usable, error_pass=error_ok(m,band), **m)
            rows.append(row); parts[part].append((x,y,row))
    summaries = {}
    for part, pts in parts.items():
        if not pts: continue
        # All fixed points remain in pooled residuals; no candidate-dependent deletion.
        m = metrics(np.concatenate([x.ravel() for x,y,z in pts]), np.concatenate([y.ravel() for x,y,z in pts]))
        usable = sum(z['usable'] for x,y,z in pts)
        point_gate = all(z['error_pass'] for x,y,z in pts if z['usable'])
        summaries[part] = dict(**m, usable_points=usable,
                               pass_gate=usable >= 20 and point_gate and error_ok(m,band))
    return rows, summaries


def bilinear_full(raw):
    # Exactly the 20m center between four native 10m centers; no antialias filter.
    q = raw.astype('float64')
    return (q[0::2,0::2]+q[0::2,1::2]+q[1::2,0::2]+q[1::2,1::2])*.25


def bilinear_window(raw, native_row0, native_col0, target_row, target_col, size=N):
    # Absolute target centers: native center indices (2*row+.5, 2*col+.5).
    rr = 2*(target_row+np.arange(size))+.5-native_row0
    cc = 2*(target_col+np.arange(size))+.5-native_col0
    r = np.floor(rr).astype(int); c = np.floor(cc).astype(int)
    wr = (rr-r)[:,None]; wc = (cc-c)[None,:]
    return ((1-wr)*((1-wc)*raw[r[:,None],c]+wc*raw[r[:,None],c+1]) +
            wr*((1-wc)*raw[r[:,None]+1,c]+wc*raw[r[:,None]+1,c+1]))


def download(obj, dest, budget):
    expected = int(obj['size']); url = ('https://storage.googleapis.com/gcp-public-data-sentinel-2/'+
        urllib.parse.quote(obj['name'], safe='/')+'?generation='+obj['generation'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists(): raise ValueError('Exclusive run refuses existing destination')
    partial = dest.with_suffix('.jp2.partial')
    req = urllib.request.Request(url, headers={'Accept-Encoding':'identity'})
    with urllib.request.urlopen(req, timeout=60) as response:
        gen = response.headers.get('x-goog-generation')
        if gen and gen != obj['generation']: raise ValueError('Generation mismatch')
        if int(response.headers.get('Content-Length', -1)) != expected: raise ValueError('HTTP size mismatch')
        h=hashlib.sha256(); md=hashlib.md5(); got=0
        with partial.open('xb') as f:
            while True:
                b=response.read(min(1024*1024, expected-got+1))
                if not b: break
                got+=len(b); budget[0]+=len(b)
                if got>expected or budget[0]>1_000_000_000: raise ValueError('Download budget/size exceeded')
                h.update(b);md.update(b);f.write(b)
    if got!=expected or base64.b64encode(md.digest()).decode()!=obj['md5Hash']:
        raise ValueError('Length/MD5 failure; partial retained')
    partial.rename(dest)
    return dict(url=url,size=got,md5_base64=obj['md5Hash'],generation=obj['generation'],sha256=h.hexdigest())


def verify_grid(ds, row, band):
    from affine import Affine
    g=row['source_grid']['bands'][band]['grid']
    t=Affine(g['XDIM'],0,g['ULX'],0,g['YDIM'],g['ULY'])
    if ds.crs.to_string()!=row['source_grid']['crs_code'] or ds.transform!=t or ds.shape!=(g['NROWS'],g['NCOLS']) or ds.count!=1:
        raise ValueError('Decoded native grid differs from XML')


def affine_info(row, r0, c0, k):
    from affine import Affine
    g=row['source_grid']['resolutions']['20']; a,b,d,e,f,h=D4[k]
    # Pixel edge mapping offsets corrected by half a pixel for reflections/rotations.
    to_native=Affine(f,e,c0+h+.5-(e+f)*.5,b,a,r0+d+.5-(a+b)*.5)
    native=Affine(g['XDIM'],0,g['ULX'],0,g['YDIM'],g['ULY']); t=native*to_native
    return dict(crs=row['source_grid']['crs_code'],affine_first6=list(t)[:6],
                corners=[list(t*p) for p in [(0,0),(N,0),(N,N),(0,N)]],
                first_last_centers=[list(t*p) for p in [(.5,.5),(N-.5,N-.5)]],
                formula_coefficients=list(D4[k]),
                row_formula=f'native_row={r0}+({a})*r+({b})*c+({d})',
                col_formula=f'native_col={c0}+({e})*r+({f})*c+({h})')


def scene(row, cube, paths, out):
    import rasterio
    product=row['product']; dest=out/product; dest.mkdir()
    g=row['source_grid']['resolutions']['20']; xmin,ymin,xmax,ymax=row['cube_chain']['annotation_bounds']
    baser=(ymax-g['ULY'])/g['YDIM']; basec=(xmin-g['ULX'])/g['XDIM']
    assert baser==round(baser) and basec==round(basec)
    specials=list(row['radiometry']['special_values'].values()); assert not row['radiometry']['reflectance_offsets']
    quant=float(row['radiometry']['quantification_value']); assert quant==10000
    with rasterio.open(paths['B11']) as ds:
        verify_grid(ds,row,'B11'); raw=ds.read(1)
    native=raw.astype('float64')/quant; native[np.isin(raw,specials)]=np.nan
    point_rows=[]; candidates=[]
    for dr,dc in OFFSETS:
        r0=int(baser)+dr;c0=int(basec)+dc
        cut=native[r0:r0+N,c0:c0+N]; assert cut.shape==(N,N)
        for k in range(8):
            name=f'inset_{dr}_{dc}_d4_{k}'
            ref=orient(cut,k); points,summ=sample(cube[:,:,11],ref,'B11',name,('selection',))
            # Heldout residuals are computed only for the uniquely selected candidate.
            point_rows += [p for p in points if p['partition']=='selection']
            candidates.append(dict(candidate=name,row=r0,col=c0,d4=k,**summ['selection']))
    csvwrite(dest/'B11_selection_points.csv',point_rows); csvwrite(dest/'B11_candidates.csv',candidates)
    ranked=sorted(candidates,key=lambda c:float('inf') if c['rmse'] is None else c['rmse'])
    qualified=[c for c in candidates if c['pass_gate']]
    result=dict(product=product,B11_mapping_verified=False,B04_resampling_verified=False,
                cube_mapping_status='ambiguous' if len(qualified)>1 else 'unresolved',
                selection_qualified_count=len(qualified),lowest_selection_rmse=ranked[0],
                six_band_mapping_verified=False,geographic_north_to_image_verified=False)
    if len(qualified)!=1 or qualified[0]['candidate']!=ranked[0]['candidate']:
        write(dest/'status.json',result); return result
    chosen=qualified[0];r0=chosen['row'];c0=chosen['col'];k=chosen['d4']
    ref=orient(native[r0:r0+N,c0:c0+N],k)
    points,summ=sample(cube[:,:,11],ref,'B11',chosen['candidate'])
    csvwrite(dest/'B11_heldout_points.csv',[p for p in points if p['partition']=='heldout'])
    full=metrics(cube[:,:,11],ref)
    # Whole-array gate uses same error bounds, >=80% joint support and every usable
    # fixed patch passing. Spatial quadrant/edge diagnostics are separately retained.
    fullpass=full['joint_fraction']>=.8 and error_ok(full,'B11')
    result.update(selected=chosen,B11_heldout=summ['heldout'],B11_full=full,
                  B11_mapping_verified=summ['heldout']['pass_gate'] and fullpass)
    residual=cube[:,:,11].astype('float64')-ref
    np.save(dest/'B11_full_residual.npy',residual)
    spatial=[]
    for label,sl in [('NW',(slice(0,N//2),slice(0,N//2))),('NE',(slice(0,N//2),slice(N//2,N))),
                     ('SW',(slice(N//2,N),slice(0,N//2))),('SE',(slice(N//2,N),slice(N//2,N)))]:
        m=metrics(cube[:,:,11][sl],ref[sl]);spatial.append(dict(region=label,**m))
    csvwrite(dest/'B11_spatial_diagnostics.csv',spatial)
    # Report any whole-array quadrant error anomaly conservatively as pending.
    result['B11_spatial_error_anomaly']=any(m['joint_valid'] and not error_ok(m,'B11') for m in spatial)
    result['B11_mapping_verified'] &= not result['B11_spatial_error_anomaly']
    if not result['B11_mapping_verified']:
        result['cube_mapping_status']='B11_heldout_or_full_pending';write(dest/'status.json',result);return result
    mapping=affine_info(row,r0,c0,k);result['B11_verified_mapping']=mapping
    with rasterio.open(paths['B04']) as ds:
        verify_grid(ds,row,'B04');raw=ds.read(1)
    raw=raw.astype('float64');raw[np.isin(raw,specials)]=np.nan
    # Implementation 1 resamples the entire tile, then crops.
    whole=bilinear_full(raw)/quant;one=orient(whole[r0:r0+N,c0:c0+N],k)
    # Implementation 2: exact covering window plus two native pixels each side.
    wr=2*r0-2;wc=2*c0-2
    window=raw[wr:2*(r0+N)+2,wc:2*(c0+N)+2]
    two=orient(bilinear_window(window,wr,wc,r0,c0)/quant,k)
    np.save(dest/'B04_implementation_difference.npy',one-two)
    difference=metrics(np.where(np.isfinite(one),one+1,np.nan),np.where(np.isfinite(two),two+1,np.nan))
    identical=bool(np.array_equal(np.isfinite(one),np.isfinite(two)) and np.array_equal(one[np.isfinite(one)],two[np.isfinite(two)]))
    result['B04_implementation_difference']=difference;result['B04_implementations_exactly_equal']=identical
    result['B04_absolute_northup_target_transform']=[20,0,g['ULX']+20*c0,0,-20,g['ULY']-20*r0]
    bandpasses=[]
    for label,x in [('full_grid',one),('window_plus2',two)]:
        points,summ=sample(cube[:,:,3],x,'B04',label);csvwrite(dest/f'B04_{label}_points.csv',points)
        full=metrics(cube[:,:,3],x);result['B04_'+label]=dict(partitions=summ,full=full)
        bandpasses.append(all(s['pass_gate'] for s in summ.values()) and full['joint_fraction']>=.8 and error_ok(full,'B04'))
        np.save(dest/f'B04_{label}_full_residual.npy',cube[:,:,3].astype('float64')-x)
    result['B04_resampling_verified']=identical and all(bandpasses)
    result['cube_mapping_status']='two_band_mapping_verified' if result['B04_resampling_verified'] else 'B11_verified_B04_pending'
    if result['B04_resampling_verified']:result['verified_cube_transform']=mapping
    write(dest/'status.json',result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--phase75',type=Path,required=True)
    ap.add_argument('--bindings',type=Path,required=True);ap.add_argument('--candidates',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();import rasterio
    for p in [a.data,a.phase75,a.bindings,a.candidates,a.output]:
        if not any(p.resolve().is_relative_to(Path(root)) for root in ['/home/scv/shared','/home/scv/Cloud-Adapter-light']):raise ValueError('Outside authorized roots')
    a.output.mkdir(exist_ok=False);out=a.output;report=out/'reports';report.mkdir();(out/'objects').mkdir()
    try:
        b=json.loads(a.bindings.read_text());rows=b['rows'];assert len(rows)==3 and len(b['objects'])==6
        inputs={}
        for f,h in b['phase75_inputs'].items():
            p=a.candidates if f=='cube_mapping_candidates.csv' else a.phase75/f
            if sha(p)!=h:raise ValueError('Phase75 input SHA mismatch: '+f)
            inputs[f]=h
        assert json.loads((a.phase75/'reparse01/metadata_audit.json').read_text())['rows']==rows
        manifest=a.data/'phase65a_explore_20261008/manifest.json';m=json.loads(manifest.read_text())
        cubes={}
        for r in rows:
            fit=next(x for x in m['rows'] if x['product']==r['product']);assert fit['split']=='fit' and fit['is_representative'] and fit['group_id']==r['group_id']
            cubes[r['product']]=fit['member_sha256']['image']
        total=sum(int(o['object']['size']) for o in b['objects']);assert total<=1_000_000_000
        lock=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  inputs=inputs,bindings_sha256=sha(a.bindings),code_sha256=sha(Path(__file__)),manifest_sha256=sha(manifest),cube_member_sha256=cubes,
                  objects=b['objects'],download_limit_bytes=1_000_000_000,planned_bytes=total,offsets=OFFSETS,d4_formula_coefficients=D4,
                  d4_definition='u=a*r+b*c+d; v=e*r+f*c+h; native=(base+inset)+(u,v)',centers=CENTERS,patch_width=17,
                  selection='grid index sum even (25); odd heldout (24)',support='>=20 usable points per partition, usable means joint-valid >=80%; all fixed points retained in pooled metrics; all usable points and pooled error gates must pass',
                  full_gate='joint valid >=80%; same pooled error bounds; B11 quadrant error anomalies keep pending',
                  error_gates=dict(B11=[5e-6,2e-5],B04=[5e-5,2e-4]),validity='cube corresponding channel finite and nonzero; native excludes every XML special value; bilinear requires all four valid (strict NaN propagation)',
                  radiometry='native DN / XML quantification=10000; no offsets/gain fitting; float64 arithmetic',
                  bilinear='pure center interpolation, no antialias: target native center index=(2*r+0.5,2*c+0.5); four equal weights; full tile vs window+2 using same absolute grid; require exact equality and identical validity',
                  versions=dict(numpy=np.__version__,rasterio=rasterio.__version__,GDAL=rasterio.__gdal_version__),
                  no_labels_or_predictions=True,first_jp2_pixel_read_has_not_occurred=True)
        write(report/'mapping_lock.json',lock);print('LOCK FROZEN',total,flush=True)
        budget=[0];paths={};verified=[]
        # All objects fully checked before any raster open/read.
        for o in b['objects']:
            dest=out/'objects'/o['product']/(o['band']+'.jp2')
            v=download(o['object'],dest,budget);v.update(product=o['product'],band=o['band'],path=str(dest))
            verified.append(v);paths.setdefault(o['product'],{})[o['band']]=dest
            print('OBJECT VERIFIED',o['product'],o['band'],flush=True)
        write(report/'object_verification.json',dict(files=verified,new_download_bytes=budget[0]))
        results=[]
        with zipfile.ZipFile(a.data/'subscenes.zip') as z:
            for r in rows:
                raw=z.read(r['cube_chain']['cube_member'])
                if hashlib.sha256(raw).hexdigest()!=cubes[r['product']]:raise ValueError('Full cube member SHA mismatch')
                cube=np.load(io.BytesIO(raw),allow_pickle=False);assert cube.shape==(N,N,13) and cube.dtype==np.float32
                result=scene(r,cube,paths[r['product']],report);results.append(result)
                print('SCENE COMPLETE',r['product'],result['cube_mapping_status'],flush=True)
        write(report/'summary.json',dict(status='completed',scenes=results,new_download_bytes=budget[0],no_labels_or_predictions_read=True))
        write(report/'artifact_hashes.json',{str(p.relative_to(report)):sha(p) for p in report.rglob('*') if p.is_file()})
    except Exception as e:
        write(report/'failure.json',dict(type=type(e).__name__,error=str(e)));raise


if __name__=='__main__':main()

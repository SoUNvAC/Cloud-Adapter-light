"""Frozen three-fit-scene B11 directional information screen; no fitting."""
import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np
import run_phase76 as prev

DISTANCES=(100,200,400,800,1600)
NAMES=('sun','orthogonal90','reverse','orthogonal270','all','dark')


def endpoint(rr,cc,az,distance,affine,to_geo,from_geo,geod):
    a,b,c,d,e,f=affine
    x=a*(cc+.5)+b*(rr+.5)+c;y=d*(cc+.5)+e*(rr+.5)+f
    lon,lat=to_geo.transform(x,y)
    lon2,lat2,_=geod.fwd(lon,lat,np.full(np.shape(lon),az),np.full(np.shape(lon),distance))
    x2,y2=from_geo.transform(lon2,lat2)
    det=a*e-b*d
    col=(e*(x2-c)-b*(y2-f))/det-.5
    row=(-d*(x2-c)+a*(y2-f))/det-.5
    return row,col,(lon,lat,lon2,lat2)


def bilinear(raw,rr,cc):
    """All four neighbors required even when an interpolation weight is zero."""
    r=np.floor(rr).astype(int);c=np.floor(cc).astype(int)
    good=(r>=0)&(c>=0)&(r+1<raw.shape[0])&(c+1<raw.shape[1])
    fr=rr-r;fc=cc-c;out=np.zeros(rr.shape,dtype='float64')
    for dr,wr in [(0,1-fr),(1,fr)]:
        for dc,wc in [(0,1-fc),(1,fc)]:
            v=raw[np.clip(r+dr,0,raw.shape[0]-1),np.clip(c+dc,0,raw.shape[1]-1)]
            good &= np.isfinite(v);out+=np.nan_to_num(v)*wr*wc
    return np.where(good,out,np.nan)


def evaluate(truth,score):
    from sklearn.metrics import average_precision_score,roc_auc_score
    y=truth==2;both=bool(y.any() and (~y).any())
    result=dict(AP=float(average_precision_score(y,score)) if both else None,
                ROC_AUC=float(roc_auc_score(y,score)) if both else None)
    for cls,name in [(0,'Surface'),(1,'Cloud')]:
        v=(truth==2)|(truth==cls)
        result['Shadow_vs_'+name+'_AUC']=float(roc_auc_score(y[v],score[v])) if (truth==2).any() and (truth==cls).any() else None
    return result


def validate_directions(affine,to_geo,from_geo,geod,az,n=1022):
    result=[]
    for name,r,c in [('center',n//2,n//2),('NW',80,80),('NE',80,n-81),('SW',n-81,80),('SE',n-81,n-81)]:
        offsets={}
        for angle in sorted(set([0.,90.,180.,270.]+[(az+k)%360 for k in (0,90,180,270)])):
            rr,cc,geo=endpoint(np.array(float(r)),np.array(float(c)),angle,1600.,affine,to_geo,from_geo,geod)
            actual,_,dist=geod.inv(*geo)
            err=abs((float(actual)-angle+180)%360-180)
            assert err<1e-7 and abs(float(dist)-1600)<1e-5
            dr=float(rr-r);dc=float(cc-c);offsets[round(angle,9)]=(dr,dc)
            if angle==0:assert dr<0
            if angle==90:assert dc>0
            result.append(dict(point=name,center_row=r,center_col=c,azimuth_deg=angle,distance_m=1600,
                longitude=float(geo[0]),latitude=float(geo[1]),end_longitude=float(geo[2]),end_latitude=float(geo[3]),
                delta_row=dr,delta_col=dc,geod_inverse_azimuth=float(actual)%360,azimuth_error_deg=err,distance_error_m=abs(float(dist)-1600)))
        for angle in [(az+k)%360 for k in (0,90)]:
            p=np.array(offsets[round(angle,9)]);q=np.array(offsets[round((angle+180)%360,9)])
            assert np.dot(p,q)<0 and np.linalg.norm(p+q)<.1,'Opposite geodesic endpoints inconsistent'
    return result


def plot_overview(dest,raw,truth,common,scores,tile):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    v=raw[np.isfinite(raw)];lo,hi=np.percentile(v,[2,98])
    axes.flat[0].imshow(raw,cmap='gray',vmin=lo,vmax=hi,interpolation='nearest');axes.flat[0].set_title('B11 TOA (2-98% display)')
    labels=np.where(common,truth,3)
    axes.flat[1].imshow(labels,cmap=ListedColormap(['#009E73','#E69F00','#0072B2','#999999']),vmin=0,vmax=3,interpolation='nearest')
    axes.flat[1].set_title('Reference: green Surface / orange Cloud\nblue Shadow / gray excluded')
    directional=np.concatenate([scores[k][common] for k in NAMES[:5]])
    lim=max(float(np.percentile(np.abs(directional),98)),1e-10)
    for ax,name in zip(list(axes.flat)[2:],NAMES):
        vmin,vmax=(-lim,lim) if name!='dark' else tuple(np.percentile(scores[name][common],[2,98]))
        im=ax.imshow(np.where(common,scores[name],np.nan),cmap='RdBu_r' if name!='dark' else 'viridis',vmin=vmin,vmax=vmax,interpolation='nearest')
        ax.set_title(name+' (higher = more Shadow)');fig.colorbar(im,ax=ax,shrink=.72)
    for ax in axes.flat:ax.set_axis_off()
    fig.suptitle(tile+' | display stretch only; all six scores share the same evaluation mask',fontsize=14)
    fig.savefig(dest/'overview.png',dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--bindings',type=Path,required=True);a=ap.parse_args()
    import rasterio,pyproj,sklearn
    for p in (a.data,a.output,a.bindings):assert any(p.resolve().is_relative_to(Path(x)) for x in ('/home/scv/shared','/home/scv/Cloud-Adapter-light'))
    a.output.mkdir(exist_ok=False);out=a.output
    try:
        bindings=json.loads(a.bindings.read_text());oldroot=a.data/'phase76_20261010/reports'
        for rel,h in bindings['previous'].items():assert prev.sha(oldroot/rel)==h,rel
        oldlock=json.loads((oldroot/'mapping_lock.json').read_text());old=json.loads((oldroot/'summary.json').read_text())
        artifact_hashes=json.loads((oldroot/'artifact_hashes.json').read_text())
        assert prev.sha(Path(prev.__file__))==oldlock['code_sha256']
        p=Path('src/research_plans/PHASE76_INPUT_BINDINGS_20261010.json');assert prev.sha(p)==oldlock['bindings_sha256']
        rows=json.loads(p.read_text())['rows'];assert len(rows)==3
        meta=a.data/'phase75_20261010/reparse01/metadata_audit.json';assert prev.sha(meta)==oldlock['inputs']['reparse01/metadata_audit.json']
        assert json.loads(meta.read_text())['rows']==rows
        manifest=a.data/'phase65a_explore_20261008/manifest.json';assert prev.sha(manifest)==oldlock['manifest_sha256']
        fit=json.loads(manifest.read_text())['rows'];selected=[]
        for row in rows:
            r=next(x for x in fit if x['product']==row['product']);assert r['split']=='fit' and r['is_representative'] and r['group_id']==row['group_id'];selected.append(r)
        objects=json.loads((oldroot/'object_verification.json').read_text())['files'];paths={}
        for o in objects:
            if o['band']=='B11':
                p=Path(o['path']);assert p.resolve().is_relative_to((a.data/'phase76_20261010/objects').resolve()) and prev.sha(p)==o['sha256'];paths[o['product']]=p
        transforms=json.loads((oldroot/'B11_only_transforms.json').read_text())
        lock=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            code_sha256=prev.sha(Path(__file__)),helper_sha256=prev.sha(Path(prev.__file__)),bindings_sha256=prev.sha(a.bindings),
            previous=bindings['previous'],selected=[dict(product=r['product'],group_id=r['group_id'],member_sha256=r['member_sha256']) for r in selected],
            B11_objects=[o for o in objects if o['band']=='B11'],transforms=transforms,distances_m=DISTANCES,directions_degrees=[0,90,180,270],
            angle='tile mean geographic North clockwise; toward Sun; not per-pixel or shadow projection',
            coordinate_chain='B11-only affine pixel center -> always_xy CRS to EPSG:4326 -> WGS84 Geod.fwd -> native CRS -> inverse affine minus .5',
            sampling='native 20m verified B11 cube extent only; pure bilinear; all four neighbors valid even zero weight; no reflect/clamp/pad/product stitching',
            formula='S_direction=max(five B11 samples)-center; S_all=max(all 20)-center; S_dark=-center; positive=Shadow',
            validity='center valid native B11 and Catalogue B11 finite/nonzero; native XML special values excluded; all20 samples valid; label-independent common mask; evaluate intersection with original one-hot reference',
            classes=['Surface','Cloud','Shadow'],AP='sklearn average_precision_score; undefined if either binary class missing => null',
            continuation='sun minus reverse/orthogonal90/orthogonal270 mean AP each >0 and each positive in >=2 scenes; sun minus dark mean AP >0; all reported without substitution',
            scope='three fixed fit scenes only; Phase77 unresolved; no download, fitting, model scores or confirmation; do not extrapolate outside common mask',
            versions=dict(numpy=np.__version__,pyproj=pyproj.__version__,PROJ=pyproj.proj_version_str,rasterio=rasterio.__version__,sklearn=sklearn.__version__),new_scores_not_yet_computed=True)
        prev.write(out/'input_formula_lock.json',lock)
        rawscenes={};reproductions=[]
        with zipfile.ZipFile(a.data/'subscenes.zip') as z:
            for row in rows:
                product=row['product'];r=next(x for x in selected if x['product']==product)
                b=z.read(row['cube_chain']['cube_member']);assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
                cube=np.load(io.BytesIO(b),allow_pickle=False);assert cube.shape==(1022,1022,13)
                prior=next(x for x in old['scenes'] if x['product']==product);assert prior['B11_mapping_verified'] and prior['selected']['candidate']=='inset_65_65_d4_0'
                r0=prior['selected']['row'];c0=prior['selected']['col']
                with rasterio.open(paths[product]) as ds:
                    prev.verify_grid(ds,row,'B11');raw=ds.read(1,window=rasterio.windows.Window(c0,r0,1022,1022))
                    transform=next(x for x in transforms if x['product']==product)
                    assert list(ds.window_transform(rasterio.windows.Window(c0,r0,1022,1022)))[:6]==transform['affine_first6']
                ref=raw.astype('float64')/float(row['radiometry']['quantification_value']);ref[np.isin(raw,list(row['radiometry']['special_values'].values()))]=np.nan
                ref[~(np.isfinite(cube[:,:,11])&(cube[:,:,11]!=0))]=np.nan
                residual=cube[:,:,11].astype('float64')-ref
                residual_path=oldroot/product/'B11_full_residual.npy'
                assert prev.sha(residual_path)==artifact_hashes[str(residual_path.relative_to(oldroot))]
                saved=np.load(residual_path,allow_pickle=False)
                assert np.array_equal(residual,saved,equal_nan=True),'B11 correspondence changed'
                reproductions.append(dict(product=product,exact_previous_residual_reproduction=True,**prev.metrics(cube[:,:,11],ref)))
                rawscenes[product]=ref
        prev.write(out/'B11_reproduction.json',reproductions);print('ALL THREE B11 BASELINES REPRODUCED',flush=True)
        metrics=[];coverage=[];diffs=[];directions=[]
        for row in rows:
            product=row['product'];tile=product.split('_')[5];dest=out/product;dest.mkdir();raw=rawscenes[product]
            geo=next(x for x in transforms if x['product']==product);affine=geo['affine_first6']
            to_geo=pyproj.Transformer.from_crs(geo['crs'],'EPSG:4326',always_xy=True);from_geo=pyproj.Transformer.from_crs('EPSG:4326',geo['crs'],always_xy=True);geod=pyproj.Geod(ellps='WGS84');az=row['solar']['azimuth']
            directions += [dict(product=product,**x) for x in validate_directions(affine,to_geo,from_geo,geod,az)]
            rr,cc=np.indices(raw.shape,dtype='float64');scores={};common=np.isfinite(raw)
            for k,name in enumerate(NAMES[:4]):
                samples=[]
                for distance in DISTANCES:
                    r,c,_=endpoint(rr,cc,(az+90*k)%360,distance,affine,to_geo,from_geo,geod);samples.append(bilinear(raw,r,c))
                samples=np.stack(samples);common &= np.isfinite(samples).all(axis=0);scores[name]=np.max(samples,axis=0)-raw
            scores['all']=np.max(np.stack([scores[n] for n in NAMES[:4]]),axis=0);scores['dark']=-raw
            # Common support is frozen without reading labels.
            np.save(dest/'common_mask.npy',common);np.savez_compressed(dest/'native_scores.npz',**scores)
            prev.write(dest/'common_mask_freeze.json',dict(sha256=prev.sha(dest/'common_mask.npy'),label_pixels_not_yet_read=True,valid_centers=int(np.isfinite(raw).sum()),common_centers=int(common.sum())))
            r=next(x for x in selected if x['product']==product)
            with zipfile.ZipFile(a.data/'masks.zip') as z:b=z.read('masks/'+product+'.npy')
            assert hashlib.sha256(b).hexdigest()==r['member_sha256']['mask'];mask=np.load(io.BytesIO(b),allow_pickle=False);assert mask.shape==(1022,1022,3) and mask.dtype==np.bool_
            label_valid=mask.sum(-1)==1;truth=mask.argmax(-1).astype('uint8');valid=common&label_valid
            np.save(dest/'evaluation_mask.npy',valid)
            counts=np.bincount(truth[valid],minlength=3);orig=np.bincount(truth[label_valid],minlength=3)
            coverage.append(dict(product=product,tile=tile,B11_valid_centers=int(np.isfinite(raw).sum()),common_centers=int(common.sum()),original_reference_valid=int(label_valid.sum()),evaluation_centers=int(valid.sum()),coverage=float(valid.sum()/label_valid.sum()),Surface=int(counts[0]),Cloud=int(counts[1]),Shadow=int(counts[2]),original_Surface=int(orig[0]),original_Cloud=int(orig[1]),original_Shadow=int(orig[2])))
            scene=[]
            for name in NAMES:
                m=dict(product=product,tile=tile,score=name,**evaluate(truth[valid],scores[name][valid]));metrics.append(m);scene.append(m)
            sun=scene[0]['AP']
            for m in scene[1:]:diffs.append(dict(product=product,tile=tile,comparison='sun-'+m['score'],delta_AP_pp=(sun-m['AP'])*100 if sun is not None and m['AP'] is not None else None))
            plot_overview(dest,raw,np.where(label_valid,truth,3),valid,scores,tile)
            print('SCENE COMPLETE',tile,[(x['score'],x['AP']) for x in scene],flush=True)
        macro=[]
        for name in NAMES[1:]:
            vals=[d['delta_AP_pp'] for d in diffs if d['comparison']=='sun-'+name]
            macro.append(dict(comparison='sun-'+name,mean_delta_AP_pp=float(np.mean(vals)) if all(v is not None for v in vals) else None,positive_scenes=sum(v is not None and v>0 for v in vals),scene_count=3))
        passed=all(x['mean_delta_AP_pp'] is not None and x['mean_delta_AP_pp']>0 and (x['comparison']=='sun-dark' or x['positive_scenes']>=2) for x in macro if x['comparison']!='sun-all')
        for name,table in [('score_metrics.csv',metrics),('coverage.csv',coverage),('paired_differences.csv',diffs),('macro_differences.csv',macro),('direction_validation.csv',directions)]:prev.csvwrite(out/name,table)
        prev.write(out/'summary.json',dict(status='completed',screen_passed=passed,next_action='requires_new_protocol_for_expansion' if passed else 'stop_this_fixed_B11_direction_score_scheme',metrics=metrics,coverage=coverage,macro=macro,new_download_bytes=0,model_or_confirmation_pixels_read=False))
        prev.write(out/'artifact_hashes.json',{str(p.relative_to(out)):prev.sha(p) for p in out.rglob('*') if p.is_file()})
    except Exception as e:
        prev.write(out/'failure.json',dict(type=type(e).__name__,error=str(e)));raise


if __name__=='__main__':main()

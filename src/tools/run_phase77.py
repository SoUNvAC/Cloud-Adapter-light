"""Finite B04 provenance diagnostics. Exactly 12 candidates; zero HTTP calls."""
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

N=1022
CANDIDATES=[dict(name=f'{scope}_{center}',scope=scope,center=center,
                 equivalence='half_pixel_shared' if center=='half_pixel' else f'{scope}_{center}')
            for scope in ['tile','outer','cube'] for center in ['half_pixel','align_corners']]
CANDIDATES += [dict(name='gaussian_sigma05',equivalence='gaussian_sigma05'),
               dict(name='wide_triangle',equivalence='wide_triangle')]
CANDIDATES += [dict(name=f'integer_{r}_{c}',equivalence=f'integer_{r}_{c}',row_phase=r,col_phase=c)
               for r in [0,1] for c in [0,1]]


def coordinates(nin,nout,start,count,center):
    j=start+np.arange(count,dtype='float64')
    if center=='half_pixel':return (j+.5)*nin/nout-.5
    if center=='align_corners':return j*(nin-1)/(nout-1)
    raise ValueError(center)


def gather(raw,rr,cc):
    goodr=(rr>=0)&(rr<raw.shape[0]);goodc=(cc>=0)&(cc<raw.shape[1])
    out=raw[np.clip(rr,0,raw.shape[0]-1)[:,None],np.clip(cc,0,raw.shape[1]-1)]
    return np.where(goodr[:,None]&goodc[None,:],out,np.nan)


def bilinear(raw,rr,cc):
    r=np.floor(rr).astype(int);c=np.floor(cc).astype(int)
    fr=(rr-r)[:,None];fc=(cc-c)[None,:]
    out=np.zeros((len(rr),len(cc)),dtype='float64')
    for dr,wr in [(0,1-fr),(1,fr)]:
        for dc,wc in [(0,1-fc),(1,fc)]:
            w=wr*wc
            # Only genuinely participating pixels contribute invalidity; zero
            # weights at exact integer centers must not propagate neighboring NaN.
            out+=np.where(w>0,gather(raw,r+dr,c+dc)*w,0)
    return out


def separable(raw,rr,cc,offsets,weights):
    out=np.zeros((len(rr),len(cc)),dtype='float64')
    for dr,wr in zip(offsets,weights):
        for dc,wc in zip(offsets,weights):out+=gather(raw,rr+dr,cc+dc)*(wr*wc)
    return out


def gaussian_kernel():
    offsets=np.arange(-2,3);w=np.exp(-.5*(offsets/.5)**2);w/=w.sum()
    # Gaussian first, then half-pixel interpolation: equivalent six-tap kernel
    # evaluated only at requested tile outputs, using original tile support.
    effective=np.zeros(6);effective[:5]+=w*.5;effective[1:]+=w*.5
    return np.arange(-2,4),effective


def candidate(raw,spec,r0,c0,base_r,base_c):
    if 'scope' in spec:
        scope=spec['scope'];center=spec['center']
        if scope=='tile':
            rr=coordinates(raw.shape[0],raw.shape[0]//2,r0,N,center)
            cc=coordinates(raw.shape[1],raw.shape[1]//2,c0,N,center)
        elif scope=='outer':
            rr=2*base_r+coordinates(2304,1152,65,N,center)
            cc=2*base_c+coordinates(2304,1152,65,N,center)
        else:
            rr=2*r0+coordinates(2044,1022,0,N,center)
            cc=2*c0+coordinates(2044,1022,0,N,center)
        return bilinear(raw,rr,cc)
    rr=2*(r0+np.arange(N));cc=2*(c0+np.arange(N))
    if spec['name']=='gaussian_sigma05':
        offsets,weights=gaussian_kernel();return separable(raw,rr,cc,offsets,weights)
    if spec['name']=='wide_triangle':return separable(raw,rr,cc,[-1,0,1,2],np.array([1,3,3,1])/8)
    return gather(raw,rr+spec['row_phase'],cc+spec['col_phase'])


def spatial(cube,ref,name):
    rows=[]
    for region,rs,cs in [('NW',slice(0,N//2),slice(0,N//2)),('NE',slice(0,N//2),slice(N//2,N)),
                         ('SW',slice(N//2,N),slice(0,N//2)),('SE',slice(N//2,N),slice(N//2,N))]:
        rows.append(dict(candidate=name,region=region,**prev.metrics(cube[rs,cs],ref[rs,cs])))
    return rows


def run_scene(row,old,cube,paths,out,baseline_only=False):
    import rasterio
    dest=out/row['product']
    if baseline_only:dest.mkdir()
    else:assert (dest/'baseline_reproduction.json').exists()
    r0=old['selected']['row'];c0=old['selected']['col']
    assert old['B11_mapping_verified'] and old['selected']['candidate']=='inset_65_65_d4_0'
    specials=list(row['radiometry']['special_values'].values());assert float(row['radiometry']['quantification_value'])==10000 and not row['radiometry']['reflectance_offsets']
    arrays={}
    for band in ['B11','B04']:
        with rasterio.open(paths[band]) as ds:prev.verify_grid(ds,row,band);raw=ds.read(1)
        x=raw.astype('float64');x[np.isin(raw,specials)]=np.nan;arrays[band]=x
    b11=arrays.pop('B11')/10000;raw=arrays.pop('B04')
    # Explicit previous path is provided independently; never infer image IDs.
    baseline=reproduce_arrays(cube,b11,raw,r0,c0,old)
    for band,ch,ref in [('B11',11,b11[r0:r0+N,c0:c0+N]),('B04',3,prev.bilinear_window(raw[2*r0-2:2*(r0+N)+2,2*c0-2:2*(c0+N)+2],2*r0-2,2*c0-2,r0,c0)/10000)]:
        saved=np.load(paths['previous_report']/old['product']/(f'{band}_'+('full_residual.npy' if band=='B11' else 'full_grid_full_residual.npy')),allow_pickle=False)
        assert np.array_equal(cube[:,:,ch].astype('float64')-ref,saved,equal_nan=True), 'Prior residual changed'
    if baseline_only:
        prev.write(dest/'baseline_reproduction.json',baseline)
        return baseline
    assert json.loads((dest/'baseline_reproduction.json').read_text())==baseline
    del b11
    refs={};selection=[];selection_points=[]
    for spec in CANDIDATES:
        ref=candidate(raw,spec,r0,c0,r0-65,c0-65)/10000;refs[spec['name']]=ref
        pts,parts=prev.sample(cube[:,:,3],ref,'B04',spec['name'],('selection',))
        selection_points+=pts;selection.append(dict(candidate=spec['name'],equivalence=spec['equivalence'],**parts['selection']))
    # Freeze eligibility before inspecting any new heldout/full residual metrics.
    prev.csvwrite(dest/'selection_points.csv',selection_points);prev.csvwrite(dest/'selection_decisions.csv',selection)
    eligible={r['candidate'] for r in selection if r['pass_gate']}
    prev.write(dest/'selection_freeze.json',dict(eligible=sorted(eligible),created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),heldout_metrics_not_yet_computed=True))
    equivalent=[];reference=refs['tile_half_pixel']
    previous_residual=np.load(paths['previous_report']/old['product']/'B04_full_grid_full_residual.npy',allow_pickle=False)
    assert np.array_equal(cube[:,:,3].astype('float64')-reference,previous_residual,equal_nan=True), 'Half-pixel baseline differs'
    del previous_residual
    for name in ['outer_half_pixel','cube_half_pixel']:
        same=bool(np.array_equal(reference,refs[name],equal_nan=True))
        equivalent.append(dict(left='tile_half_pixel',right=name,mathematical_equivalence_class='half_pixel_shared',exact_values_and_validity_equal=same,max_abs=float(np.nanmax(abs(reference-refs[name])))))
        if not same:raise ValueError('Mathematically equivalent half-pixel implementations differ')
    prev.write(dest/'mathematical_equivalence.json',equivalent)
    points=[];fullrows=[];spatialrows=[];passed=[]
    for spec in CANDIDATES:
        name=spec['name'];ref=refs[name]
        pts,parts=prev.sample(cube[:,:,3],ref,'B04',name);points+=pts
        full=prev.metrics(cube[:,:,3],ref);regions=spatial(cube[:,:,3],ref,name);spatialrows+=regions
        gate=name in eligible and parts['heldout']['pass_gate'] and full['joint_fraction']>=.8 and prev.error_ok(full,'B04')
        fullrows.append(dict(candidate=name,equivalence=spec['equivalence'],selection_eligible=name in eligible,
                             heldout_diagnostic_only=name not in eligible,heldout_pass=parts['heldout']['pass_gate'],all_checks_pass=gate,**full))
        if gate:passed.append(spec)
        np.save(dest/(name+'_residual.npy'),cube[:,:,3].astype('float64')-ref)
    prev.csvwrite(dest/'all_points.csv',points);prev.csvwrite(dest/'full_metrics.csv',fullrows);prev.csvwrite(dest/'spatial_metrics.csv',spatialrows)
    classes=sorted({s['equivalence'] for s in passed})
    status='B04_candidate_consistent' if len(classes)==1 else 'ambiguous' if len(classes)>1 else 'unresolved'
    result=dict(product=row['product'],status=status,passing_candidates=[s['name'] for s in passed],passing_equivalence_classes=classes,
                B11_evidence_retained=True,cube_mapping_status='B11_verified_B04_finite_candidate_consistency' if len(classes)==1 else 'B11_verified_B04_pending',
                unique_author_software_identified=False,all_185_or_other_bands_verified=False,metrics=fullrows)
    prev.write(dest/'status.json',result);return result


def reproduce_arrays(cube,b11,raw,r0,c0,old):
    refs={'B11':b11[r0:r0+N,c0:c0+N],
          'B04':prev.bilinear_window(raw[2*r0-2:2*(r0+N)+2,2*c0-2:2*(c0+N)+2],2*r0-2,2*c0-2,r0,c0)/10000}
    results={}
    for band,ch in [('B11',11),('B04',3)]:
        m=prev.metrics(cube[:,:,ch],refs[band]);expected=old['B11_full'] if band=='B11' else old['B04_full_grid']['full']
        if m!=expected:raise ValueError('Phase76 baseline metrics differ: '+band)
        pts,parts=prev.sample(cube[:,:,ch],refs[band],band,'baseline')
        if band=='B11':assert parts['heldout']==old['B11_heldout'] and parts['heldout']['pass_gate'] and prev.error_ok(m,band)
        else:assert parts==old['B04_full_grid']['partitions']
        results[band]=dict(full=m,partitions=parts,exact_saved_residual_reproduced=True)
    return results


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--previous',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--bindings',type=Path,required=True);a=ap.parse_args()
    import rasterio
    for p in [a.data,a.previous,a.output,a.bindings]:
        assert any(p.resolve().is_relative_to(Path(root)) for root in ['/home/scv/shared','/home/scv/Cloud-Adapter-light'])
    assert len(CANDIDATES)==12
    a.output.mkdir(exist_ok=False);out=a.output
    try:
        expected=json.loads(a.bindings.read_text())
        for f,h in expected.items():assert prev.sha(a.previous/'reports'/f)==h,f
        oldlock=json.loads((a.previous/'reports/mapping_lock.json').read_text());oldsummary=json.loads((a.previous/'reports/summary.json').read_text())
        assert prev.sha(Path(prev.__file__))==oldlock['code_sha256'], 'Phase76 helper changed'
        oldhashes=json.loads((a.previous/'reports/artifact_hashes.json').read_text())
        for f,h in oldhashes.items():assert prev.sha(a.previous/'reports'/f)==h,f
        binding_path=Path('src/research_plans/PHASE76_INPUT_BINDINGS_20261010.json');assert prev.sha(binding_path)==oldlock['bindings_sha256']
        bindings=json.loads(binding_path.read_text());rows=bindings['rows'];assert len(rows)==3
        for f,h in oldlock['inputs'].items():
            p=Path('/home/scv/shared/phase76_control_20261010/cube_mapping_candidates.csv') if f=='cube_mapping_candidates.csv' else a.data/'phase75_20261010'/f
            assert prev.sha(p)==h,f
        verified=json.loads((a.previous/'reports/object_verification.json').read_text())['files'];paths={}
        for obj in verified:
            p=Path(obj['path']);assert p.resolve().is_relative_to(a.previous.resolve()) and prev.sha(p)==obj['sha256'] and p.stat().st_size==obj['size']
            paths.setdefault(obj['product'],{})[obj['band']]=p
        assert len(verified)==6
        offsets,weights=gaussian_kernel()
        lock=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  code_sha256=prev.sha(Path(__file__)),helper_sha256=prev.sha(Path(prev.__file__)),bindings_sha256=prev.sha(a.bindings),previous_inputs=expected,
                  previous_mapping_lock=oldlock,candidates=CANDIDATES,maximum_candidates=12,new_download_bytes=0,centers=prev.CENTERS,patch_width=17,
                  geometry='native20m origin from B11 only; inset65 and d4_0 fixed; no geometry changes',
                  formulas=dict(half_pixel='(j+.5)*N_in/N_out-.5',align_corners='j*(N_in-1)/(N_out-1)',
                    gaussian='sigma=.5 truncate=4 radius2 normalized exp(-.5*(x/.5)^2); separable Gaussian then half-pixel interpolation; algebraically fused 6-tap evaluated at tile target outputs',
                    gaussian_effective_offsets=offsets.tolist(),gaussian_effective_weights=weights.tolist(),triangle_offsets=[-1,0,1,2],triangle_weights=[.125,.375,.375,.125],integer='2*j+row/col_phase, phases in {0,1}'),
                  validity='all strictly positive-weight contributors finite and not XML special; no reflected crop borders; out-of-tile support invalid; cube channel finite/nonzero',
                  support=oldlock['support'],error_gates=oldlock['error_gates']['B04'],full_gate='>=80% joint-valid; same median/P99 bounds; quadrant residuals diagnostic',
                  sequence='first reproduce Phase76 baseline; selection eligibility saved before heldout/full; unqualified heldout/full reported for complete delivery only and cannot regain eligibility',
                  equivalence='three half-pixel scopes share one mathematical class; all other candidates separate regardless of empirical closeness',
                  classification='one passing equivalence class consistent; >1 ambiguous; none unresolved; never smallest RMSE alone',
                  versions=dict(numpy=np.__version__,rasterio=rasterio.__version__,GDAL=rasterio.__gdal_version__),new_candidate_residuals_not_yet_computed=True)
        prev.write(out/'candidate_lock.json',lock);print('LOCK FROZEN, ZERO DOWNLOADS',flush=True)
        results=[]
        with zipfile.ZipFile(a.data/'subscenes.zip') as z:
            # All three original baselines must reproduce before any new candidate.
            for row in rows:
                raw=z.read(row['cube_chain']['cube_member']);assert hashlib.sha256(raw).hexdigest()==oldlock['cube_member_sha256'][row['product']]
                cube=np.load(io.BytesIO(raw),allow_pickle=False);assert cube.shape==(N,N,13) and cube.dtype==np.float32
                old=next(r for r in oldsummary['scenes'] if r['product']==row['product'])
                paths[row['product']]['previous_report']=a.previous/'reports'
                run_scene(row,old,cube,paths[row['product']],out,baseline_only=True)
                print('BASELINE REPRODUCED',row['product'],flush=True)
            print('ALL THREE BASELINES REPRODUCED; NEW CANDIDATES BEGIN',flush=True)
            for row in rows:
                raw=z.read(row['cube_chain']['cube_member']);assert hashlib.sha256(raw).hexdigest()==oldlock['cube_member_sha256'][row['product']]
                cube=np.load(io.BytesIO(raw),allow_pickle=False);assert cube.shape==(N,N,13) and cube.dtype==np.float32
                old=next(r for r in oldsummary['scenes'] if r['product']==row['product'])
                paths[row['product']]['previous_report']=a.previous/'reports'
                result=run_scene(row,old,cube,paths[row['product']],out);results.append(result)
                print('SCENE COMPLETE',row['product'],result['status'],result['passing_candidates'],flush=True)
        prev.write(out/'summary.json',dict(status='completed',scenes=results,new_download_bytes=0,no_labels_models_confirmation_read=True))
        prev.write(out/'artifact_hashes.json',{str(p.relative_to(out)):prev.sha(p) for p in out.rglob('*') if p.is_file()})
    except Exception as e:
        prev.write(out/'failure.json',dict(type=type(e).__name__,error=str(e)));raise


if __name__=='__main__':main()

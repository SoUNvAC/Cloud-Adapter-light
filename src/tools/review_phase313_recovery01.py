"""Supplement old76/77 metadata only; render acquired primary-source pages. Never open JP2."""
import resource
import shutil
import sys
import time
from phase313_common import *
from datetime import datetime,timezone
def main():
    assert sys.platform=='linux';start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    assert not (ROOT/'REVIEW_LOCK.json').exists()
    write(ROOT/'REVIEW_FAILURE01.json',dict(error='Original review stopped before any inventory write: expected phase76 top-level B11_only_transforms.json absent. Original transforms are in reports/summary.json; local B11_only_transforms was editorial derived. Metadata path fix only.',original_code_sha256=sha(R/'src/tools/review_phase313.py'),failed_exec_session=12975,recovery_code_sha256=sha(Path(__file__)),utc=datetime.now(timezone.utc).isoformat()))
    state=load(ROOT/'STATUS.json');assert state['status']=='inventory_complete_pending_editorial'
    inputs=load(ROOT/'INPUT_SHA256.json');ev=read(ROOT/'EVIDENCE_INDEX.csv')
    paths=[BASE/'phase76_20261010/reports'/n for n in ['mapping_lock.json','object_verification.json','summary.json','artifact_hashes.json']]
    paths += [BASE/'phase77_20261011'/n for n in ['summary.json','candidate_lock.json','artifact_hashes.json']]
    paths += sorted((S/'prior_inputs').glob('*'))
    for p in paths:
        assert p.is_file(),str(p);inputs[str(p)]=sha(p);ev.append(dict(path=str(p),kind='old76/77 report/lock metadata only; no original raster or residual read',sha256=sha(p),bytes=p.stat().st_size,read_utc=datetime.now(timezone.utc).isoformat()))
    old76=load(BASE/'phase76_20261010/reports/summary.json');assert all(v['B11_mapping_verified'] and not v['B04_resampling_verified'] for v in old76['scenes'])
    transforms={v['product']:v['B11_verified_mapping'] for v in old76['scenes']}
    old=ROOT/'INPUT_READINESS.csv';assert not (ROOT/'INPUT_READINESS_INITIAL.csv').exists();shutil.copyfile(old,ROOT/'INPUT_READINESS_INITIAL.csv')
    rows=read(old)
    for row in rows:
        v=transforms.get(row['product']);row['B11_mapping_status']='ready' if v else 'missing';row['B11_mapping_evidence']=str(BASE/'phase76_20261010/reports/summary.json') if v else 'NA'
        row['B11_affine_only']=v['affine_first6'] if v else 'NA';row['B11_pixel_centers_only']=v['first_last_centers'] if v else 'NA'
        row['RGB_mapping_status']='contradictory' if v else 'missing'
        row['RGB_mapping_detail']='Phase76 pure bilinear fails; Phase77 0/12 frozen candidates qualify; author preprocessing unresolved; not proof all possible mappings contradictory' if v else 'NA'
        row['geographic_north_to_grid_validation']='missing';row['all_band_cube_mapping_status']='missing'
        if v:
            row['cube_mapping_status']='approximate-only';row['cube_origin']='B11 only: '+json.dumps(v['affine_first6']);row['cube_rowcol_orientation']='B11 only row down / col right; not propagated to RGB';row['cube_pixel_center']='B11 only: '+json.dumps(v['first_last_centers'])
            row['mapping_evidence']='Phase76 B11 independent49points/fullframe verified; Phase77 B04 unresolved; Phase309 alone cannot prove65inset'
            row['ready_status']='approximate-only';row['reason']='B11 coordinate evidence reusable for this exact product only; RGB preprocessing/north conversion/height/context remain unresolved; geometry experiment not executable'
    table(old,rows);accept=load(ROOT/'INVENTORY_ACCEPTANCE.json');accept.update(B11_verified=sum(r['B11_mapping_status']=='ready' for r in rows),approximate_only=sum(r['ready_status']=='approximate-only' for r in rows),missing=sum(r['ready_status']=='missing' for r in rows),RGB_tested_chains_contradictory=3,all_band_cube_verified=0,initial_inventory_preserved='INPUT_READINESS_INITIAL.csv',supplement='old76/77 metadata discovered under phase76_20261010, not incorrect _20261011 path; no pixel rerun')
    write(ROOT/'INVENTORY_ACCEPTANCE.json',accept)
    sys.path.insert(0,'/home/scv/shared/phase309_20261025/toolenv');import pypdfium2 as pdfium
    render=[]
    for filename,pages in [('ESA_ATBD_0.pdf',[1,39,40]),('FMASK2015_0.pdf',[6])]:
        pdf=pdfium.PdfDocument(ROOT/'literature'/filename)
        for number in pages:
            assert shutil.disk_usage(ROOT).free>=4000000000
            dest=ROOT/'literature'/f'{Path(filename).stem}_page{number:03}.png';assert not dest.exists()
            page=pdf[number-1];bitmap=page.render(scale=1.3);image=bitmap.to_pil();image.save(dest);image.close();bitmap.close();page.close();render.append(dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest),source_pdf=filename,page=number))
        pdf.close()
    write(ROOT/'INPUT_SHA256.json',inputs);table(ROOT/'EVIDENCE_INDEX.csv',ev)
    elapsed=time.monotonic()-start;state['metadata_and_review_seconds']=state['wall_seconds']+elapsed;assert state['metadata_and_review_seconds']<3600
    state['peak_cpu_bytes']=max(state['peak_cpu_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);assert state['peak_cpu_bytes']<8*1024**3
    write(ROOT/'STATUS.json',state);write(ROOT/'REVIEW_LOCK.json',dict(code_sha256=sha(Path(__file__)),seconds=elapsed,renders=render,no_JP2_or_residual_arrays_opened=True,no_new_downloads=True,old76_path_correction=True))
    for name in ['ESA_ATBD_page001.txt','ESA_ATBD_page039.txt','ESA_ATBD_page040.txt','FMASK2015_page006.txt']:
        print('\nFILE',name);print((ROOT/'literature'/name).read_text())
    print(json.dumps(accept))
if __name__=='__main__':main()

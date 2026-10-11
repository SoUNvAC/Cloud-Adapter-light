"""Phase88 fixed expert blind-review packet from existing local ALCD imagery.

No model execution/download; never fill human judgments. Subcommands enforce
sample freeze before extraction and complete human A lock before reference reveal.
"""
import argparse
import csv
import hashlib
import html
import io
import json
import struct
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/phase88/phase88_20261011'
PREV = ROOT / 'outputs/phase87/phase87_20261011'
OLD = ROOT / 'outputs/phase65/alcd_visual_review_v1'
DATA = ROOT / 'data/sentinel2_alcd_1460961'
BANDS = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']


def jp2_precision(path):
    """Read unsigned single-component codestream precision, not a data heuristic.

Pillow expands sub-16-bit JPEG2000 integer samples to the 16-bit display range;
GDAL preserves codestream DN. Undo that exact bit shift before any stretch/mask.
"""
    with path.open('rb') as f:
        while True:
            header=f.read(8); assert len(header)==8, 'Missing JP2 codestream'
            length,kind=struct.unpack('>I4s',header); header_size=8
            if length==1: length=struct.unpack('>Q',f.read(8))[0];header_size=16
            if kind==b'jp2c':
                siz=f.read(43);assert siz[:4]==b'\xff\x4f\xff\x51'
                # SIZ offsets include two-byte preceding SOC.
                assert struct.unpack('>H',siz[40:42])[0]==1
                ssiz=siz[42];assert not ssiz&128
                return (ssiz&127)+1
            assert length>=header_size
            f.seek(length-header_size,1)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p, obj):
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def csvwrite(p, rows):
    with p.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def freeze():
    assert not OUT.exists(), 'Preserve existing Phase88 delivery'
    hashes = read(PREV / 'DELIVERY_SHA256.json')
    for n in ['pending_adjudication.csv', 'sample_alias_lock.json', 'blind_initial.csv']:
        assert sha(PREV/n) == hashes[n]
    with (PREV/'pending_adjudication.csv').open(encoding='utf-8-sig', newline='') as f:
        prior = list(csv.DictReader(f))
    lock = read(PREV/'sample_alias_lock.json')
    p1 = [r for r in prior if r['priority'] == 'P1']; assert len(p1) == 30
    chosen = {r['source_id']: 'P1' for r in p1}; short = []
    for alias in lock['aliases']:
        eligible = [r for r in prior if r['product'] == alias['representative'] and
                    r['priority'] != 'P1' and 'uniform' in r['sources'].split(';')]
        # Do not inspect reference class, AI judgment, confidence or model output.
        eligible.sort(key=lambda r: hashlib.sha256(f"PHASE88|{r['product']}|{r['source_id']}".encode()).digest())
        for r in eligible[:3]:
            assert r['source_id'] not in chosen
            chosen[r['source_id']] = 'regular_control'
        if len(eligible) < 3: short.append(dict(product=alias['representative'], available=len(eligible)))
    samples = [dict(s, phase88_stratum=chosen[s['source_id']]) for s in lock['samples'] if s['source_id'] in chosen]
    samples.sort(key=lambda s: hashlib.sha256(f"PHASE88_DISPLAY|{s['product']}|{s['source_id']}".encode()).digest())
    for i, s in enumerate(samples, 1): s['review_id'] = f'E{i:03d}'
    assert len(samples) <= 57 and len({s['source_id'] for s in samples}) == len(samples)
    OUT.mkdir(parents=True)
    write(OUT/'sample_input_lock.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
          protocol_sha256=sha(ROOT/'src/research_plans/PHASE88_EXPERT_LABEL_REVIEW_20261011.md'),
          phase87_delivery_sha256=sha(PREV/'DELIVERY_SHA256.json'),
          phase86_candidate_sha256=sha(ROOT/'outputs/phase86/phase86_20261011/candidate_lock.json'),
          imagery_manifest_sha256=sha(DATA/'imagery/product_manifest.json'),
          local_imagery_status_sha256=sha(DATA/'imagery/imagery_download_status.json'),
          geometry_audit_sha256=sha(ROOT/'outputs/phase65/alcd_imagery/remote_geometry_audit.json'),
          aliases=lock['aliases'], samples=samples, short_controls=short,
          total=len(samples), strata=dict(Counter(s['phase88_stratum'] for s in samples)),
          human_review_status='not_executed', confirmation_ready_groups=0))
    write(OUT/'SAMPLING_FROZEN.json', dict(sample_input_lock_sha256=sha(OUT/'sample_input_lock.json'),
          frozen_before_extraction_and_human_review=True, created_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(total=len(samples), strata=dict(Counter(s['phase88_stratum'] for s in samples)))))


def build():
    import numpy as np
    from PIL import Image, ImageDraw
    # Failed preflight may leave empty directories; never replace any prior artifact.
    assert not any(p.is_file() for d in ['images','windows'] for p in (OUT/d).rglob('*')), 'Preserve prior extraction'
    lock = read(OUT/'sample_input_lock.json')
    assert sha(OUT/'sample_input_lock.json') == read(OUT/'SAMPLING_FROZEN.json')['sample_input_lock_sha256']
    manifest = read(DATA/'imagery/product_manifest.json')
    status = read(DATA/'imagery/imagery_download_status.json')
    geompath = ROOT/'outputs/phase65/alcd_imagery/remote_geometry_audit.json'
    assert sha(geompath) == lock['geometry_audit_sha256']
    assert sha(DATA/'imagery/product_manifest.json') == lock['imagery_manifest_sha256']
    assert sha(DATA/'imagery/imagery_download_status.json') == lock['local_imagery_status_sha256']
    geometry = {s['key']: s for s in read(geompath)['scenes']}
    scenes = {s['key']: s for s in manifest['scenes']}
    archive = DATA/'SENTINEL_2_reference_cloud_masks_Baetens_Hagolle.tgz'
    assert sha(archive) == manifest['archive_sha256']
    (OUT/'images').mkdir(exist_ok=True); (OUT/'windows').mkdir(exist_ok=True)
    support = []; missing = []; source_hashes = {}
    with tarfile.open(archive, 'r:gz') as tar:
        for alias in lock['aliases']:
            group = [s for s in lock['samples'] if s['product'] == alias['representative']]
            scene = scenes[alias['alias']]; geom = geometry[alias['alias']]
            assert scene['gcs_product_prefix'].rstrip('/').split('/')[-1] == alias['representative']+'.SAFE'
            assert geom['alignment_status'] == 'all_six_grids_match'
            assert geom['shape'] == [1830,1830] and geom['transform'][0] == 60 and geom['transform'][4] == -60
            raw = tar.extractfile(scene['label_members'][0]).read()
            assert hashlib.sha256(raw).hexdigest() == geom['label_sha256']
            # Only label validity enters displayed mask; no reference class shown.
            label_valid = np.isin(np.asarray(Image.open(io.BytesIO(raw))), [2,3,4,5,6,7])
            windows = {}
            for s in group:
                y,x=s['row'],s['col']; box=[max(0,y-25),min(1830,y+26),max(0,x-25),min(1830,x+26)]
                y0,y1,x0,x1=box
                windows[s['review_id']] = dict(box=box, arrays={}, valid=label_valid[y0:y1,x0:x1].copy())
                assert sha(OLD/'windows'/(s['source_id']+'.npz')) == s['window_sha256']
            for band in BANDS:
                obj = [o for o in scene['objects'] if o['name'].endswith('_'+band+'.jp2')]
                assert len(obj) == 1
                entry = status['files'][obj[0]['name']]
                assert entry['status'] == 'verified' and entry['generation'] == obj[0]['generation']
                path = DATA/'imagery'/Path(entry['path'].replace(chr(92), '/'))
                if not path.exists():
                    missing.append(dict(product=alias['representative'],band=band,reason='exact_original_not_present'))
                    continue
                assert path.stat().st_size == entry['size']
                digest=sha(path)
                assert digest == entry['sha256'] == geom['bands'][band]['sha256']
                assert all(digest == s['input_band_sha256'][band] for s in group)
                precision=jp2_precision(path);assert 9<=precision<=16
                source_hashes[obj[0]['name']] = dict(sha256=digest,size=entry['size'],generation=entry['generation'],codestream_precision=precision,pillow_to_native_right_shift=16-precision)
                with Image.open(path) as im:
                    assert [im.height,im.width] == geom['bands'][band]['shape']
                    factor = int(60/geom['bands'][band]['resolution'][0]); assert factor in (3,6)
                    for s in group:
                        w=windows[s['review_id']];y0,y1,x0,x1=w['box']
                        arr=np.asarray(im.crop((x0*factor,y0*factor,x1*factor,y1*factor))).copy()
                        assert im.mode=='I;16' and arr.dtype==np.uint16
                        if precision<16:
                            shift=16-precision
                            assert not np.any(arr & ((1<<shift)-1)), 'Non-reversible Pillow expansion'
                            arr >>= shift
                        good=np.ones(arr.shape,dtype=bool)
                        for v in {0,*geom['special_values'].values()}: good &= arr != v
                        w['valid'] &= good.reshape(y1-y0,factor,x1-x0,factor).all(axis=(1,3))
                        w['arrays'][band]=arr
                        # Exact existing small-window pixels must agree with the new extraction.
                        sy0,sy1,sx0,sx1=s['window']
                        with np.load(OLD/'windows'/(s['source_id']+'.npz'),allow_pickle=False) as prior:
                            overlap=arr[(sy0-y0)*factor:(sy1-y0)*factor,(sx0-x0)*factor:(sx1-x0)*factor]
                            assert np.array_equal(overlap,prior[band]), (s['review_id'],band)
                print(alias['tile']+' '+band+' verified/decoded',flush=True)
            for s in group:
                w=windows[s['review_id']]
                if set(w['arrays']) != set(BANDS):
                    missing.append(dict(review_id=s['review_id'],reason='missing_exact_band'));continue
                y0,y1,x0,x1=w['box'];sy0,sy1,sx0,sx1=s['window']
                with np.load(OLD/'windows'/(s['source_id']+'.npz'),allow_pickle=False) as prior:
                    assert np.array_equal(w['valid'][sy0-y0:sy1-y0,sx0-x0:sx1-x0],prior['joint_valid'])
                assert w['valid'][s['row']-y0,s['col']-x0]
                np.savez_compressed(OUT/'windows'/(s['review_id']+'.npz'),**w['arrays'],joint_valid_60m=w['valid'])
                # Keep original small-window stretch values also for larger context.
                panel=Image.new('RGB',(2040,1080),'white');draw=ImageDraw.Draw(panel)
                for i,bands in enumerate([['B04','B03','B02'],['B08','B11','B12']]):
                    channels=[]
                    for band in bands:
                        lo,hi=s['stretch'][band];ar=w['arrays'][band]
                        channel=np.uint8(np.clip((ar.astype(float)-lo)/max(hi-lo,1),0,1)*255)
                        channels.append(np.asarray(Image.fromarray(channel).resize(((x1-x0)*6,(y1-y0)*6),Image.Resampling.NEAREST)))
                    image=Image.fromarray(np.stack(channels,axis=-1)).resize((1020,1020),Image.Resampling.NEAREST)
                    invalid=Image.fromarray(np.uint8(~w['valid'])*255).resize((1020,1020),Image.Resampling.NEAREST)
                    image.paste((96,96,96),(0,0,1020,1020),invalid)
                    d=ImageDraw.Draw(image);cx=(s['col']-x0+.5)/(x1-x0)*1020;cy=(s['row']-y0+.5)/(y1-y0)*1020
                    d.rectangle((cx-510/(x1-x0),cy-510/(y1-y0),cx+510/(x1-x0),cy+510/(y1-y0)),outline='red',width=3)
                    d.line((30,990,30+600/((x1-x0)*60)*1020,990),fill='white',width=5)
                    d.text((30,967),'600m',fill='white')
                    panel.paste(image,(i*1020,60))
                draw.text((12,10),s['review_id']+' | RGB / B08 B11 B12 | red box = 60m | gray = invalid',fill='black')
                draw.text((12,32),f"True support {60*(x1-x0)} x {60*(y1-y0)} m; clipped at tile edges; fixed original stretch",fill='black')
                panel.save(OUT/'images'/(s['review_id']+'_large.png'))
                smallpath=PREV/'blind_panels'/(s['anonymous_id']+'.png')
                assert sha(smallpath)==s['blind_panel_sha256']
                small=Image.new('RGB',(880,470),'white');small.paste(Image.open(smallpath).crop((0,30,880,470)),(0,30))
                ImageDraw.Draw(small).text((10,8),s['review_id']+' | Small context | RGB / B08 B11 B12 | red box = 60m',fill='black')
                small.save(OUT/'images'/(s['review_id']+'_small.png'))
                a,b,c,d,e,f=geom['transform']; x=c+(s['col']+.5)*a; y=f+(s['row']+.5)*e
                support.append(dict(review_id=s['review_id'],product=s['product'],crs=geom['crs'],
                    row=s['row'],col=s['col'],center_xy=[x,y],window_60m=w['box'],
                    bounds=[c+x0*a,f+y1*e,c+x1*a,f+y0*e],
                    center_offset_60m=[s['row']-y0,s['col']-x0],shape_60m=list(w['valid'].shape),
                    stretch=s['stretch'],mask_policy='valid original-label support AND all six bands exclude XML special values at every contributing native pixel; small-mask exact equality checked',
                    original_small_native_pixels_match=True,window_sha256=sha(OUT/'windows'/(s['review_id']+'.npz')),
                    small_sha256=sha(OUT/'images'/(s['review_id']+'_small.png')),large_sha256=sha(OUT/'images'/(s['review_id']+'_large.png'))))
    write(OUT/'extraction_audit.json',dict(sampling_lock_sha256=sha(OUT/'sample_input_lock.json'),
          support=support,missing=missing,source_objects=source_hashes,original_archive_sha256=manifest['archive_sha256'],
          method='Pillow OpenJPEG native pixels; verified existing CRS/grid audit; no resampling of reference labels',
          human_review_status='not_executed',confirmation_ready_groups=0))
    make_page([s['review_id'] for s in lock['samples']],{s['review_id'] for s in support})
    fields=dict(review_id='',judgment='',reason='',small_sufficient='',large_sufficient='',needs_clear_date='',needs_DEM='',reviewer_id='',experience='',review_date='')
    csvwrite(OUT/'expert_A_template.csv',[dict(fields,review_id=s['review_id']) for s in lock['samples']])
    csvwrite(OUT/'expert_B_template.csv',[dict(review_id=s['review_id'],original_assessment='',revised_assessment='',reason='',evidence_needed='',reviewer_id='',review_date='') for s in lock['samples']])
    csvwrite(OUT/'evidence_needs.csv',[dict(review_id=s['review_id'],larger_context='prepared' if s['review_id'] in {x['review_id'] for x in support} else 'missing',clear_date='not_downloaded_need_to_be_judged',DEM='not_downloaded_need_to_be_judged',human_status='not_executed') for s in lock['samples']])
    print(json.dumps(dict(prepared=len(support),missing=len(missing),human_reviewed=0)))


def make_page(ids, available):
    template=ROOT/'src/tools/phase88_blind_page.html'
    text=template.read_text(encoding='utf-8').replace('__IDS__',json.dumps(ids)).replace('__AVAILABLE__',json.dumps(sorted(available))).replace('__LOCK_SHA__',sha(OUT/'sample_input_lock.json'))
    with (OUT/'index.html').open('x',encoding='utf-8') as f:f.write(text)


def render_metric():
    """Preserve the first layout; final contexts retain metric aspect at tile edges."""
    import numpy as np
    from PIL import Image, ImageDraw
    audit=read(OUT/'extraction_audit.json');records=[]
    for s in audit['support']:
        y0,y1,x0,x1=s['window_60m'];width=(x1-x0)*20;height=(y1-y0)*20
        path=OUT/'images'/(s['review_id']+'_context.png');assert not path.exists()
        panel=Image.new('RGB',(width*2,height+60),'white');draw=ImageDraw.Draw(panel)
        with np.load(OUT/'windows'/(s['review_id']+'.npz'),allow_pickle=False) as z:
            assert 'labels' not in z.files
            for i,bands in enumerate([['B04','B03','B02'],['B08','B11','B12']]):
                channels=[]
                for band in bands:
                    lo,hi=s['stretch'][band]
                    ar=np.uint8(np.clip((z[band].astype(float)-lo)/max(hi-lo,1),0,1)*255)
                    channels.append(np.asarray(Image.fromarray(ar).resize((width,height),Image.Resampling.NEAREST)))
                image=Image.fromarray(np.stack(channels,axis=-1));invalid=Image.fromarray(np.uint8(~z['joint_valid_60m'])*255).resize((width,height),Image.Resampling.NEAREST)
                image.paste((96,96,96),(0,0,width,height),invalid)
                d=ImageDraw.Draw(image);cy,cx=s['center_offset_60m'];cx=(cx+.5)*20;cy=(cy+.5)*20
                d.rectangle((cx-10,cy-10,cx+10,cy+10),outline='red',width=3)
                d.line((30,height-30,230,height-30),fill='white',width=5);d.text((30,height-53),'600m',fill='white')
                panel.paste(image,(i*width,60))
        draw.text((10,8),s['review_id']+' | RGB / B08 B11 B12 | red box = 60m; gray = invalid',fill='black')
        draw.text((10,30),f'{60*(x1-x0)} x {60*(y1-y0)} m true support; 20 display pixels per 60m cell; no edge padding',fill='black')
        panel.save(path)
        records.append(dict(review_id=s['review_id'],path='images/'+path.name,sha256=sha(path),display_size=list(panel.size),metres_per_display_pixel=3))
    write(OUT/'metric_display_audit.json',dict(extraction_audit_sha256=sha(OUT/'extraction_audit.json'),contexts=records,
        note='Final large-context display preserves metric aspect, including nine clipped windows. Initial square preview layout retained separately; never used as final expert view.'))
    prior=OUT/'layout_v1.html';assert not prior.exists()
    (OUT/'index.html').rename(prior)
    template=(ROOT/'src/tools/phase88_blind_page.html').read_text(encoding='utf-8')
    text=template.replace('__IDS__',json.dumps([s['review_id'] for s in read(OUT/'sample_input_lock.json')['samples']])).replace('__AVAILABLE__',json.dumps([r['review_id'] for r in records])).replace('__LOCK_SHA__',sha(OUT/'sample_input_lock.json'))
    text=text.replace("id+'_'+k+'.png'", "id+(k==='large'?'_context.png':'_small.png')")
    with (OUT/'index.html').open('x',encoding='utf-8') as f:f.write(text)
    print('Final metric-aspect context images and anonymous index prepared:',len(records))


def lock_human(path):
    assert path.resolve().is_relative_to(ROOT), 'Place returned human file inside authorized repository'
    packet=read(path);lock=read(OUT/'sample_input_lock.json')
    assert packet['stage']=='A' and packet.get('complete') is True and packet['reviewer_id'].strip() and packet['experience'].strip()
    assert packet['reviewer_type']=='human', 'AI cannot be registered as human'
    assert not any(v in packet['reviewer_id'].lower() for v in ['chatgpt','codex','gpt-','ai assistant'])
    assert packet['sampling_lock_sha256']==sha(OUT/'sample_input_lock.json')
    rows=packet['rows'];ids={s['review_id'] for s in lock['samples']}
    assert len(rows)==len(ids) and {r['review_id'] for r in rows}==ids
    for r in rows:
        assert r['judgment'] in ['Cloud','Shadow','Surface','混合','无法判定'] and r['reason'].strip()
        assert all(r[k] in ['yes','no'] for k in ['small_sufficient','large_sufficient','needs_clear_date','needs_DEM'])
        assert r['review_date']
    assert not (OUT/'expert_A_original.json').exists()
    with (OUT/'expert_A_original.json').open('xb') as f:f.write(path.read_bytes())
    csvwrite(OUT/'expert_A_locked.csv',rows)
    write(OUT/'EXPERT_A_LOCK.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),
          original_json_sha256=sha(OUT/'expert_A_original.json'),csv_sha256=sha(OUT/'expert_A_locked.csv'),
          reviewer_id=packet['reviewer_id'],experience=packet['experience'],reviewed=len(rows),
          identity_is_self_reported_not_independently_certified=True,confirmation_ready_groups=0))
    # Only now provide the separate B page. Never embed references in the A page.
    old={s['id']:s for s in read(OLD/'sample_manifest.json')['samples']}
    content=['<!doctype html><meta charset="utf-8"><title>Phase88 B</title><h1>B阶段：A判断已锁定</h1><p>逐点填写expert_B_template.csv；保留A判断。参考图左右为原版／修订版，白云、蓝云影、绿地表，灰无效；缺失参考不是Surface。原类号2/3云、4云影、5/6/7地表；修订64云影、128地表、255云。多类中心保留混合，不多数投票。参考不是真值。</p><p>原版与修订版分别填：未发现明确冲突／有证据支持可能冲突／仍无法裁决，并写理由和证据需求；修订缺失时注明缺失。不能仅因与参考一致就认定云影。</p>']
    expected=read(OLD/'SHA256.json');expected={k.replace(chr(92),'/'):v for k,v in expected.items()}
    for s in lock['samples']:
        p=OLD/'images'/(s['source_id']+'_label.png');assert sha(p)==expected[p.relative_to(OLD).as_posix()]
        r=old[s['source_id']]
        content.append(f'<h2>{s["review_id"]}</h2><p>原类号：{r["original_class"]}；修订中心类号：{html.escape(str(r["revised_center"]))}</p><img style="max-width:100%" src="../../phase65/alcd_visual_review_v1/images/{s["source_id"]}_label.png">')
    with (OUT/'stage_B.html').open('x',encoding='utf-8') as f:f.write('\n'.join(content))
    print('Human A locked; B page created. No adjudications filled automatically.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','build','render-metric','lock-human']);p.add_argument('--file',type=Path)
    args=p.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='build':build()
    elif args.action=='render-metric':render_metric()
    else:
        assert args.file is not None;lock_human(args.file)

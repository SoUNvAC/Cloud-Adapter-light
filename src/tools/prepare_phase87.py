"""Freeze ALCD Phase87 samples and image-only inputs, without model execution."""
import csv, hashlib, json, re, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageDraw
from audit_phase86 import canonical, original_match

REPO = Path(__file__).resolve().parents[2]
BASE = REPO/'outputs/phase65/alcd_visual_review_v1'
OUT = REPO/'outputs/phase87/phase87_20261011'
DIRECTIONS = [a+'_to_'+b for a in ['Shadow','Cloud','Surface'] for b in ['Shadow','Cloud','Surface'] if a != b]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, obj):
    with p.open('x', encoding='utf-8') as f: json.dump(obj, f, ensure_ascii=False, indent=2)
def main():
    lockpath=REPO/'outputs/phase86/phase86_20261011/candidate_lock.json'
    grouppath=lockpath.with_name('groups.csv')
    lock=json.loads(lockpath.read_text()); groups=lock['groups']
    with grouppath.open(encoding='utf-8-sig',newline='') as f: csvgroups=list(csv.DictReader(f))
    assert len(groups)==len(csvgroups)==9
    assert [g['representative'] for g in groups]==[g['representative'] for g in csvgroups]
    manifestpath=REPO/'data/sentinel2_alcd_1460961/imagery/product_manifest.json'
    manifest=json.loads(manifestpath.read_text())
    samplespath=BASE/'sample_manifest.json'; prior=json.loads(samplespath.read_text())
    expected={k.replace(chr(92),'/'):v for k,v in json.loads((BASE/'SHA256.json').read_text()).items()}
    assert sha(samplespath)==expected['sample_manifest.json']
    mapping=[]; chosen={}; missing=[]
    for g in groups:
        product=g['representative']; ident=canonical(product)
        matches=[s for s in manifest['scenes'] if original_match(s['parameters']['cloudy_product_name'],s['parameters']['tile'],['tiles/'+product+'.SAFE/'])]
        assert len(matches)==1, (product,len(matches))
        scene=matches[0]; assert scene['gcs_product_prefix'].rstrip('/').split('/')[-1]==product+'.SAFE'
        rows=[s for s in prior['samples'] if s['product']==scene['key']]
        assert rows
        uniform=[s for s in rows if 'uniform' in s['groups']]; assert len(uniform)<=25
        selected={s['id']:['uniform'] for s in uniform}
        for direction in DIRECTIONS:
            options=[s for s in rows if direction in s['groups']]
            if not options: missing.append(dict(product=product,direction=direction,reason='no_existing_direction_point')); continue
            s=min(options,key=lambda s:hashlib.sha256(f"PHASE87|{product}|{s['id']}".encode()).digest())
            oc={2:'Cloud',3:'Cloud',4:'Shadow',5:'Surface',6:'Surface',7:'Surface'}[s['original_class']]
            rc=s['revised_center']; assert len(rc)==1
            assert direction==oc+'_to_'+{64:'Shadow',128:'Surface',255:'Cloud'}[rc[0]]
            selected.setdefault(s['id'],[]).append(direction)
        mapping.append(dict(group_id=g['group_id'],representative=product,alias=scene['key'],tile=ident['tile'],uniform_count=len(uniform),missing_uniform_blocks=25-len(uniform),available_directions=[d for d in DIRECTIONS if any(d in s['groups'] for s in rows)]))
        for sid,sources in selected.items():
            s=next(s for s in rows if s['id']==sid)
            chosen[sid]=dict(group_id=g['group_id'],product=product,alias=scene['key'],source_id=sid,sources=sources,row=s['row'],col=s['col'],window=s['window'],stretch=s['stretch'],input_band_sha256=s['input_sha256'])
    assert len(chosen)<=279
    # Anonymous display order is independent of reference classes and strata.
    ordered=sorted(chosen.values(),key=lambda s:hashlib.sha256(('PHASE87_DISPLAY|'+s['product']+'|'+s['source_id']).encode()).digest())
    assert not OUT.exists() or not any(p.is_file() for p in OUT.rglob('*')), 'Preserve existing outputs'
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'blind_panels').mkdir(exist_ok=True); (OUT/'blind_sheets').mkdir(exist_ok=True)
    for i,s in enumerate(ordered,1):
        aid=f'A{i:03d}'; s['anonymous_id']=aid
        window=BASE/'windows'/f"{s['source_id']}.npz"; panel=BASE/'images'/f"{s['source_id']}.png"
        for p in [window,panel]: assert sha(p)==expected[p.relative_to(BASE).as_posix()]
        s['window_sha256']=sha(window); s['original_blind_panel_sha256']=sha(panel)
        # Only joint_valid is loaded; labels and reference versions never enter stage A.
        with np.load(window,allow_pickle=False) as z: valid=z['joint_valid']
        assert valid.dtype==np.bool_ and valid.any()
        im=Image.open(panel).convert('RGB'); assert im.size==(880,440)
        invalid=Image.fromarray(np.uint8(~valid)*255).resize((440,440),Image.Resampling.NEAREST)
        for x in [0,440]: im.paste((96,96,96),(x,0,x+440,440),invalid)
        card=Image.new('RGB',(880,470),'white'); card.paste(im,(0,30))
        ImageDraw.Draw(card).text((10,8),aid+' | RGB / B08 B11 B12 | red box = 60m; gray = invalid',fill='black')
        card.save(OUT/'blind_panels'/f'{aid}.png')
        s['blind_panel_sha256']=sha(OUT/'blind_panels'/f'{aid}.png')
    sheets=[]
    for start in range(0,len(ordered),12):
        batch=ordered[start:start+12]; sheet=Image.new('RGB',(1760,470*6),(225,225,225))
        for j,s in enumerate(batch): sheet.paste(Image.open(OUT/'blind_panels'/f"{s['anonymous_id']}.png"),((j%2)*880,(j//2)*470))
        name=f'{start//12+1:02d}.png'; sheet.save(OUT/'blind_sheets'/name)
        sheets.append(dict(path='blind_sheets/'+name,ids=[s['anonymous_id'] for s in batch],sha256=sha(OUT/'blind_sheets'/name)))
    frozen=dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(REPO/'src/research_plans/PHASE87_ALCD_BLIND_LABEL_TRIAGE_20261011.md'),candidate_lock_sha256=sha(lockpath),groups_sha256=sha(grouppath),product_manifest_sha256=sha(manifestpath),old_sample_manifest_sha256=sha(samplespath),aliases=mapping,samples=ordered,missing_directions=missing,confirmation_ready_groups=0)
    write(OUT/'sample_alias_lock.json',frozen)
    # Stage A manifest deliberately contains no aliases, labels, groups or directions.
    write(OUT/'blind_input_lock.json',dict(sample_lock_sha256=sha(OUT/'sample_alias_lock.json'),panels=[dict(id=s['anonymous_id'],path='blind_panels/'+s['anonymous_id']+'.png',sha256=s['blind_panel_sha256']) for s in ordered],sheets=sheets,policy='image_only_RGB_NIR_SWIR_joint_valid_fixed_stretch_no_labels_no_models'))
    write(OUT/'SAMPLING_FROZEN.json',dict(sample_alias_lock_sha256=sha(OUT/'sample_alias_lock.json'),blind_input_lock_sha256=sha(OUT/'blind_input_lock.json'),frozen_before_judgment=True,total=len(ordered),uniform=sum('uniform' in s['sources'] for s in ordered),difference=sum(any(t!='uniform' for t in s['sources']) for s in ordered),groups=9))
    print(json.dumps(dict(total=len(ordered),sheets=len(sheets),uniform=sum('uniform' in s['sources'] for s in ordered),difference=sum(any(t!='uniform' for t in s['sources']) for s in ordered),missing_directions=len(missing))))
if __name__=='__main__': main()



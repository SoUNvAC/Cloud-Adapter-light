"""Fixed post-confirmation failure diagnosis. No network inference or fitting."""
import csv,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from prepare_phase65a_explore import digest
from phase65d_recovery_math import logit_score,predict_logit
from phase65d_metrics import confusion,corrected

REPO=Path('/home/scv/Cloud-Adapter-light')
DATA=Path('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871')
OLD=DATA/'phase72_20261010'
ROOT=DATA/'phase73_20261010'
CATS=['new_FP','lost_TP','rescued_TP','stable_TP']
COLORS=np.array([[0,158,115],[240,228,66],[0,114,178],[150,150,150]],dtype=np.uint8)

def read(p):return json.loads(p.read_text())
def write(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def table(p,rows,fields=None):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def sample(mask,product,category):
    candidates=np.flatnonzero(mask.ravel());seed=73010+int(hashlib.sha256(product.encode()).hexdigest()[:8],16)+category
    idx=np.random.default_rng(seed).choice(len(candidates),4,replace=False) if len(candidates)>=4 else np.arange(len(candidates))
    return candidates[idx],dict(seed=seed,candidates=len(candidates),candidate_sha256=hashlib.sha256(candidates.tobytes()).hexdigest(),
        indices_sha256=hashlib.sha256(idx.tobytes()).hexdigest(),selected_indices=idx.tolist())
def choose(rows):
    fpr=sorted(rows,key=lambda r:(-r['delta_fpr'],r['product']))[:3]
    ap=[r for r in rows if r['delta_ap_pp'] is not None]
    worst=sorted(ap,key=lambda r:(r['delta_ap_pp'],r['product']))[:3]
    best=sorted(ap,key=lambda r:(-r['delta_ap_pp'],r['product']))[:3]
    picks={}
    for reason,subset in [('worst_fpr',fpr),('worst_ap',worst),('best_ap',best),('T06CWV',[r for r in rows if '_T06CWV_' in r['product']])]:
        for r in subset:picks.setdefault(r['product'],dict(row=r,reasons=[]))['reasons'].append(reason)
    assert len(picks)<=10
    return picks
def crop(a,y,x,fill):
    out=np.full((257,257)+a.shape[2:],fill,dtype=a.dtype);y0=max(0,y-128);x0=max(0,x-128);y1=min(a.shape[0],y+129);x1=min(a.shape[1],x+129)
    out[y0-y+128:y1-y+128,x0-x+128:x1-x+128]=a[y0:y1,x0:x1];return out
def marked(a,title):
    im=Image.new('RGB',(257,281),(220,220,220));im.paste(Image.fromarray(a),(0,24));d=ImageDraw.Draw(im)
    d.text((3,5),title,fill=(0,0,0));d.rectangle((98,122,158,182),outline=(255,0,255),width=1)
    d.line((123,152,133,152),fill=(255,0,255),width=1);d.line((128,147,128,157),fill=(255,0,255),width=1);return im
def pack(tiles,cols):
    result=Image.new('RGB',(257*cols,281*((len(tiles)+cols-1)//cols)),(200,200,200))
    for i,t in enumerate(tiles):result.paste(t,((i%cols)*257,(i//cols)*281))
    return result

def main():
    assert not ROOT.exists(),'Preserve previous Phase73 output'
    assert digest(OLD/'delivery/phase72_report.json')=='0536646278acc4e2eaff43d7181371ff9a7b1a02367ebb03ecbe329ba0d00372'
    assert digest(OLD/'delivery/execution_lock.json')=='e4d43687bf85db0d1fb1b390f8142b11bff53999e0dd97a1c1a944e3e298fed3'
    report=read(OLD/'delivery/phase72_report.json');lock=read(OLD/'delivery/confirmation_lock.json')
    assert digest(OLD/'delivery/confirmation_lock.json')==report['confirmation_lock_sha256']
    assert digest(REPO/'src/work_dirs/phase65a/split_lock.json')==lock['original_split_sha256']
    status=read(DATA/'download_status.json');assert status['status']=='all_files_verified'
    for name,reference in lock['official_files'].items():
        assert status['verified_files'][name]['sha256']==reference['sha256']
        assert (DATA/name).stat().st_size==reference['bytes']
    assert digest(DATA/'classification_tags.csv')==lock['official_files']['classification_tags.csv']['sha256']
    md=read(OLD/'prepared/manifest.json');assert digest(OLD/'prepared/manifest.json')==report['prepared_manifest_sha256']
    assert [r['product'] for r in md['rows']]==[g['representative'] for g in lock['rows']]
    indices=[];inputs={str(OLD/'delivery/phase72_report.json'):digest(OLD/'delivery/phase72_report.json')}
    for name in ['source','msre']:
        p=OLD/('scores_'+name)/'index.json';assert digest(p)==report['score_index_sha256'][name];inputs[str(p)]=digest(p)
        idx=read(p);assert idx['manifest_sha256']==report['prepared_manifest_sha256'];indices.append({r['product']:r for r in idx['rows']})
    refs={n:{r['product']:r for r in report['policies'][n]['rows']} for n in ['L3','B']}
    ROOT.mkdir();(ROOT/'panels').mkdir();(ROOT/'blind_pages').mkdir();(ROOT/'label_pages').mkdir()
    allrows=[];prepared={r['product']:r for r in md['rows']}
    def load(r):
        product=r['product']
        for suffix,sha in r['prepared_sha256'].items():
            path=OLD/'prepared'/(r['prefix']+suffix);assert digest(path)==sha;inputs[str(path)]=sha
        truth=np.load(OLD/'prepared'/r['mask_path']);valid=truth!=255
        scores=[]
        for idx in indices:
            item=idx[product]['files']['score'];p=OLD/item['path'];assert digest(p)==item['sha256'];inputs[str(p)]=item['sha256'];scores.append(np.load(p)[valid])
        item=indices[1][product]['files']['nonshadow'];p=OLD/item['path'];assert digest(p)==item['sha256'];inputs[str(p)]=item['sha256'];ns=np.load(p)[valid]
        spec=np.load(OLD/'prepared'/(r['prefix']+'_spectral.npy'))[valid]
        original=np.column_stack([logit_score(scores[1]),logit_score(scores[0]),spec]).astype(np.float32)
        preds={};fullscores={}
        for n in ['L3','B']:
            x=original if n=='L3' else np.column_stack([original,np.load(OLD/'prepared'/(r['prefix']+'_B.npy'))])
            s=predict_logit(x,lock['models'][n]);pred=corrected(s,lock['models'][n]['threshold'],ns)
            assert np.array_equal(confusion(truth[valid],pred),refs[n][product]['confusion'])
            preds[n]=np.full(truth.shape,255,np.uint8);preds[n][valid]=pred
            fullscores[n]=np.full(truth.shape,np.nan,np.float32);fullscores[n][valid]=s
        return truth,valid,preds,fullscores
    for i,r in enumerate(md['rows']):
        truth,valid,pred,_=load(r);product=r['product'];cm={n:np.array(refs[n][product]['confusion']) for n in ['L3','B']}
        row=dict(product=product,group_id=r['group_id'],delta_fpr=refs['B'][product]['fpr']-refs['L3'][product]['fpr'],
            delta_ap_pp=None if refs['B'][product]['ranking']['ap'] is None else 100*(refs['B'][product]['ranking']['ap']-refs['L3'][product]['ranking']['ap']))
        for n in ['L3','B']:
            for name,a,b in [('Surface_to_Shadow',0,2),('Cloud_to_Shadow',1,2),('Shadow_to_Surface',2,0),('Shadow_to_Cloud',2,1)]:
                den=int(cm[n][a].sum());row[n+'_'+name]=int(cm[n][a,b]);row[n+'_'+name+'_denominator']=den;row[n+'_'+name+'_fraction']=float(cm[n][a,b]/den) if den else None
        a=pred['L3']==2;b=pred['B']==2;s=truth==2
        flags=[valid&~s&b&~a,valid&s&a&~b,valid&s&b&~a,valid&s&a&b]
        for name,m in zip(CATS,flags):row[name]=int(m.sum())
        row['removed_FP']=int((valid&~s&a&~b).sum());allrows.append(row)
        assert row['new_FP']-row['removed_FP']==refs['B'][product]['false_positive']-refs['L3'][product]['false_positive']
        assert row['rescued_TP']-row['lost_TP']==refs['B'][product]['true_positive']-refs['L3'][product]['true_positive']
        print('Phase73 confusion and flows',i+1,64,flush=True)
    table(ROOT/'error_flows.csv',allrows);picks=choose(allrows);write(ROOT/'selected_groups.json',picks)
    tags={r['scene']:r for r in csv.DictReader((DATA/'classification_tags.csv').open())};support=[];points=[];sampling=[];stretch=[]
    # Open official archives only for the prespecified selected or disagreement products.
    with zipfile.ZipFile(DATA/'subscenes.zip') as images,zipfile.ZipFile(DATA/'masks.zip') as masks:
        for product in report['support_disagreements']:
            r=prepared[product];rawbytes=images.read('subscenes/'+product+'.npy');mb=masks.read('masks/'+product+'.npy')
            assert hashlib.sha256(rawbytes).hexdigest()==r['member_sha256']['image'] and hashlib.sha256(mb).hexdigest()==r['member_sha256']['mask']
            raw=np.load(io.BytesIO(rawbytes));mask=np.load(io.BytesIO(mb));truth,valid,_,_=load(r);iv=np.load(OLD/'prepared'/(r['prefix']+'_image_valid.npy'))
            onehot=mask.sum(-1)==1;rawshadow=mask[...,2];effective=valid&(truth==2)
            count=int(effective.sum());inval=int((rawshadow&~valid).sum())
            reason='public_ratio_positive_but_official_mask_shadow_channel_empty' if not rawshadow.any() else 'all_reference_shadow_excluded_by_validity_mask' if not count else 'unknown'
            support.append(dict(product=product,public_shadow_percent=float(tags[product]['shadow_percent']),raw_shadow_channel_count=int(rawshadow.sum()),
                effective_shadow_count=count,invalid_shadow_channel_count=inval,non_onehot_shadow_count=int((rawshadow&~onehot).sum()),
                image_invalid_shadow_count=int((rawshadow&~iv).sum()),reason=reason,upstream_cause='unknown'))
        for scene_no,product in enumerate(sorted(picks),1):
            r=prepared[product];rb=images.read('subscenes/'+product+'.npy');assert hashlib.sha256(rb).hexdigest()==r['member_sha256']['image'];raw=np.load(io.BytesIO(rb))
            truth,valid,pred,score=load(r);iv=np.load(OLD/'prepared'/(r['prefix']+'_image_valid.npy'));bounds={}
            for ch in [1,2,3,7,11,12]:bounds[ch]=np.percentile(raw[...,ch][iv],[2,98]).tolist()
            stretch.append(dict(product=product,bands={str(ch):v for ch,v in bounds.items()},scope='product_image_valid_2_98_percentile_display_only'))
            displays={}
            for ch in bounds:
                lo,hi=bounds[ch];a=np.clip((raw[...,ch]-lo)/(hi-lo if hi>lo else 1),0,1);a=np.nan_to_num(a)
                displays[ch]=(a*255).astype(np.uint8)
            rgb=np.stack([displays[ch] for ch in [3,2,1]],-1);fcc=np.stack([displays[ch] for ch in [12,7,3]],-1)
            rgb[~iv]=150;fcc[~iv]=150
            a=pred['L3']==2;b=pred['B']==2;s=truth==2;flags=[valid&~s&b&~a,valid&s&a&~b,valid&s&b&~a,valid&s&a&b]
            scene_points=[]
            for c,m in enumerate(flags):
                selected,record=sample(m,product,c);sampling.append(dict(product=product,category=CATS[c],**record))
                for j,flat in enumerate(selected):
                    y,x=np.unravel_index(flat,truth.shape);pid=f'S{scene_no:02d}_{CATS[c]}_{j+1:02d}'
                    point=dict(point_id=pid,product=product,group_id=r['group_id'],category=CATS[c],row=int(y),column=int(x),
                        gt=int(truth[y,x]),L3_prediction=int(pred['L3'][y,x]),B_prediction=int(pred['B'][y,x]),
                        L3_logit=float(score['L3'][y,x]),B_logit=float(score['B'][y,x]),
                        imagery_panel='panels/'+pid+'_imagery.png',label_panel='panels/'+pid+'_labels.png')
                    points.append(point);scene_points.append(point)
                    chips=[crop(rgb,y,x,150),crop(fcc,y,x,150)]
                    for ch in [7,11,12]:
                        grey=np.repeat(displays[ch][...,None],3,-1);grey[~iv]=150;chips.append(crop(grey,y,x,150))
                    vm=np.repeat((valid.astype(np.uint8)*255)[...,None],3,-1);chips.append(crop(vm,y,x,150))
                    tiles=[marked(a,title) for a,title in zip(chips,['RGB','B12/B08/B04','B08','B11','B12','reference-valid mask'])]
                    pack(tiles,3).save(ROOT/point['imagery_panel'])
                    label_tiles=[]
                    for name,label in [('GT',truth),('L3 prediction',pred['L3']),('B prediction',pred['B'])]:
                        indices=np.where(label==255,3,label);label_tiles.append(marked(crop(COLORS[indices],y,x,150),name))
                    # Scores remain logits, displayed using frozen sigmoid [0,1], identical across policies.
                    for name in ['L3','B']:
                        prob=1/(1+np.exp(-np.clip(score[name],-80,80)));g=np.repeat((np.nan_to_num(prob)*255).astype(np.uint8)[...,None],3,-1);g[~valid]=150
                        label_tiles.append(marked(crop(g,y,x,150),name+' sigmoid(score) [0,1]'))
                    pack(label_tiles,3).save(ROOT/point['label_panel'])
            # Blind overview includes full native RGB/FCC chips; labels are separate pages.
            for start in range(0,len(scene_points),4):
                tiles=[];labeltiles=[]
                for point in scene_points[start:start+4]:
                    y=point['row'];x=point['column'];pid=point['point_id']
                    tiles.extend([marked(crop(rgb,y,x,150),pid+' RGB'),marked(crop(fcc,y,x,150),pid+' FCC')])
                    for ch in [7,11,12]:
                        grey=np.repeat(displays[ch][...,None],3,-1);grey[~iv]=150
                        tiles.append(marked(crop(grey,y,x,150),pid+' '+{7:'B08',11:'B11',12:'B12'}[ch]))
                    for name,label in [('GT',truth),('L3',pred['L3']),('B',pred['B'])]:labeltiles.append(marked(crop(COLORS[np.where(label==255,3,label)],y,x,150),pid+' '+name))
                pack(tiles,5).save(ROOT/'blind_pages'/f'S{scene_no:02d}_page{start//4+1}.png')
                pack(labeltiles,3).save(ROOT/'label_pages'/f'S{scene_no:02d}_page{start//4+1}.png')
            print('Phase73 panels',scene_no,len(picks),product,len(scene_points),flush=True)
    table(ROOT/'support_difference.csv',support);table(ROOT/'sampled_points.csv',points);write(ROOT/'sampling_audit.json',sampling);write(ROOT/'display_stretch.json',stretch)
    interpretations=[dict(point_id=p['point_id'],product=p['product'],group_id=p['group_id'],category=p['category'],
        imagery_types='',imagery_evidence='',confidence='',label_issue_types='',label_evidence='',status='pending_visual_review') for p in points]
    table(ROOT/'interpretation.csv',interpretations)
    geo=DATA/'phase70_20261010/metadata_readiness.json'
    if not geo.exists():geo=DATA/'phase70_20261010/delivery/metadata_readiness.json'
    readiness=read(geo);inputs[str(geo)]=digest(geo)
    missing=[dict(field=k,status='not_ready',required_source=v) for k,v in [
        ('solar_azimuth/zenith','Exact original-product/granule solar-angle metadata with verified product correspondence'),
        ('cube_CRS/affine/pixel_size/orientation','Verified original raster grid and reproducible crop/resampling provenance for released cube'),
        ('cube_georeferencing_correspondence','Control-point/corner correspondence; annotation CRS alone insufficient'),
        ('cloud_height','Independent cloud-height information if required by a later projection hypothesis; absent, do not assume')]]
    table(ROOT/'geometry_missing.csv',missing)
    write(ROOT/'geometry_readiness.json',dict(status='not_ready',existing_record_status=readiness['status'],existing_record_products=readiness['products'],
        note='Existing Phase70 list covers fit/development, not a verified per-product geometry audit of 65c; no new verified geometry acquired'))
    for p in [OLD/'delivery/confirmation_lock.json',OLD/'delivery/execution_lock.json',OLD/'prepared/manifest.json',DATA/'classification_tags.csv',REPO/'src/tools/run_phase73.py',REPO/'src/research_plans/PHASE73_FAILURE_TYPING_20261010.md']:
        inputs[str(p)]=digest(p)
    write(ROOT/'input_sha256.json',inputs)
    summary=dict(status='numerical_analysis_and_panels_complete_visual_review_pending',groups=64,selected_groups=len(picks),points=len(points),
        confusion_reproduced=True,new_network_forwards=0,fits=0,threshold_changes=0,
        totals={k:sum(r[k] for r in allrows) for k in ['new_FP','removed_FP','rescued_TP','lost_TP','stable_TP']},
        support_difference=support,phase72_failure_unchanged=True,used65c_now_exploratory=True)
    write(ROOT/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()

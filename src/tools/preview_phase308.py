"""Fixed three-scene alignment panels; display clipping never changes features."""
import io
import json
import struct
import zipfile
import numpy as np
from PIL import Image,ImageDraw
from phase308_common import *
from math_phase308 import ah

def grid(mask):
    if not mask.any():return None
    choices=[(int(mask[t:min(t+128,1022),l:min(l+128,1022)].sum()),t,l) for t in range(0,1022,128) for l in range(0,1022,128)]
    n,t,l=min(choices,key=lambda a:(-a[0],a[1],a[2]))
    return dict(top=t,left=l,bottom=min(t+128,1022),right=min(l+128,1022),target_count=n)
def panel(path,arrays,box,title):
    names=('RGB TOA [0,1]','B11 [0,1]','B12 [0,1]','Reference','Source','Q90','MS6 fixed .5')
    if box is None:
        canvas=Image.new('RGB',(896,180),'white');ImageDraw.Draw(canvas).text((5,5),title+' NA: no target pixels',fill='black');canvas.save(path);return
    t,l,b,r=[box[k] for k in ('top','left','bottom','right')];height=b-t;width=r-l
    canvas=Image.new('RGB',(width*7,max(height,128)+58),'white');draw=ImageDraw.Draw(canvas)
    draw.text((3,3),title+f' rows[{t},{b}) cols[{l},{r})',fill='black')
    for j,(name,arr) in enumerate(zip(names,arrays)):
        canvas.paste(Image.fromarray(arr[t:b,l:r]),(j*width,40));draw.text((j*width+2,23),name,fill='black')
    draw.text((3,max(height,128)+43),'Classes: Surface=green, Cloud=white, Shadow=blue; invalid=magenta',fill='black');canvas.save(path)
def previews(groups,top,inputs,pin):
    target=ROOT/'previews';target.mkdir()
    directory=Path('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871')
    docs={}
    for n in ('record.json','README.pdf','classification_tags.csv'):
        p=directory/n;docs[n]=dict(sha256=pin(p),bytes=p.stat().st_size)
    record=load(directory/'record.json')
    docs['dataset_description']=record['metadata']['description']
    docs['README_content_gap']='No existing PDF text parser/pdftotext in fixed runtime; SHA retained; no environment change.'
    docs['alt_masks_status']='unknown: existing record describes primary 3-class masks; alternate encoding/process/independence not established; alternate masks not read or treated as gold.'
    write(ROOT/'DOCUMENT_AUDIT.json',docs)
    records={r['product']:r for r in load(B/'EVAL_IMAGES.json')['rows']};pin(B/'EVAL_IMAGES.json')
    plan={r['product_id']:r for r in load(Path('/home/scv/shared/phase300_20261020/EVALUATION_INPUT_PLAN.json'))['rows'] if r['pool']=='Catalogue'}
    pin(Path('/home/scv/shared/phase300_20261020/EVALUATION_INPUT_PLAN.json'))
    prior=load(P/'INPUT_SHA256.json');tags=read(directory/'classification_tags.csv')
    manifest={r['product']:r for r in read(P/'SUPPORT_AND_SAMPLE_SHA.csv')}
    rows=[];coords=[];audit=[];saturation=[];pal=np.array([[45,140,60],[250,250,250],[35,70,180],[220,20,220]],np.uint8)
    with zipfile.ZipFile(directory/'subscenes.zip') as images,zipfile.ZipFile(directory/'masks.zip') as masks,zipfile.ZipFile(directory/'shapefiles.zip') as shapes:
        for pid in top:
            g=next(g for g in groups if g['product']==pid);row=records[pid];raw=images.read(row['image_member']);h=__import__('hashlib').sha256(raw).hexdigest()
            assert h==row['image_member_sha256']==manifest[pid]['cube_member_sha256'];cube=np.load(io.BytesIO(raw));assert cube.shape==(1022,1022,13)
            rawmask=masks.read('masks/'+pid+'.npy');mh=__import__('hashlib').sha256(rawmask).hexdigest();assert mh==plan[pid]['group']['mask_member_sha256']
            mask=np.load(io.BytesIO(rawmask));assert mask.shape==(1022,1022,3);unique=np.unique(mask);onehot=np.isin(mask,[0,1]).all(-1)&(mask.sum(-1)==1)
            with np.load(P/manifest[pid]['samples_file']) as z:
                common=np.unpackbits(z['common_packbits'],count=1022**2).astype(bool).reshape(1022,1022)
                fullflat=z['flat_indices'];event=z['event'];toa=z['TOA']
            assert ah(common)==manifest[pid]['common_sha256'] and onehot[common].all()
            assert np.array_equal(cube.reshape(-1,13)[fullflat][:,INDICES],toa)
            truth=mask.argmax(-1).astype(np.uint8);displaytruth=truth.copy();displaytruth[~onehot]=3
            endpoints=[]
            for base,prefix in ((A,'Source_'),(B,'LoRA_')):
                p=base/'evaluation'/(prefix+pid+'.npz');pin(p,prior[str(p)])
                with np.load(p) as z:
                    pred=z['pred'];valid=z['valid'];s=z['scores'] if base==A else None
                assert pred.shape==(1022,1022) and not (common&~valid).any();endpoints.append(pred)
            source,q90=endpoints
            with np.load(A/'evaluation'/('Source_'+pid+'.npz')) as z:ss=z['scores']
            denom=ss.sum(0);qs=np.divide(ss[0],denom,out=np.zeros_like(denom),where=denom>1e-12)
            region=common&(source==0)&(denom>1e-12)&(qs<np.float32(.9))
            expected_event=np.where(truth==0,np.where(q90==1,0,1),np.where((truth==1)&(q90==1),2,3))
            assert np.array_equal(expected_event.ravel()[fullflat],event)
            assert np.all(region.ravel()[fullflat]) and np.all(q90.ravel()[g['flat']]==1)
            with np.load(Q/'oof'/f"fold{g['fold']:02d}_MS6.npz") as z:p=z['p_K'];assert np.array_equal(z['flat_indices'],g['flat'])
            ms=q90.copy();ms.ravel()[g['flat'][p<.5]]=0
            dd=np.zeros((1022,1022),bool);kk=dd.copy();dd.ravel()[g['flat'][g['codes']==0]]=True;kk.ravel()[g['flat'][g['codes']==2]]=True
            assert int(dd.sum())==int(((event==0)).sum()) and int(kk.sum())==int((event==2).sum())
            display=[]
            for name,ix in [('RGB',[3,2,1]),('B11',[11]),('B12',[12])]:
                a=cube[:,:,ix];finite=np.isfinite(a)
                saturation.append(dict(product=pid,channel=name,total_values=a.size,below_zero=int((a<0).sum()),above_one=int((a>1).sum()),nonfinite=int((~finite).sum()),clipped_fraction=float(((a<0)|(a>1)|~finite).mean()),display_only=True))
                v=np.rint(np.clip(np.nan_to_num(a,nan=0,posinf=1,neginf=0),0,1)*255).astype(np.uint8)
                if v.shape[-1]==1:v=np.repeat(v,3,axis=-1)
                display.append(v)
            display.extend(pal[a] for a in (displaytruth,source,q90,ms))
            tile=next(t for t in TOP if t in pid)
            boxes=[('panorama',dict(top=0,left=0,bottom=1022,right=1022,target_count=None)),('center',dict(top=447,left=447,bottom=575,right=575,target_count=None)),('max_D',grid(dd)),('max_K',grid(kk))]
            for name,box in boxes:
                path=target/(tile+'_'+name+'.png');panel(path,display,box,tile+' '+name)
                coords.append(dict(product=pid,selection=name,box=box,path=path.relative_to(ROOT).as_posix(),sha256=sha(path),D=int(dd[box['top']:box['bottom'],box['left']:box['right']].sum()) if box else None,K=int(kk[box['top']:box['bottom'],box['left']:box['right']].sum()) if box else None,flat_index_formula='row*1022+column',NA=box is None))
            matched=[t for t in tags if t['scene']==pid or t['scene'].removesuffix('.npy')==pid]
            assert len(matched)==1 and str(matched[0]['snow/ice'])==g['snow']
            geodata=[]
            for n in shapes.namelist():
                if pid in n and n.lower().endswith(('.shp','.prj')):
                    data=shapes.read(n);v=dict(member=n,sha256=__import__('hashlib').sha256(data).hexdigest())
                    if n.lower().endswith('.shp'):v['bounding_box_xy']=list(struct.unpack('<4d',data[36:68]));assert struct.unpack('>i',data[:4])[0]==9994
                    else:v['CRS_WKT']=data.decode('utf-8',errors='replace')
                    geodata.append(v)
            counts=[int((truth==i).sum()) for i in range(3)];commoncounts=[int(((truth==i)&common).sum()) for i in range(3)]
            audit.append(dict(product=pid,image_member=row['image_member'],image_sha256=h,image_shape=list(cube.shape),image_dtype=str(cube.dtype),mask_sha256=mh,mask_shape=list(mask.shape),mask_dtype=str(mask.dtype),unique_values=unique.tolist(),one_hot_all=bool(onehot.all()),one_hot_common=True,raw_counts=counts,common_counts=commoncounts,common_support=int(common.sum()),snow_ice=g['snow'],original_tags=matched[0],original_coordinate_metadata=geodata,coordinate_gap=None if geodata else 'no exact product match in existing shapefiles',identity_outcome='未发现明显身份异常',reference_semantics_outcome='无法判定：原参考与缓存一致，不能由RGB或snow/ice场景标签证明像元真值',alt_masks='unknown'))
            rows.append(dict(product=pid,tile=tile,D=int(dd.sum()),K=int(kk.sum()),common=int(common.sum()),common_Surface=commoncounts[0],common_Cloud=commoncounts[1],common_Shadow=commoncounts[2],snow_ice=g['snow'],image_SHA=h,reference_SHA=mh,one_hot_all=bool(onehot.all()),identity='未发现明显身份异常',reference='无法判定',alt_masks='unknown'))
    write(ROOT/'IMAGE_REFERENCE_AUDIT.json',audit);write(ROOT/'PREVIEW_COORDINATES.json',coords);table(ROOT/'IMAGE_REFERENCE_AUDIT.csv',rows);table(ROOT/'DISPLAY_SATURATION.csv',saturation)

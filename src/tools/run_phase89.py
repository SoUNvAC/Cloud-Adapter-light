"""Phase89 fixed ALCD reference consistency; no fit, tuning or downloads."""
import argparse,csv,hashlib,io,json,os,subprocess,sys,tarfile,time,zipfile
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from phase89_math import NAMES,area,all_valid,calibrate,decisions,interval,sigmoid
from phase65d_metrics import curve,confusion,measures,corrected
from phase65d_recovery_math import predict_logit
from freeze_phase66 import RECOVERY_SHA,code_digest
from prepare_phase65a_explore import SOURCE_SHA,SPLIT_SHA,digest
from run_phase65d_recovery import ADAPTED_SHA,records,prepare_verify,load,features

REPO=Path('/home/scv/Cloud-Adapter-light');SHARED=Path('/home/scv/shared')
ROOT=SHARED/'phase89_20261011_run02';ALCD=SHARED/'data/sentinel2_alcd_1460961';IM=ALCD/'imagery'
CAT=SHARED/'data/sentinel2_cloud_mask_catalogue_4172871'
REC=CAT/'phase65d_recovery_20261008';OLD=CAT/'phase65d_20261008_recovery01';PREP=CAT/'phase65a_explore_20261008'
BIND=REPO/'src/research_plans/PHASE89_INPUT_BINDINGS_20261011.json'
PROTOCOL=REPO/'src/research_plans/PHASE89_EXTERNAL_REFERENCE_EVALUATION_20261011.md'
HUMAN=REPO/'src/research_results/phase88_final_20261011/expert_A_final_original.json'
SAMPLES=REPO/'src/research_results/phase88_20261011/sample_input_lock.json'
CK={'Source':REPO/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth',
    'MsRE':REPO/'src/work_dirs/phase65a_explore_20261008/msre_seed65/best_mIoU_iter_2000.pth'}
BANDS=('B02','B03','B04','B08','B11','B12')
ORIGINAL=ALCD/'SENTINEL_2_reference_cloud_masks_Baetens_Hagolle.tgz'
REVISED=IM/'literature_CESBIO2024.tar.gz'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def csvwrite(p,rows):
    with Path(p).open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def now():return datetime.now(timezone.utc).isoformat()
def save(p,a):
    with Path(p).open('xb') as f:np.save(f,a,allow_pickle=False)
def codepaths():
    return [p for p in (REPO/'src/tools').glob('*phase89*') if p.is_file()]+list((REPO/'src/configs/protocol').glob('phase65a_explore_rgb.py'))+[REPO/'src/tools/phase65d_recovery_math.py',REPO/'src/tools/phase65d_metrics.py',REPO/'src/cloud_adapter/datasets/phase65_catalogue.py']
def locked():
    lock=read(ROOT/'input_scale_threshold_reference_lock.json')
    assert digest(BIND)==lock['bindings_sha256'] and code_digest(PROTOCOL)==lock['protocol_sha256']
    for p,h in lock['code_sha256'].items():assert code_digest(REPO/p)==h,p
    assert digest(REC/'recovery_lock.json')==RECOVERY_SHA
    return lock,read(BIND),read(REC/'recovery_lock.json')['models']

def reproduce():
    """Only already exposed Catalogue development, no new forwards or fits."""
    assert digest(REC/'recovery_lock.json')==RECOVERY_SHA
    models=read(REC/'recovery_lock.json')['models'];report=read(REC/'recovery_report.json')
    idx=[{r['product']:r for r in read(OLD/'development_val'/('index_'+n+'.json'))['rows']} for n in ['source','msre']]
    verified=[]
    with zipfile.ZipFile(CAT/'subscenes.zip') as z:
        for product,r in records('development_val').items():
            prepare_verify(r);truth=np.load(PREP/r['mask_path']);v=truth!=255
            b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
            raw=np.load(io.BytesIO(b));ss=load(OLD,idx[0][product],'score');ms=load(OLD,idx[1][product],'score')
            x=features(raw,ms,ss,v);ns=load(OLD,idx[1][product],'nonshadow')[v]
            row=dict(product=product,group_id=r['group_id'])
            for n,key,k in [('L2','msre_source',2),('L3','msre_source_spectral',5)]:
                score=predict_logit(x[:,:k],models[key]);path=REC/'development_scores'/(product+'_'+key+'.npy')
                assert np.array_equal(score,np.load(path)),('score drift',product,n)
                cm=confusion(truth[v],corrected(score,models[key]['threshold'],ns))
                ref=next(t for t in report['policies'][key]['rows'] if t['product']==product)
                assert np.array_equal(cm,np.asarray(ref['confusion'])),('confusion drift',product,n)
                row[n]=dict(score_sha256=digest(path),confusion=cm.tolist(),score_bitwise_equal=True)
            verified.append(row);print('Catalogue reproduction',len(verified),32,flush=True)
    assert len(verified)==32
    write(ROOT/'delivery/catalogue_reproduction.json',dict(created_utc=now(),groups=32,rows=verified,
        recovery_sha256=RECOVERY_SHA,reference_report_sha256=digest(REC/'recovery_report.json'),no_fit_no_forward=True))

def freeze():
    assert not ROOT.exists(),'Preserve any Phase89 attempt; never automatic rerun'
    ROOT.mkdir();(ROOT/'delivery').mkdir();b=read(BIND)
    assert b['recovery_sha256']==RECOVERY_SHA and digest(REC/'recovery_lock.json')==RECOVERY_SHA
    assert digest(REPO/'src/work_dirs/phase65a/split_lock.json')==SPLIT_SHA
    for n,h in [('Source',SOURCE_SHA),('MsRE',ADAPTED_SHA)]:assert digest(CK[n])==h
    assert digest(HUMAN)==b['human_original_sha256'] and digest(SAMPLES)==b['sample_lock_sha256']
    assert digest(IM/'product_manifest.json')==b['manifest_sha256']
    assert digest(IM/'geometry_audit.json')==b['geometry_sha256']
    assert digest(IM/'reference_disagreement_20261008.json')==b['reference_disagreement_sha256']
    assert digest(SHARED/'phase86_20261011_run04/candidate_lock.json')==b['phase86_candidate_sha256']
    for p,k in [(ORIGINAL,'original_archive_sha256'),(REVISED,'revised_archive_sha256')]:assert digest(p)==b[k]
    status=read(IM/'imagery_download_status.json');assert status['status']=='all_resolved_files_verified'
    checked=[]
    for r in b['rows']:
        assert r['geometry']['alignment_status']=='all_six_grids_match'
        assert not r['group']['confirmation_ready']
        for name,expected in r['objects'].items():
            f=status['files'][name]
            assert all(f[k]==v for k,v in expected.items()),name
            p=(IM/f['path'].replace('\\','/')).resolve();assert p.is_relative_to(IM)
            assert p.stat().st_size==expected['size'] and digest(p)==expected['sha256'],name
            checked.append(dict(object=name,path=str(p),**expected))
    reproduce()
    # All identity/grid/chain rules frozen before any ALCD model forward or metric.
    lock=dict(created_utc=now(),bindings_sha256=digest(BIND),protocol_sha256=code_digest(PROTOCOL),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_sha256=SOURCE_SHA,msre_2000_sha256=ADAPTED_SHA,recovery_sha256=RECOVERY_SHA,
        original_split_sha256=SPLIT_SHA,groups=[r['group'] for r in b['rows']],objects=checked,
        classes=['Surface','Cloud','Shadow'],models=read(REC/'recovery_lock.json')['models'],
        code_sha256={str(p.relative_to(REPO)):code_digest(p) for p in codepaths()},
        processing=dict(DN_to_TOA='(DN+XML band offset)/XML quantification',area_10m_to_20m='all four valid, arithmetic area mean',
            RGB='B04/B03/B02 TOA*255, no clipping; BGR loader then original preprocessor',
            block_60m=340,block_20m=1020,network_input=512,padding='right/bottom zero; never evaluated',
            model_postprocess='original MMSeg postprocess_result restores 1020; crop true edge then assemble',
            L2_L3='original float32 calibrated logits at20m; sigmoid float64 then 3x3 probability mean',
            scores='60m Source/MsRE softmax probability means; L2/L3 mean sigmoid; AP uses existing tied-score float32 implementation',
            threshold='sigmoid(frozen original logit threshold), >=; no 60m recalibration',
            common_mask='original2..7 AND all six bands every3x3 at20m valid, including every2x2 fine contributor',
            spectral='physical TOA B08/B11/B12; no labels/GT proportions/confidence features'),
        revisions='exact product + CRS/bounds/3x3 homogeneous valid support only; both versions same support',
        prior_exposure='quality metadata, sampled image/reference inspection and human A; not unexposed confirmation',
        human_B='not_executed; automatic comparison only',zero_training_fitting_tuning_downloads=True,
        catalogue_reproduction_sha256=digest(ROOT/'delivery/catalogue_reproduction.json'),frozen_before_ALCD_prediction=True)
    write(ROOT/'input_scale_threshold_reference_lock.json',lock)
    write(ROOT/'delivery/input_scale_threshold_reference_lock.json',lock)
    print('Phase89 FROZEN:9 groups/6 revised matches',flush=True)

def radiometry(row,status):
    xml=[o for o in row['scene']['objects'] if o['name'].endswith('.xml') and '/GRANULE/' not in o['name'] and not o['name'].endswith('/INSPIRE.xml')]
    assert len(xml)==1
    tree=ET.parse(IM/status['files'][xml[0]['name']]['path'].replace('\\','/'))
    vals=[float(e.text) for e in tree.iter() if e.tag.split('}')[-1]=='QUANTIFICATION_VALUE'];assert len(vals)==1 and vals[0]>0
    offsets={int(e.attrib['band_id']):float(e.text) for e in tree.iter() if e.tag.split('}')[-1]=='RADIO_ADD_OFFSET'}
    ids={'B02':1,'B03':2,'B04':3,'B08':7,'B11':11,'B12':12}
    assert not offsets or all(i in offsets for i in ids.values()),'Partial XML offsets'
    special={0}
    for parent in tree.iter():
        c={e.tag.split('}')[-1]:e.text for e in parent}
        if 'SPECIAL_VALUE_TEXT' in c and 'SPECIAL_VALUE_INDEX' in c:special.add(int(c['SPECIAL_VALUE_INDEX']))
    return dict(quantification=vals[0],offsets={k:offsets.get(v,0.) for k,v in ids.items()},special_values=sorted(special),xml_sha256=digest(IM/status['files'][xml[0]['name']]['path'].replace('\\','/')))

def prepare():
    import rasterio
    from rasterio.io import MemoryFile
    lock,b,models=locked();dest=ROOT/'prepared';dest.mkdir();status=read(IM/'imagery_download_status.json');out=[]
    with tarfile.open(ORIGINAL,'r:gz') as old,tarfile.open(REVISED,'r:gz') as rev:
        for row in b['rows']:
            product=row['group']['representative'];d=dest/product;d.mkdir();t0=time.monotonic()
            raw=old.extractfile(row['scene']['label_members'][0]).read();assert hashlib.sha256(raw).hexdigest()==row['geometry']['label_sha256']
            with MemoryFile(raw) as mem,mem.open() as lab:
                labels=lab.read(1);h,w=labels.shape;assert (h,w)==(1830,1830) and lab.res==(60,60)
                assert np.isin(labels,range(8)).all();lut=np.full(256,255,np.uint8);lut[[2,3]]=1;lut[4]=2;lut[[5,6,7]]=0;truth=lut[labels]
                rad=radiometry(row,status);spectra=np.empty((h*3,w*3,6),np.float32);v=np.ones((h*3,w*3),bool)
                for j,band in enumerate(BANDS):
                    objects=[o for o in row['scene']['objects'] if o['name'].endswith('_'+band+'.jp2')];assert len(objects)==1
                    path=IM/status['files'][objects[0]['name']]['path'].replace('\\','/')
                    with rasterio.open(path) as im:
                        factor=6 if band in BANDS[:4] else 3
                        assert im.crs==lab.crs and np.allclose(im.bounds,lab.bounds,rtol=0,atol=1e-6)
                        assert im.res==(60/factor,60/factor) and im.shape==(h*factor,w*factor) and im.transform.b==im.transform.d==0
                        a=im.read(1);ok=im.read_masks(1)>0
                        for x in rad['special_values']:ok &= a!=x
                        toa=(a.astype(np.float32)+rad['offsets'][band])/rad['quantification'];ok &= np.isfinite(toa)
                        if factor==6:toa=area(toa,2).astype(np.float32);ok=all_valid(ok,2)
                        toa[~ok]=0.;spectra[...,j]=toa;v &= ok
                    print('prepare',product,band,flush=True)
                valid=(truth!=255)&all_valid(v,3)
                assert int(valid.sum())==row['geometry']['joint_valid_pixels_60m']
                revised=np.full((h,w),255,np.uint8);common=np.zeros((h,w),bool);counts=None
                if row['revised']:
                    rr=row['revised'];rb=rev.extractfile(rr['reference_member']).read();assert hashlib.sha256(rb).hexdigest()==rr['revised_label_sha256']
                    with MemoryFile(rb) as rm,rm.open() as nl:
                        assert nl.crs==lab.crs and np.allclose(nl.bounds,lab.bounds,rtol=0,atol=1e-6)
                        assert nl.res==(20,20) and nl.shape==(h*3,w*3) and nl.transform.b==nl.transform.d==0
                        n=nl.read(1);assert np.isin(n,[0,64,128,255]).all()
                        ll=np.full(256,255,np.uint8);ll[64]=2;ll[128]=0;ll[255]=1;n=ll[n].reshape(h,3,w,3)
                        lo=n.min((1,3));hi=n.max((1,3));homogeneous=(lo==hi)&(lo<3)
                        revised=lo;common=valid&homogeneous
                        counts=dict(image_label_valid=int(valid.sum()),common_valid=int(common.sum()),
                            revised_invalid_or_mixed_excluded=int((valid&~homogeneous).sum()),
                            retention=float(common.sum()/valid.sum()) if valid.any() else None,
                            revised_native_fill_cells=int((n==255).any((1,3)).sum()),
                            revised_mixed_nonfill_cells=int(((lo!=hi)&(hi<3)).sum()))
                for name,a in [('spectra20',spectra),('image_valid20',v),('truth',truth),('valid',valid),('revised',revised),('common',common)]:save(d/(name+'.npy'),a)
                out.append(dict(product=product,group_id=row['group']['group_id'],alias=row['scene']['key'],tile=row['group']['tiles'][0],
                    radiometry=rad,shape60=[h,w],crs=str(lab.crs),transform=list(lab.transform)[:6],
                    revised_available=row['revised'] is not None,common_support=counts,
                    files={p.stem:dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for p in d.glob('*.npy')},seconds=time.monotonic()-t0))
    write(dest/'index.json',dict(lock_sha256=digest(ROOT/'input_scale_threshold_reference_lock.json'),rows=out));print('prepare complete',flush=True)

def array(r,key):
    f=r['files'][key];p=ROOT/f['path'];assert digest(p)==f['sha256'];return np.load(p,mmap_mode='r',allow_pickle=False)

def setup(name):
    import torch
    sys.path.insert(0,str(REPO/'src'))
    from mmengine.config import Config
    from mmseg.utils import register_all_modules
    register_all_modules();import cloud_adapter.models,cloud_adapter.datasets.phase65_catalogue
    from mmseg.registry import MODELS
    os.environ['PHASE65_PREPARED_MANIFEST']=str(PREP/'manifest.json');os.environ['PHASE65_SOURCE_CHECKPOINT']=str(CK['Source'])
    torch.manual_seed(65);np.random.seed(65)
    cfg=Config.fromfile(str(REPO/'src/configs/protocol/phase65a_explore_rgb.py'))
    assert cfg.model.test_cfg.mode=='whole'
    model=MODELS.build(cfg.model);model.init_weights()
    missing,unexpected=model.load_state_dict(torch.load(CK[name],map_location='cpu')['state_dict'],strict=False)
    assert not unexpected and all(k.startswith(('backbone.target_msre.','backbone.target_head_delta.')) for k in missing)
    if name=='MsRE':assert not missing
    model.train();model.backbone.set_target_enabled(name=='MsRE');model.requires_grad_(False);model.eval().cuda()
    assert not any(m.training for m in model.modules())
    from mmcv.transforms import Compose
    pipeline=Compose([dict(type='Resize',scale=(512,512)),dict(type='PackSegInputs')])
    return model,pipeline

def state_sha(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def infer(name):
    import torch
    from mmengine.dataset import pseudo_collate
    lock,b,models=locked();assert name in NAMES[:2]
    assert digest(CK[name])==(SOURCE_SHA if name=='Source' else ADAPTED_SHA)
    root=ROOT/('scores_'+name);root.mkdir();model,pipeline=setup(name);before=state_sha(model);rows=[]
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    for r in read(ROOT/'prepared/index.json')['rows']:
        product=r['product'];d=root/product;d.mkdir();spec=array(r,'spectra20');h,w=r['shape60'];probs=np.empty((h,w,3),np.float32);blocks=[]
        for y in range(0,h,340):
            for x in range(0,w,340):
                if datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()>='2026-10-20':raise RuntimeError('Budget deadline reached')
                hh=min(340,h-y);ww=min(340,w-x);hh20=hh*3;ww20=ww*3
                rgb=np.zeros((1020,1020,3),np.float32)
                rgb[:hh20,:ww20]=np.asarray(spec[y*3:y*3+hh20,x*3:x*3+ww20,[2,1,0]])*np.float32(255.)
                assert np.isfinite(rgb).all()
                sample=pipeline(dict(img=rgb[...,::-1].copy(),img_shape=(1020,1020),ori_shape=(1020,1020),img_path=product))
                assert tuple(sample['inputs'].shape)==(3,512,512)
                key=f'{y:04d}_{x:04d}';write(d/(key+'_forward_receipt.json'),dict(started_utc=now(),y60=y,x60=x,true_shape20=[hh20,ww20],input_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
                t=time.monotonic()
                with torch.no_grad():p=model.test_step(pseudo_collate([sample]))[0]
                assert tuple(p.seg_logits.data.shape)==(3,1020,1020)
                assert torch.equal(p.pred_sem_seg.data[0],p.seg_logits.data.argmax(0))
                full=p.seg_logits.data.softmax(0).permute(1,2,0).cpu().numpy().astype(np.float32)
                a=full[:hh20,:ww20].copy();assert np.isfinite(a).all() and np.allclose(a.sum(-1),1,rtol=0,atol=3e-7)
                save(d/(key+'.npy'),a);probs[y:y+hh,x:x+ww]=area(a,3).astype(np.float32)
                blocks.append(dict(y60=y,x60=x,h60=hh,w60=ww,path=str((d/(key+'.npy')).relative_to(ROOT)),sha256=digest(d/(key+'.npy')),
                    seconds=time.monotonic()-t,padding20=int(1020**2-hh20*ww20),seam_support='nonoverlap; no blend',input_shape=[1020,1020],network_shape=[512,512]))
                print('forward',name,product,key,len(blocks),36,flush=True)
        assert len(blocks)==36
        save(d/'prob60.npy',probs);rows.append(dict(product=product,group_id=r['group_id'],blocks=blocks,prob60_path=str((d/'prob60.npy').relative_to(ROOT)),prob60_sha256=digest(d/'prob60.npy')))
    after=state_sha(model);assert before==after
    write(root/'index.json',dict(lock_sha256=digest(ROOT/'input_scale_threshold_reference_lock.json'),checkpoint_sha256=digest(CK[name]),
        before_state_sha256=before,after_state_sha256=after,forward_count=sum(len(r['blocks']) for r in rows),
        elapsed_seconds=time.monotonic()-started,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),rows=rows))

def shallow():
    lock,b,models=locked();d=ROOT/'calibrated';d.mkdir();indices=[{r['product']:r for r in read(ROOT/('scores_'+n)/'index.json')['rows']} for n in NAMES[:2]];rows=[]
    for r in read(ROOT/'prepared/index.json')['rows']:
        product=r['product'];spec=array(r,'spectra20');h,w=r['shape60'];prob={n:np.empty((h,w),np.float64) for n in NAMES[2:]}
        sb=indices[0][product]['blocks'];mb=indices[1][product]['blocks'];assert len(sb)==len(mb)==36
        for s,m in zip(sb,mb):
            assert all(s[k]==m[k] for k in ['y60','x60','h60','w60'])
            for a in [s,m]:assert digest(ROOT/a['path'])==a['sha256']
            ss=np.load(ROOT/s['path']);ms=np.load(ROOT/m['path']);y=s['y60'];x=s['x60'];hh=s['h60'];ww=s['w60']
            p=calibrate(ms[...,2],ss[...,2],spec[y*3:(y+hh)*3,x*3:(x+ww)*3,3:],models)
            for n in prob:prob[n][y:y+hh,x:x+ww]=area(p[n],3)
        files={}
        for n,a in prob.items():
            path=d/(product+'_'+n+'.npy');save(path,a);files[n]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path))
        rows.append(dict(product=product,group_id=r['group_id'],files=files));print('calibration',product,flush=True)
    write(d/'index.json',dict(rows=rows,no_fit=True))

def probabilities(product):
    out={}
    for n in NAMES[:2]:
        r=next(r for r in read(ROOT/('scores_'+n)/'index.json')['rows'] if r['product']==product)
        assert digest(ROOT/r['prob60_path'])==r['prob60_sha256'];out[n]=np.load(ROOT/r['prob60_path'])
    r=next(r for r in read(ROOT/'calibrated/index.json')['rows'] if r['product']==product)
    for n in NAMES[2:]:out[n]=array(r,n)
    return out

def ap(scores,labels):return curve(scores,labels,plot_points=2)['ap']
def row_metrics(truth,pred,scores):
    cm=confusion(truth,pred);r=measures(cm);union=cm.sum(0)+cm.sum(1)-np.diag(cm)
    r.update(ap=ap(scores,truth==2),valid_pixels=int(len(truth)),valid_classes=int((union>0).sum()),
        surface_fpr=float(cm[0,2]/cm[0].sum()) if cm[0].sum() else None,
        cloud_fpr=float(cm[1,2]/cm[1].sum()) if cm[1].sum() else None,
        predicted_shadow_fraction=float((pred==2).mean()))
    return r

def analyze():
    lock,b,models=locked();rows={v:{n:[] for n in NAMES} for v in ['original_main','original_common','revised_common']}
    pooled={v:{n:[] for n in NAMES} for v in rows};ys={v:[] for v in rows};human=[];prep=read(ROOT/'prepared/index.json')['rows']
    samples=read(SAMPLES);judgments={r['review_id']:r for r in read(HUMAN)['rows']}
    for r in prep:
        product=r['product'];p=probabilities(product);pred=decisions(p,models)
        truth=array(r,'truth');valid=array(r,'valid');common=array(r,'common');revised=array(r,'revised')
        for v,t,mask in [('original_main',truth,valid),('original_common',truth,common),('revised_common',revised,common)]:
            if not mask.any():continue
            y=np.asarray(t[mask]);ys[v].append(y)
            for n in NAMES:
                scores=np.asarray(p[n][...,2][mask] if n in NAMES[:2] else p[n][mask]);pooled[v][n].append(scores)
                rows[v][n].append(dict(product=product,group_id=r['group_id'],tile=r['tile'],**row_metrics(y,pred[n][mask],scores)))
        for s in samples['samples']:
            if s['product']!=product:continue
            yy=s['row'];xx=s['col'];j=judgments[s['review_id']];oc=int(truth[yy,xx]);rc=int(revised[yy,xx]) if r['revised_available'] and common[yy,xx] else 255
            names=['Surface','Cloud','Shadow'];hc=j['judgment']
            def compare(c):return '无法判断' if hc=='无法判定' or c==255 else '混合' if hc=='混合' else '一致' if hc==names[c] else '不一致'
            human.append(dict(review_id=s['review_id'],stratum=s['phase88_stratum'],product=product,row60=yy,col60=xx,
                original_reference=names[oc] if oc<3 else 'invalid',revised_reference=names[rc] if rc<3 else 'unavailable_on_common_support',
                image_valid=bool(valid[yy,xx]),revised_common_valid=bool(common[yy,xx]),original_comparison=compare(oc),revised_comparison=compare(rc),
                **{k:j.get(k,'') for k in ['judgment','reason','small_sufficient','large_sufficient','needs_clear_date','needs_DEM','review_date']},human_B='not_executed'))
        print('reference consistency',product,flush=True)
    report=dict(created_utc=now(),lock_sha256=digest(ROOT/'input_scale_threshold_reference_lock.json'),versions={},
                reference_is_not_certified_truth=True,human_B='not_executed',formal_confirmation=False)
    tables=[];groupcsv=[];comparisons=[]
    for v in rows:
        stats={};labels=np.concatenate(ys[v]) if ys[v] else np.empty(0,np.uint8)
        for n in NAMES:
            rr=rows[v][n];valid_ap=[r['ap'] for r in rr if r['ap'] is not None]
            cm=np.sum([r['confusion'] for r in rr],axis=0);po=measures(cm)
            score=np.concatenate(pooled[v][n]) if pooled[v][n] else np.empty(0)
            stats[n]=dict(rows=rr,macro_ap=interval(valid_ap),macro_recall=interval([r['recall'] for r in rr if r['recall'] is not None]),
                macro_precision=interval([r['precision'] for r in rr if r['precision'] is not None]),macro_fpr=interval([r['fpr'] for r in rr if r['fpr'] is not None]),
                macro_miou_percent=interval([r['miou_percent'] for r in rr]),macro_iou_percent=[interval([r['iou_percent'][k] for r in rr if r['iou_percent'][k] is not None]) for k in range(3)],
                macro_surface_fpr=interval([r['surface_fpr'] for r in rr if r['surface_fpr'] is not None]),macro_cloud_fpr=interval([r['cloud_fpr'] for r in rr if r['cloud_fpr'] is not None]),
                pooled=dict(ap=ap(score,labels==2),**po),zero_recall_groups=[r['product'] for r in rr if r['recall']==0],
                groups_above_1percent=sum(r['fpr'] is not None and r['fpr']>.01 for r in rr),no_reference_shadow_groups=sum(r['positive']==0 for r in rr))
            q=stats[n];tables.append(dict(version=v,policy=n,groups=len(rr),ap_groups=len(valid_ap),macro_ap=q['macro_ap']['mean'],pooled_ap=q['pooled']['ap'],
                macro_recall=q['macro_recall']['mean'],macro_precision=q['macro_precision']['mean'],macro_fpr=q['macro_fpr']['mean'],fpr_upper95=q['macro_fpr']['upper95_one_sided'],
                macro_miou_percent=q['macro_miou_percent']['mean'],pooled_miou_percent=po['miou_percent'],pooled_recall=po['recall'],pooled_precision=po['precision'],pooled_fpr=po['fpr'],
                pooled_surface_iou=po['iou_percent'][0],pooled_cloud_iou=po['iou_percent'][1],pooled_shadow_iou=po['iou_percent'][2],
                shadow_to_surface=po['shadow_to_surface'],shadow_to_cloud=po['shadow_to_cloud'],groups_above_1percent=q['groups_above_1percent'],zero_recall_groups=len(q['zero_recall_groups'])))
            for t in rr:groupcsv.append(dict(version=v,policy=n,**{k:json.dumps(val) if isinstance(val,list) else val for k,val in t.items()}))
        pairs=[]
        for l2,l3 in zip(rows[v]['L2'],rows[v]['L3']):
            assert l2['group_id']==l3['group_id']
            pair=dict(version=v,product=l2['product'],group_id=l2['group_id'],delta_ap_pp=(l3['ap']-l2['ap'])*100 if l2['ap'] is not None and l3['ap'] is not None else None,
                delta_recall_pp=(l3['recall']-l2['recall'])*100 if l3['recall'] is not None and l2['recall'] is not None else None,
                delta_miou_pp=l3['miou_percent']-l2['miou_percent'])
            comparisons.append(pair);pairs.append(pair)
        delta=[p['delta_ap_pp'] for p in pairs if p['delta_ap_pp'] is not None];primary=interval(delta)
        primary.update(improved=sum(x>1 for x in delta),stable=sum(-1<=x<=1 for x in delta),declined=sum(x< -1 for x in delta),
            worst_delta_ap_pp=min(delta) if delta else None,worst_delta_recall_pp=min(p['delta_recall_pp'] for p in pairs if p['delta_recall_pp'] is not None),
            interpretation='descriptive reference consistency; not certified label accuracy or formal confirmation')
        report['versions'][v]=dict(policies=stats,primary_L3_minus_L2=primary,pairs=pairs)
        del score,pooled[v]
    assert len(human)==57 and {r['review_id'] for r in human}==set(judgments)
    report['human_automatic_comparison']={t:{ver:{c:sum(r['stratum']==t and r[ver+'_comparison']==c for r in human) for c in ['一致','不一致','混合','无法判断']} for ver in ['original','revised']} for t in ['P1','regular_control']}
    report['support']=[{k:r[k] for k in ['product','group_id','tile','revised_available','common_support','shape60','radiometry']} for r in prep]
    report['runtime']={n:{k:v for k,v in read(ROOT/('scores_'+n)/'index.json').items() if k!='rows'} for n in NAMES[:2]}
    d=ROOT/'delivery';write(d/'phase89_report.json',report);csvwrite(d/'four_policy_tables.csv',tables);csvwrite(d/'per_group.csv',groupcsv)
    csvwrite(d/'paired_AP_changes.csv',comparisons);csvwrite(d/'human57_automatic_comparison.csv',sorted(human,key=lambda r:r['review_id']))
    csvwrite(d/'reference_support.csv',[dict(product=r['product'],tile=r['tile'],revised_available=r['revised_available'],**(r['common_support'] or {})) for r in prep])

def verify():
    """Separate CPU statistics implementation, and CPU rebuild from native block outputs."""
    from sklearn.metrics import average_precision_score
    lock,b,models=locked();report=read(ROOT/'delivery/phase89_report.json');checked=[]
    for r in read(ROOT/'prepared/index.json')['rows']:
        product=r['product'];prob=probabilities(product);pred=decisions(prob,models)
        for name in NAMES[:2]:
            idx=next(t for t in read(ROOT/('scores_'+name)/'index.json')['rows'] if t['product']==product)
            coverage=np.zeros(r['shape60'],np.uint8)
            for block in idx['blocks']:
                a=np.load(ROOT/block['path']);assert digest(ROOT/block['path'])==block['sha256']
                y,x,h,w=[block[k] for k in ['y60','x60','h60','w60']]
                # Independent CPU explicit nine-strided sums (no area helper).
                cpu=sum(a[i::3,j::3].astype(np.float64) for i in range(3) for j in range(3))/9
                assert np.array_equal(cpu.astype(np.float32),prob[name][y:y+h,x:x+w])
                coverage[y:y+h,x:x+w]+=1
            assert (coverage==1).all()
        # Independently reconstruct calibrated20m logits and sigmoid means on CPU.
        si=next(t for t in read(ROOT/'scores_Source/index.json')['rows'] if t['product']==product)
        mi=next(t for t in read(ROOT/'scores_MsRE/index.json')['rows'] if t['product']==product)
        spectra=array(r,'spectra20')
        for s,m in zip(si['blocks'],mi['blocks']):
            sp=np.load(ROOT/s['path']);mp=np.load(ROOT/m['path']);y,x,h,w=[s[k] for k in ['y60','x60','h60','w60']]
            mscore=np.clip(mp[...,2].astype(np.float64),1e-6,1-1e-6);sscore=np.clip(sp[...,2].astype(np.float64),1e-6,1-1e-6)
            xx=np.column_stack([(np.log(mscore)-np.log1p(-mscore)).ravel(),(np.log(sscore)-np.log1p(-sscore)).ravel(),
                                spectra[y*3:(y+h)*3,x*3:(x+w)*3,3:].reshape(-1,3)]).astype(np.float32)
            for n,key,k in [('L2','msre_source',2),('L3','msre_source_spectral',5)]:
                model=models[key];z=(xx[:,:k]-np.array(model['mean']))/np.array(model['scale'])
                lg=(z@np.array(model['theta'][:-1])+model['theta'][-1]).astype(np.float32).astype(np.float64)
                pr=(np.exp(-np.logaddexp(0.,-lg))).reshape(h*3,w*3)
                cpu=sum(pr[i::3,j::3] for i in range(3) for j in range(3))/9
                np.testing.assert_allclose(cpu,prob[n][y:y+h,x:x+w],rtol=0,atol=3e-16)
        for v,t,mask in [('original_main',array(r,'truth'),array(r,'valid')),('original_common',array(r,'truth'),array(r,'common')),('revised_common',array(r,'revised'),array(r,'common'))]:
            if not mask.any():continue
            y=np.asarray(t[mask]);assert np.isin(y,[0,1,2]).all()
            for n in NAMES:
                p=pred[n][mask];ref=next(t for t in report['versions'][v]['policies'][n]['rows'] if t['product']==product)
                cm=np.array([[np.count_nonzero((y==i)&(p==j)) for j in range(3)] for i in range(3)])
                assert np.array_equal(cm,ref['confusion'])
                score=np.asarray(prob[n][...,2][mask] if n in NAMES[:2] else prob[n][mask],np.float32)
                a=float(average_precision_score(y==2,score)) if (y==2).any() and (y!=2).any() else None
                assert a is None and ref['ap'] is None or a is not None and abs(a-ref['ap'])<1e-12
                checked.append(dict(product=product,version=v,policy=n,confusion_exact=True,sklearn_AP_matches=True))
        print('CPU verified',product,flush=True)
    write(ROOT/'delivery/CPU_VERIFIED.json',dict(created_utc=now(),checks=checked,statistic_rows=len(checked),
        all_native_probability_blocks_reaggregated=True,all_grid_cells_coverage_exactly_one=True,all_masks_common_across_policies=True,
        all_calibrated_logits_sigmoids_and_aggregates_independently_recomputed=True,
        no_GPU_forward=True,original_human_sha256=digest(HUMAN),no_human_fields_modified=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',choices=['freeze','prepare','Source','MsRE','shallow','analyze','verify']);a=p.parse_args()
    if a.step in NAMES[:2]:infer(a.step)
    else:globals()[a.step]()

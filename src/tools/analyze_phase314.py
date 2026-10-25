"""Frozen dual-reference sensitivity: A60 labels, B4 cached models, C12 teacher only."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from datetime import datetime,timezone
import gc
import io
import resource
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
import numpy as np
import analyze_phase304 as legacy
from prepare_phase65_scene_groups import group_products
from phase314_common import *

DEV_EXPECTED={
'S2A_MSIL1C_20180410T214911_N0206_R043_T02LKH_20180410T231102',
'S2A_MSIL1C_20180602T081011_N0206_R078_T36PWC_20180602T115947',
'S2A_MSIL1C_20180703T023551_N0206_R089_T50PLR_20180703T053714',
'S2B_MSIL1C_20180214T025819_N0206_R032_T51TUM_20180214T072411'}
C=('Surface','Cloud','Shadow');INPUTS={};ACCESS=[];START=time.monotonic();MINFREE=10**20;CHECKS=0
def utc():return datetime.now(timezone.utc).isoformat()
def pin(p,expected=None,kind='metadata'):
    p=Path(p);h=sha(p)
    if expected:assert h==expected,str(p)
    INPUTS[str(p)]=h;ACCESS.append(dict(path=str(p),mode=kind,sha256=h,utc=utc()));return h
def obj(p,expected=None):pin(p,expected);return load(p)
def ah(a):return hashlib.sha256(a.tobytes(order='C')).hexdigest()
def guard():
    global MINFREE
    mem={k:int(v.split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())}
    MINFREE=min(MINFREE,mem['MemAvailable']);assert MINFREE>=4000000000
    assert shutil.disk_usage(ROOT).free>=4000000000
    assert time.monotonic()-START<3600
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*1024**3
    assert 4*sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())+30000000<=500000000
def watch():
    while True:
        try:guard()
        except BaseException:write(ROOT/'RESOURCE_FAILURE.json',dict(traceback=traceback.format_exc(),utc=utc()));os._exit(74)
        time.sleep(.5)
def mat(y,p):
    global CHECKS
    m=np.bincount(3*y.astype(np.int64)+p,minlength=9).reshape(3,3)
    independent=np.array([[int(((y==a)&(p==b)).sum()) for b in range(3)] for a in range(3)])
    assert np.array_equal(m,independent);CHECKS+=9;return m
def metrics(y,p,score):
    m=mat(y,p);d=np.diag(m);un=m.sum(0)+m.sum(1)-d;ious=np.divide(d,un,out=np.full(3,np.nan),where=un>0)
    result=dict(support=len(y),confusion=m.tolist(),union_classes=int((un>0).sum()),mIoU_pct=100*float(np.nanmean(ious)) if (un>0).any() else None,accuracy_pct=100*int(d.sum())/len(y) if len(y) else None,Shadow_AP_pct=None,Shadow_recall_pct=100*int(m[2,2])/int(m[2].sum()) if m[2].sum() else None,Shadow_positive=int(m[2].sum()),Shadow_negative=int(m[:2].sum()))
    a=legacy.ap(y==2,score) if len(y) else None;result['Shadow_AP_pct']=None if a is None else 100*a
    for k in range(3):result[C[k]+'_IoU_pct']=100*float(ious[k]) if un[k] else None
    for a in range(3):
        for b in range(3):
            if a!=b:result[C[a]+'->'+C[b]+'_errors']=int(m[a,b])
    return result
def cells(m,prefix,names):
    return [dict(**prefix,**{name:int(i) for name,i in zip(names,index)},count=int(m[index])) for index in np.ndindex(m.shape)]
def disagree(m,prefix):
    total=int(m.sum());same=int(np.trace(m));r=dict(**prefix,support=total,agree=same,disagree=total-same,disagreement_pct=100*(total-same)/total if total else None,A_support=m.sum(1).tolist(),B_support=m.sum(0).tolist())
    for a in range(3):
        r['A_'+C[a]+'_conditional_disagreement']=float((m[a].sum()-m[a,a])/m[a].sum()) if m[a].sum() else None
        r['B_'+C[a]+'_conditional_disagreement']=float((m[:,a].sum()-m[a,a])/m[:,a].sum()) if m[:,a].sum() else None
        for b in range(3):
            if a!=b:r[C[a]+'->'+C[b]]=int(m[a,b])
    return r
def references(z,member,kind):
    raw=z.read(member);h=hashlib.sha256(raw).hexdigest();m=np.load(io.BytesIO(raw),allow_pickle=False)
    assert m.dtype==np.bool_ and m.shape==(1022,1022,3),(member,m.shape,m.dtype)
    n=m.sum(-1);valid=n==1;y=np.full((1022,1022),255,np.uint8);y[valid]=m[valid].argmax(-1)
    support=dict(valid=int(valid.sum()),zero_hot=int((n==0).sum()),multi_hot=int((n>1).sum()))
    ACCESS.append(dict(path=str(BASE/(kind+'.zip'))+'::'+member,mode='authorized dual-reference pixels; labels only',sha256=h,utc=utc()))
    return y,valid,h,support
def cached(p,expected):
    pin(p,expected,'authorized B4 Source/PL scores,pred,shadow,valid only')
    with np.load(p,allow_pickle=False) as z:s=z['scores'];pred=z['pred'];shadow=z['shadow'];valid=z['valid']
    assert s.shape==(3,1022,1022) and pred.shape==valid.shape==shadow.shape==(1022,1022)
    assert valid.dtype==bool and np.array_equal(pred,s.argmax(0))
    shifted=s-s.max(0,keepdims=True);prob=np.exp(shifted);prob/=prob.sum(0,keepdims=True)
    assert np.allclose(shadow,prob[2],rtol=1e-6,atol=1e-7)
    return pred,shadow,valid
def decompose(a,b,s,p,choose,prefix):
    n=int(choose.sum());changed=choose&(s!=p);ea=choose&(a==b);dd=choose&(a!=b)
    r=dict(**prefix,support=n,prediction_changes=int(changed.sum()))
    for name,y in [('A',a),('B',b)]:
        corrected=int((choose&(s!=y)&(p==y)).sum());damaged=int((choose&(s==y)&(p!=y)).sum());r[name+'_corrected']=corrected;r[name+'_damaged']=damaged;r[name+'_net_correct']=corrected-damaged
        assert corrected-damaged==int((choose&(p==y)).sum())-int((choose&(s==y)).sum())
    sa=s==a;sb=s==b;pa=p==a;pb=p==b
    for name,cond in [('A_to_B',sa&pb),('B_to_A',sb&pa),('neither_to_A',~sa&~sb&pa),('neither_to_B',~sa&~sb&pb),('A_to_neither',sa&~pa&~pb),('B_to_neither',sb&~pa&~pb)]:r[name]=int((changed&dd&cond).sum())
    r['E_both_correct_unchanged']=int((ea&sa&pa).sum());r['E_both_wrong_unchanged']=int((ea&~sa&~pa&(s==p)).sum());r['E_both_wrong_changed']=int((ea&~sa&~pa&(s!=p)).sum())
    if not dd.any():assert all(r['A_'+k]==r['B_'+k] for k in ['corrected','damaged','net_correct'])
    assert not (dd&sa&sb).any() and not (dd&pa&pb).any()
    if dd.any():assert sum(r[k] for k in ['A_to_B','B_to_A','neither_to_A','neither_to_B','A_to_neither','B_to_neither'])==int((dd&(s!=p)).sum())
    return r

def main():
    assert sys.platform=='linux' and sys.executable.startswith('/home/scv/miniconda3/envs/cloud-lite-pt210/')
    assert not ROOT.exists(),'do not repeat analysis or overwrite failed evidence';ROOT.mkdir();resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));guard();threading.Thread(target=watch,daemon=True).start()
    write(ROOT/'STATUS.json',dict(status='running',utc=utc()))
    try:
        fit=obj(PL/'FIT_IMAGES.json','1a328d6e5b7d88fea3aa3dda89205d79ab1562c4fac76361fcc4e31219e26ea5')['rows'];dev=obj(PL/'EVAL_IMAGES.json','7bc7fd23b684f9ab8e7072cab119b1412a56971a2592ac791f580375f6e696a4')['rows']
        fmap={r['product']:r for r in fit};dmap={r['product']:r for r in dev}
        tagsfile=BASE/'classification_tags.csv';pin(tagsfile);tagrows=read(tagsfile);tags={r['scene']:r for r in tagrows};assert len(tags)==len(tagrows)==513
        groups=group_products(tagrows,set(),set(),set());gmap={p:g['group_id'] for g in groups for p in g['members']}
        for p,row in {**fmap,**dmap}.items():assert row['group_id']==gmap[p]
        pin(R/'src/tools/prepare_phase65_scene_groups.py');pin(BASE/'record.json')
        for n in ['04','09']:
            path=R/f'src/research_results/phase309_20261025/evidence/pages/README_page{n}.txt';pin(path);shutil.copyfile(path,ROOT/f'README_page{n}.txt')
        archive_main=BASE/'masks.zip';archive_alt=BASE/'alt_masks.zip';pin(archive_main,kind='byte SHA only then authorized60mainmembers');pin(archive_alt,kind='byte SHA only then authorized60altmembers')
        with zipfile.ZipFile(archive_alt) as z:
            names=[n for n in z.namelist() if not n.endswith('/')];assert len(names)==60
            alts={Path(n).name.removeprefix('ALT_').removesuffix('.npy'):n for n in names}
            assert len(alts)==60 and all(n=='alt_masks/ALT_'+p+'.npy' for p,n in alts.items())
        bset=set(alts)&set(dmap);cset=set(alts)&set(fmap);assert bset==DEV_EXPECTED and len(cset)==12
        assert sum(tags[p]['dataset']=='CALIBRATION' for p in alts)==10 and sum(tags[p]['dataset']=='VALIDATION' for p in alts)==50
        assert all(tags[p]['dataset']=='VALIDATION' for p in bset)
        code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase314*') if p.is_file()}
        write(ROOT/'EXECUTION_LOCK.json',dict(utc=utc(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code=code,plan_sha256=sha(R/'src/research_plans/phase314.md'),A=sorted(alts),B=sorted(bset),C=sorted(cset),class_order=C,neural_calls=0,authorization='execute314 plus persistent synchronization; proposal boilerplate does not cancel direct execution',support='A:Jmask; B:Jmask & restored historical common & imagevalid; C:Jmask & original teacher imagevalid & old mainvalid',no_new_predictions=True))
        endpoint=obj(PL/'ENDPOINT_LOCK.json');assert endpoint['all_base_tensors_unchanged'];pin(PL/'final_lora.pth',endpoint['final_sha256'],'weight byte SHA only; no model loading')
        pin(R/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth',SOURCE,'weight byte SHA only; no model loading')
        lock=obj(PL/'EVALUATION_LOCK.json');el={(r['product'],r['arm']):r for r in lock['rows']};ps={r['product']:r for r in obj(PL/'PSEUDO_LOCK.json')['rows']}
        old=Path('/home/scv/shared/phase300_20261020');support={r['product_id']:r for r in obj(old/'EXPECTED_SUPPORT.json')['rows'] if r['pool']=='Catalogue'}
        plan={r['product_id']:r for r in obj(old/'EVALUATION_INPUT_PLAN.json')['rows'] if r['pool']=='Catalogue'}
        fitmask={r['product']:r for r in obj(Path('/home/scv/shared/phase208_20261020/12201_FIT_INPUT_LOCK.json'))['bindings']}
        previous=PL/'analysis/PER_GROUP_TRANSITIONS.csv';pin(previous,kind='historical confusion rows filtered to fixed B4 only');prev=[r for r in read(previous) if r['product'] in bset]
        previous_fit=PL/'analysis/PSEUDO_REFERENCE_SUPPORT.csv';pin(previous_fit,kind='historical teacher support rows filtered to fixed C12 only');prev_fit={r['product']:r for r in read(previous_fit) if r['product'] in cset}
        manifest=[];disagreements=[];ab_cells=[];main_cells=[];metricrows=[];deltas=[];diffs=[];tensors=[];changes=[];enrichment=[];pseudorows=[];pseudocells=[];pseudo_support=[];reproductions=[]
        mats={};prefixes={};parts={r:{s:[] for s in ['Source','PL']} for r in ['A','B','main','alt']};mt={}
        with zipfile.ZipFile(archive_main) as mz,zipfile.ZipFile(archive_alt) as az:
            for pid in sorted(alts):
                guard();tag=tags[pid];assert tag['annotator'] in ('A','B');ma=tag['annotator'];aa='B' if ma=='A' else 'A';mainmember='masks/'+pid+'.npy'
                ym,vm,mh,ms=references(mz,mainmember,'masks');ya,va,altsha,ass=references(az,alts[pid],'alt_masks')
                if pid in bset:assert mh==plan[pid]['group']['mask_member_sha256']
                if pid in cset:assert mh==fitmask[pid]['mask_member_sha256']
                a,b=(ym,ya) if ma=='A' else (ya,ym);joint=vm&va
                am=mat(a[joint],b[joint]);mm=mat(ym[joint],ya[joint]);assert np.array_equal(am,mm if ma=='A' else mm.T)
                prefix=dict(product=pid,group_id=gmap[pid],dataset=tag['dataset'],main_annotator=ma,shadows_marked=tag['shadows_marked'])
                prefixes[pid]=prefix;mats[pid]=am;ab_cells+=cells(am,prefix,['label_A','label_B']);main_cells+=cells(mm,prefix,['main_label','alt_label']);disagreements.append(disagree(am,dict(**prefix,row_type='product',groups=1)))
                row=dict(**prefix,main_member=mainmember,alt_member=alts[pid],main_sha256=mh,alt_sha256=altsha,alt_annotator=aa,annotator_evidence='classification_tags.annotator gives main; README4 otherannotator',pool='fit' if pid in cset else 'development' if pid in bset else 'not_model_scope',range_A=True,range_B=pid in bset,range_C=pid in cset,main_valid=ms['valid'],main_zero_hot=ms['zero_hot'],main_multi_hot=ms['multi_hot'],alt_valid=ass['valid'],alt_zero_hot=ass['zero_hot'],alt_multi_hot=ass['multi_hot'],Jmask=int(joint.sum()),main_only=int((vm&~va).sum()),alt_only=int((va&~vm).sum()),J=None,original_main_support=None,extra_excluded_for_alt_invalid=None,extra_excluded_for_image_or_common=None,support_definition='NA outside B/C; label census only',shadow_completeness='public shadows_marked is not separate proof for alternate annotator; unknown where unspecified')
                if pid in bset:
                    cp=Path('/home/scv/shared/phase205_20261019/infer/R4/scores/Catalogue')/(pid+'.npz');pin(cp,kind='historical support archive byteSHA; only flat_indices,valid decoded')
                    with np.load(cp) as z:ids=z['flat_indices'];common=z['valid']
                    ex=support[pid];assert ah(ids)==ex['flat_indices_sha256'] and ah(common)==ex['common_valid_sha256']
                    original=np.zeros((1022,1022),bool);original.ravel()[ids[common]]=True;assert int(original.sum())==ex['count'] and not (original&~vm).any()
                    source,ss,sv=cached(Path('/home/scv/shared/phase302_20261020/evaluation')/f'Source_{pid}.npz',el[pid,'Source']['sha256'])
                    student,qs,qv=cached(PL/'evaluation'/f'LoRA_{pid}.npz',el[pid,'LoRA']['sha256'])
                    assert not (original&~sv).any() and not (original&~qv).any()
                    J=joint&original&sv;assert not (J&~qv).any();row.update(J=int(J.sum()),original_main_support=int(original.sum()),extra_excluded_for_alt_invalid=int((original&~va).sum()),extra_excluded_for_image_or_common=int((joint&~original).sum()),support_definition='restored Phase300/205 common, covered by both modelvalid, intersect bothonehot; no modelinvalid silently removed')
                    for arm,pred,sc in [('Source',source,ss),('PL',student,qs)]:
                        m0=metrics(ym[original],pred[original],sc[original]);metricrows.append(dict(**prefix,support_kind='original_main',reference='main',reference_alias='historical main; not authorA necessarily',strategy=arm,**m0))
                        expected=np.zeros((3,3),np.int64)
                        for r in prev:
                            if r['product']==pid:expected[C.index(r['truth']),C.index(r['Source_pred'] if arm=='Source' else r['student_pred'])]+=int(r['count'])
                        assert np.array_equal(np.array(m0['confusion']),expected)
                        reproductions.append(dict(product=pid,strategy=arm,support=m0['support'],confusion=m0['confusion'],historical_confusion_exact=True))
                    for ref,y in [('A',a),('B',b),('main',ym),('alt',ya)]:
                        for arm,pred,sc in [('Source',source,ss),('PL',student,qs)]:
                            yy=y[J];pp=pred[J];sr=sc[J];measure=metrics(yy,pp,sr);mt[pid,ref,arm]=measure
                            metricrows.append(dict(**prefix,support_kind='J',reference=ref,reference_alias='author'+(ma if ref=='main' else aa) if ref in ('main','alt') else 'independent naming ofsame2references',strategy=arm,**measure));parts[ref][arm].append((yy,pp,sr))
                    for arm in ['Source','PL']:
                        assert mt[pid,'main',arm]==mt[pid,ma,arm] and mt[pid,'alt',arm]==mt[pid,aa,arm]
                    av=a[J].astype(np.int64);bv=b[J];sp=source[J];qp=student[J]
                    tensor=np.bincount(27*av+9*bv+3*sp+qp,minlength=81).reshape(3,3,3,3)
                    for ai,bi,si,pi in np.ndindex(3,3,3,3):assert tensor[ai,bi,si,pi]==int(((av==ai)&(bv==bi)&(sp==si)&(qp==pi)).sum())
                    assert np.array_equal(tensor.sum((2,3)),mat(a[J],b[J]))
                    for ref,axis in [('A',1),('B',0)]:
                        t=tensor.sum(axis);assert np.array_equal(t.sum(2),mt[pid,ref,'Source']['confusion']) and np.array_equal(t.sum(1),mt[pid,ref,'PL']['confusion'])
                    assert np.array_equal(tensor.sum((0,1)),mat(source[J],student[J]));tensors+=cells(tensor,prefix,['label_A','label_B','Source_pred','PL_pred'])
                    E=J&(a==b);D=J&(a!=b);changed=J&(source!=student)
                    for region,choose in [('E',E),('D',D)]:
                        rr=decompose(a,b,source,student,choose,dict(**prefix,region=region));rr['share_all_model_changes']=rr['prediction_changes']/int(changed.sum()) if changed.any() else None;changes.append(rr)
                    enrichment.append(dict(**prefix,J=int(J.sum()),E=int(E.sum()),D=int(D.sum()),prediction_changes=int(changed.sum()),changes_in_D=int((changed&D).sum()),D_fraction_J=float(D.sum()/J.sum()) if J.any() else None,D_fraction_changes=float((changed&D).sum()/changed.sum()) if changed.any() else None,interpretation='descriptive only; boundary/difficulty/class confounding; no authorstyle inference'))
                if pid in cset:
                    e=ps[pid];sp=Path(e['scores_path']);pin(sp,e['scores_sha256'],'authorized C12 Source fullscores only');s=np.load(sp,allow_pickle=False);assert ah(s)==e['scores_array_sha256']
                    tp=PL/'teacher'/f'{pid}.npz';pin(tp,e['sha256'],'authorized C12 frozen teacher pseudo,pred,maximum,valid only')
                    with np.load(tp) as z:pred=z['pred'];pseudo=z['pseudo'];valid=z['valid'];maximum=z['maximum']
                    assert np.array_equal(pred,s.argmax(0));total=s.sum(0);q=np.divide(s,total[None],out=np.zeros_like(s),where=total[None]>1e-12)
                    assert np.array_equal(maximum,q.max(0));accepted=valid&(total>1e-12)&(maximum>=.9);assert np.array_equal(accepted,pseudo!=255) and np.array_equal(pred[accepted],pseudo[accepted])
                    original=valid&vm;J=joint&original;row.update(J=int(J.sum()),original_main_support=int(original.sum()),extra_excluded_for_alt_invalid=int((original&~va).sum()),extra_excluded_for_image_or_common=int((joint&~valid).sum()),support_definition='original Phase304 fit teacher audit valid & mainonehot; intersect alternateonehot; no historical devR4 restriction on fit')
                    assert int(original.sum())==int(prev_fit[pid]['reference_valid']) and int(accepted.sum())==int(prev_fit[pid]['training_accepted'])
                    pseudo_support.append(dict(**prefix,image_valid=int(valid.sum()),original_main_support=int(original.sum()),J=int(J.sum()),original_training_accepted=int(accepted.sum()),accepted_on_J=int((J&accepted).sum()),rejected_on_J=int((J&~accepted).sum()),E=int((J&(a==b)).sum()),D=int((J&(a!=b)).sum())))
                    for ref,y in [('A',a),('B',b)]:
                        whole=mat(y[J],pred[J]);truthden=whole.sum(1)
                        for zone,zchoose in [('all',J),('E',J&(a==b)),('D',J&(a!=b))]:
                            for region,choose in [('accepted',zchoose&accepted),('rejected',zchoose&~accepted),('all',zchoose)]:
                                m=mat(y[choose],pred[choose]);pre=dict(**prefix,reference=ref,zone=zone,region=region);pseudocells+=cells(m,pre,['truth','teacher_pred']);zoneden=np.bincount(y[zchoose],minlength=3)
                                for k in range(3):
                                    pn=int(m[:,k].sum());tn=int(m[k].sum());pseudorows.append(dict(**pre,category=C[k],support=int(m.sum()),predicted=pn,correct=int(m[k,k]),precision=float(m[k,k]/pn) if pn else None,truth_support=tn,truth_support_all_J=int(truthden[k]),truth_fraction_of_all_J=float(tn/truthden[k]) if truthden[k] else None,truth_support_zone=int(zoneden[k]),truth_fraction_within_zone=float(tn/zoneden[k]) if zoneden[k] else None,precision_denominator='predicted class in this identical region; NA if zero',truth_fraction_definition='classsupport this region / trueclass allJ or zone; accepted gives trueclass acceptancerate'))
                    del s,q,total
                manifest.append(row);gc.collect()
        # Conditional strata: never pool CALIBRATION+VALIDATION as independent validation.
        for dataset in ['CALIBRATION','VALIDATION']:
            for field,values in [('all',['all']),('main_annotator',['A','B']),('shadows_marked',['0','1'])]:
                for value in values:
                    ids=[p for p in alts if tags[p]['dataset']==dataset and (field=='all' or tags[p]['annotator' if field=='main_annotator' else field]==value)]
                    m=sum((mats[p] for p in ids),np.zeros((3,3),np.int64));row=disagree(m,dict(product='STRATUM',group_id='NA',dataset=dataset,main_annotator=value if field=='main_annotator' else 'all',shadows_marked=value if field=='shadows_marked' else 'all',row_type='stratum',groups=len(ids)))
                    row['group_mean_disagreement_pct']=float(np.mean([100*(mats[p].sum()-np.trace(mats[p]))/mats[p].sum() for p in ids if mats[p].sum()])) if any(mats[p].sum() for p in ids) else None;disagreements.append(row)
        # Make schemas uniform, preserving perproduct and group mean vs pooled distinction.
        for row in disagreements:row.setdefault('group_mean_disagreement_pct',row['disagreement_pct'])
        for ref in ['A','B','main','alt']:
            for arm in ['Source','PL']:
                triples=parts[ref][arm];ys=np.concatenate([t[0] for t in triples]);pr=np.concatenate([t[1] for t in triples]);sc=np.concatenate([t[2] for t in triples]);measure=metrics(ys,pr,sc);mt['POOLED4',ref,arm]=measure
                metricrows.append(dict(product='POOLED4',group_id='not independent pixel samples',dataset='VALIDATION',main_annotator='mixed',shadows_marked='mixed',support_kind='J',reference=ref,reference_alias='same2references; main/alt relabel only',strategy=arm,**measure))
                groupmetrics=[mt[p,ref,arm] for p in sorted(bset)];mean={k:float(np.mean([r[k] for r in groupmetrics if r[k] is not None])) if any(r[k] is not None for r in groupmetrics) else None for k in measure if k not in ['confusion']}
                mt['GROUP_MEAN4',ref,arm]=mean
                del ys,pr,sc
        fields=['mIoU_pct','Surface_IoU_pct','Cloud_IoU_pct','Shadow_IoU_pct','accuracy_pct','Shadow_AP_pct','Shadow_recall_pct']+[C[a]+'->'+C[b]+'_errors' for a in range(3) for b in range(3) if a!=b]
        for pid in sorted(bset)+['POOLED4','GROUP_MEAN4']:
            for ref in ['A','B','main','alt']:
                for key in fields:
                    x=mt[pid,ref,'Source'][key];y=mt[pid,ref,'PL'][key];deltas.append(dict(product=pid,reference=ref,metric=key,Source=x,PL=y,delta=None if x is None or y is None else y-x,Source_valid_groups=sum(mt[p,ref,'Source'][key] is not None for p in bset),PL_valid_groups=sum(mt[p,ref,'PL'][key] is not None for p in bset),interpretation='4posthoc development cases; count means/pooled explicitly named'))
            for key in fields:
                da=next(r['delta'] for r in deltas if r['product']==pid and r['reference']=='A' and r['metric']==key);db=next(r['delta'] for r in deltas if r['product']==pid and r['reference']=='B' and r['metric']==key)
                diffs.append(dict(product=pid,metric=key,delta_A=da,delta_B=db,delta_B_minus_delta_A=None if da is None or db is None else db-da,direction_same=None if da is None or db is None else int(np.sign(da))==int(np.sign(db))))
        outputs={'DUAL_LABEL_MANIFEST.csv':manifest,'ANNOTATOR_DISAGREEMENT.csv':disagreements,'ANNOTATOR_CONFUSIONS.csv':ab_cells,'MAIN_ALT_CONFUSIONS.csv':main_cells,'DUAL_REFERENCE_METRICS.csv':metricrows,'DUAL_REFERENCE_DELTAS.csv':deltas,'DELTA_B_MINUS_A.csv':diffs,'TENSOR81.csv':tensors,'CHANGE_DECOMPOSITION.csv':changes,'CHANGE_ENRICHMENT.csv':enrichment,'PSEUDO_LABEL_SENSITIVITY.csv':pseudorows,'PSEUDO_CONFUSIONS.csv':pseudocells,'PSEUDO_SUPPORT.csv':pseudo_support,'ORIGINAL_MAIN_REPRODUCTION.csv':reproductions}
        for n,rows in outputs.items():table(ROOT/n,rows)
        assert 'torch' not in sys.modules;guard()
        write(ROOT/'CPU_ACCEPTANCE.json',dict(status='passed',reference_products=60,calibration=10,validation=50,development_products=sorted(bset),fit_products=sorted(cset),B_groups=4,C_groups=12,main_author_B_products=sum(r['main_annotator']=='B' for r in manifest),original_confusions_exact=len(reproductions),tensor81_cells=len(tensors),onehot_checked_per_reference=True,invalid_never_argmaxed=True,independent_direct_boolean_checks=CHECKS,E_counts_identical=True,alias_checks=True,training_mask_unmodified=True,no_student_fit_predictions_read=True,reference_completeness_unknown_not_inferred=True,no_new_model_or_original_image_read=True,no_torch=True))
        write(ROOT/'STATUS.json',dict(status='statistics_complete_pending_independent_audit',utc=utc(),seconds=time.monotonic()-START,peak_cpu_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,minimum_available_memory=MINFREE,training=0,fits=0,forwards=0,downloads=0))
    except BaseException:
        write(ROOT/'FAILURE.json',dict(utc=utc(),traceback=traceback.format_exc(),action='preserve; no blind repeat'));write(ROOT/'STATUS.json',dict(status='failed',utc=utc()));raise
    finally:write(ROOT/'INPUT_SHA256.json',INPUTS);write(ROOT/'ACCESS.json',ACCESS)
    print(json.dumps(load(ROOT/'CPU_ACCEPTANCE.json')));print(json.dumps(load(ROOT/'STATUS.json')))
if __name__=='__main__':main()

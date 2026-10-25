"""Zero-fitting, matched-scene/raw-TOA descriptive diagnosis on the original 32 groups."""
import gc
import hashlib
import io
import os
import sys
import time
import traceback
import zipfile
import numpy as np
import auc_phase306 as ranking
from phase306_common import *
from analyze_phase304 import regions
from phase302_images import rgb

INPUTS={}
EVENTS=('D','U','K','C_Shadow')
BINS=(('q[0,0.5)',0.,.5),('q[0.5,0.7)',.5,.7),('q[0.7,0.9)',.7,.9))

def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def pin(p):INPUTS[str(p)]=sha(p);return load(p)
def psha(p):INPUTS[str(p)]=sha(p);return INPUTS[str(p)]
def pct(a,b):return 100*a/b if b else None
def add(rows,keys):return {k:sum(r[k] for r in rows) for k in keys}
def event_counts(y,b,area):
    d=area&(y==0)&(b==1);u=area&(y==0)&(b==0);k=area&(y==1)&(b==1)
    o=area&~(d|u|k);c=area&(b==1);cs=c&(y==2)
    assert np.array_equal(d.astype(np.uint8)+u+k+o,area.astype(np.uint8))
    assert np.array_equal(d.astype(np.uint8)+k+cs,c.astype(np.uint8))
    counts=dict(R=int(area.sum()),C=int(c.sum()),D=int(d.sum()),U=int(u.sum()),K=int(k.sum()),O=int(o.sum()),C_Shadow=int(cs.sum()),
        Surface_risk=int((area&(y==0)).sum()),truth_Surface=int((area&(y==0)).sum()),truth_Cloud=int((area&(y==1)).sum()),truth_Shadow=int((area&(y==2)).sum()))
    return counts,dict(D=d,U=u,K=k,O=o,C=c,C_Shadow=cs,R=area)

def distribution(scope,identity,snow,values,event):
    assert values.ndim==2 and values.shape[1]==6
    out=[]
    for j,band in enumerate(BANDS):
        quant=np.quantile(values[:,j],[.01,.10,.50,.90,.99],method='linear').tolist() if len(values) else [None]*5
        out.append(dict(scope=scope,identity=identity,snow_ice=snow,event=event,band=band,count=len(values),**dict(zip(('p1','p10','p50','p90','p99'),quant))))
    return out

def auc_rows(scope,identity,snow,parts):
    if parts:
        vals=np.concatenate([r['values'] for r in parts]);codes=np.concatenate([r['events'] for r in parts]);q=np.concatenate([r['q'] for r in parts])
    else:vals=np.empty((0,6),np.float32);codes=np.empty(0,np.uint8);q=np.empty(0,np.float32)
    rows=[]
    for name,lo,hi in (('all_R_q',0.,.9),)+BINS:
        choose=(q>=np.float32(lo))&(q<np.float32(hi));d=choose&(codes==0);k=choose&(codes==2);cs=choose&(codes==3)
        candidate=d|k;nD=int(d.sum());nK=int(k.sum());y=codes[candidate]==2
        for j,band in enumerate(BANDS):
            value=ranking.auc(y,vals[candidate,j])
            rows.append(dict(scope=scope,identity=identity,snow_ice=snow,q_bin=name,band=band,D=nD,K=nK,C_Shadow_excluded=int(cs.sum()),
                AUC=value,reason='both classes' if value is not None else 'no D or K' if not nD and not nK else 'no D' if not nD else 'no K',
                direction='larger raw TOA ranks true Cloud; never flip'))
    return rows

def main():
    started=time.monotonic();assert 'torch' not in sys.modules
    assert sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
    lock=pin(ROOT/'EXECUTION_LOCK.json')
    for n,h in lock['code'].items():assert sha(R/n)==h,n
    assert sha(R/'src/research_plans/phase306.md')==lock['protocol_sha256']
    tests=ranking.numerical_tests()
    s305=R/'src/research_results/phase305_20261025'
    summary=pin(s305/'SUMMARY_SHA256.json')
    for n,h in summary['files'].items():assert sha(s305/n)==h,n
    for n in ('INPUT_SHA256.json','G_RELIABILITY.csv','BOUNDARY_PER_GROUP.csv','SCENE_MAPPING.csv','MAPPING_LOCK.json'):
        assert sha(P/n)==summary['files'][n],n
    archived=pin(P/'ARCHIVE_RECEIPT.json');assert pin(s305/'BACKUP_RECEIPT.json')['sha256']==archived['sha256']
    assert psha(P/'delivery.tar.gz')==archived['sha256']
    old_inputs=pin(P/'INPUT_SHA256.json')
    endpoint=pin(B/'ENDPOINT_LOCK.json');assert endpoint['status']=='sealed' and endpoint['updates']==955
    assert psha(B/'final_lora.pth')==endpoint['final_sha256']=='2d0d56de44f57d43a5c3f9788c658ce840f6e916fd4cb49797f111682c6108c9'
    source_lock=pin(A/'EVALUATION_LOCK.json');student_lock=pin(B/'EVALUATION_LOCK.json')
    for p in (A/'EVALUATION_LOCK.json',B/'EVALUATION_LOCK.json',B/'ENDPOINT_LOCK.json'):
        assert sha(p)==old_inputs[str(p)],str(p)
    image_rows={r['product']:r for r in pin(B/'EVAL_IMAGES.json')['rows']}
    old=Path('/home/scv/shared/phase300_20261020')
    plan=[r for r in pin(old/'EVALUATION_INPUT_PLAN.json')['rows'] if r['pool']=='Catalogue']
    expected={r['product_id']:r for r in pin(old/'EXPECTED_SUPPORT.json')['rows'] if r['pool']=='Catalogue'}
    assert len(plan)==32 and {r['product_id'] for r in plan}==set(image_rows)
    mapping_lock=pin(P/'MAPPING_LOCK.json');assert psha(Path(mapping_lock['source']))==mapping_lock['sha256']
    assert psha(P/'SCENE_MAPPING.csv')==mapping_lock['rows_sha256']
    mapped=read(P/'SCENE_MAPPING.csv');mapping={(r['product'],r['field']):r['value'] for r in mapped}
    assert len(mapping)==352 and all(v in ('0','1','missing') for v in mapping.values())
    table(ROOT/'SCENE_MAPPING.csv',mapped);write(ROOT/'MAPPING_LOCK.json',mapping_lock)
    previous={r['product']:r for r in read(P/'G_RELIABILITY.csv') if r['product']!='ALL'};psha(P/'G_RELIABILITY.csv')
    previous_regions={(r['product'],r['region']):r for r in read(P/'BOUNDARY_PER_GROUP.csv')};psha(P/'BOUNDARY_PER_GROUP.csv')
    masks_path=R/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip';assert psha(masks_path)==old_inputs[str(masks_path)]
    (ROOT/'samples').mkdir()
    group_rows=[];boundary_rows=[];loss_rows=[];support_rows=[];channel_rows=[];bin_rows=[];samples=[];quantiles=[];aucs=[]
    with zipfile.ZipFile(masks_path) as masks:
        for row in plan:
            pid=row['product_id'];gid=row['group']['group_id'];snow=mapping[pid,'snow/ice'];prefix=dict(product=pid,group_id=gid,snow_ice=snow)
            pa=A/'evaluation'/('Source_'+pid+'.npz');pb=B/'evaluation'/('LoRA_'+pid+'.npz')
            ar=next(r for r in source_lock['rows'] if r['arm']=='Source' and r['product']==pid);br=next(r for r in student_lock['rows'] if r['arm']=='LoRA' and r['product']==pid)
            assert psha(pa)==ar['sha256']==old_inputs[str(pa)] and psha(pb)==br['sha256']==old_inputs[str(pb)]
            with np.load(pa) as z:s=z['scores'];a=z['pred'];image_valid=z['valid']
            with np.load(pb) as z:b=z['pred'];assert np.array_equal(z['valid'],image_valid)
            assert np.array_equal(a,s.argmax(0)) and s.dtype==np.float32 and np.isfinite(s).all() and (s>=0).all()
            sums=s.sum(0);q=np.divide(s[0],sums,out=np.zeros_like(sums),where=sums>1e-12)
            support_path=Path('/home/scv/shared/phase205_20261019/infer/R4/scores/Catalogue')/(pid+'.npz');assert psha(support_path)==old_inputs[str(support_path)]
            with np.load(support_path) as z:ids=z['flat_indices'];common=z['valid']
            ex=expected[pid];assert ah(ids)==ex['flat_indices_sha256'] and ah(common)==ex['common_valid_sha256']
            valid=np.zeros((1022,1022),bool);valid.ravel()[ids[common]]=True
            assert int(valid.sum())==ex['count'] and not (valid&~image_valid).any()
            G=(a==0)&(sums>1e-12)&(q>=np.float32(.9));region=valid&(a==0)&(sums>1e-12)&(q<np.float32(.9))
            raw=masks.read('masks/'+pid+'.npy');assert hashlib.sha256(raw).hexdigest()==row['group']['mask_member_sha256']
            mask=np.load(io.BytesIO(raw));assert (mask.sum(-1)[valid]==1).all();y=mask.argmax(-1).astype(np.uint8)
            dc=valid&(a==0)&(y==0)&(b==1);tiny=valid&(a==0)&(sums<=1e-12)
            prior=previous[pid]
            assert int(dc.sum())==int(prior['D_correct']) and int((dc&G).sum())==int(prior['D_correct_G']) and int((dc&~G).sum())==int(prior['D_correct_nonG'])
            assert int((valid&(a==0)&(y==0)&~G).sum())==int(prior['nonG_correct_Surface'])
            counts,ev=event_counts(y,b,region)
            assert counts['D']+int((tiny&(y==0)&(b==1)).sum())==int(prior['D_correct_nonG'])
            base=dict(**prefix,support=int(valid.sum()),Source_correct_Surface=int((valid&(a==0)&(y==0)).sum()),
                D_correct=int(dc.sum()),G_damage=int((dc&G).sum()),nonG_damage=int((dc&~G).sum()),
                nonG_correct_Surface=int((valid&(a==0)&(y==0)&~G).sum()),tiny_SourceSurface=int(tiny.sum()),tiny_D=int((tiny&(y==0)&(b==1)).sum()),
                common_truth_Surface=int((valid&(y==0)).sum()),common_truth_Cloud=int((valid&(y==1)).sum()),common_truth_Shadow=int((valid&(y==2)).sum()),**counts)
            brs=[]
            for name,where in regions(y,valid).items():
                cs,_=event_counts(y,b,region&where);brs.append(cs)
                assert int((dc&~G&where).sum())==int(previous_regions[pid,name]['D_correct_nonG'])
                boundary_rows.append(dict(**prefix,region=name,common_support=int(where.sum()),**cs))
                base[name+'_D']=cs['D']
            for key in counts:assert sum(r[key] for r in brs)==counts[key]
            bins=[]
            for name,lo,hi in BINS:
                choose=region&(q>=np.float32(lo))&(q<np.float32(hi));cs,_=event_counts(y,b,choose);bins.append(cs)
                bin_rows.append(dict(**prefix,q_bin=name,**cs))
            assert all(sum(v[k] for v in bins)==counts[k] for k in counts)
            image=image_rows[pid]
            with zipfile.ZipFile(image['image_archive']) as archive:cube_raw=archive.read(image['image_member'])
            cube_hash=hashlib.sha256(cube_raw).hexdigest();assert cube_hash==image['image_member_sha256']
            cube=np.load(io.BytesIO(cube_raw));assert cube.shape==(1022,1022,13) and cube.dtype==np.float32
            bands=cube[:,:,list(INDICES)].copy();finite=np.isfinite(bands).all(-1);nonzero=(bands!=0).all(-1);notfill=(bands!=np.float32(65535/10000)).all(-1)
            six_valid=finite&nonzero&notfill;joint=valid&six_valid
            # Identity check against the inherited RGB loader and row/column flat indexing.
            bgr,rgb_valid=rgb(image);assert np.array_equal(rgb_valid,image_valid)
            positions=np.unique(np.r_[np.flatnonzero(valid)[[0,len(np.flatnonzero(valid))//2,-1]],np.flatnonzero(region)[:3]])
            for flat in positions:
                yy,xx=divmod(int(flat),1022)
                assert np.array_equal(bands[yy,xx],cube[yy,xx,list(INDICES)],equal_nan=True)
                assert np.array_equal(bands.reshape(-1,6)[flat],bands[yy,xx],equal_nan=True)
                assert np.array_equal(bgr[[2,1,0],yy,xx],cube[yy,xx,[3,2,1]]*np.float32(255))
                channel_rows.append(dict(**prefix,flat_index=int(flat),row=yy,column=xx,cube_member_sha256=cube_hash,
                    bands=list(BANDS),indices=list(INDICES),TOA=bands[yy,xx].tolist(),RGB_scaled_identity=True,flat_row_column_identity=True))
            del bgr,cube_raw,cube
            original_R=counts['R'];base['six_R']=int((region&joint).sum());base['six_C']=int((ev['C']&joint).sum())
            for event in ('R','C','D','U','K','O','C_Shadow'):
                choose=ev[event];n=int(choose.sum());passed=int((choose&joint).sum())
                reasons=[int((choose&~finite).sum()),int((choose&finite&~nonzero).sum()),int((choose&finite&nonzero&~notfill).sum())]
                assert sum(reasons)==n-passed
                loss_rows.append(dict(**prefix,event=event,original=n,six_band_intersection=passed,lost=n-passed,lost_pct=pct(n-passed,n),
                    nonfinite_first=reasons[0],zero_after_finite=reasons[1],fill_after_finite_nonzero=reasons[2],cube_member_sha256=cube_hash,channel_indices=list(INDICES)))
            chosen=np.zeros(valid.shape,bool);code=np.full(valid.shape,255,np.uint8)
            for i,event in enumerate(EVENTS):choose=ev[event]&joint;chosen|=choose;code[choose]=i
            flat=np.flatnonzero(chosen).astype(np.int32);values=bands.reshape(-1,6)[flat].copy();event_codes=code.ravel()[flat].copy();qvalues=q.ravel()[flat].copy()
            assert len(flat)==sum(int((ev[e]&joint).sum()) for e in EVENTS) and np.isfinite(values).all()
            sample_path=ROOT/'samples'/(pid+'.npz')
            np.savez_compressed(sample_path,flat_indices=flat,event=event_codes,Source_qSurface=qvalues,TOA=values,
                common_packbits=np.packbits(valid),six_common_packbits=np.packbits(joint))
            support_rows.append(dict(**prefix,common=int(valid.sum()),six_common=int(joint.sum()),common_sha256=ah(valid),six_common_sha256=ah(joint),
                common_flat_indices_sha256=ah(np.flatnonzero(valid)),six_flat_indices_sha256=ah(np.flatnonzero(joint)),
                cube_archive=image['image_archive'],cube_member=image['image_member'],cube_member_sha256=cube_hash,
                samples_file=sample_path.relative_to(ROOT).as_posix(),samples_sha256=sha(sample_path),event_codes={e:i for i,e in enumerate(EVENTS)},
                selected_flat_indices_sha256=ah(flat),bands=list(BANDS),indices=list(INDICES),shape=[1022,1022,13]))
            item=dict(product=pid,group_id=gid,snow=snow,values=values,events=event_codes,q=qvalues);samples.append(item)
            for i,event in enumerate(EVENTS):quantiles+=distribution('group',pid,snow,values[event_codes==i],event)
            aucs+=auc_rows('group',pid,snow,[item]);group_rows.append(base)
            print('group',len(group_rows),pid,counts,flush=True)
            del s,mask,raw,bands,ev;gc.collect()
    keys=[k for k in group_rows[0] if k not in ('product','group_id','snow_ice')]
    totals=add(group_rows,keys)
    assert totals['D_correct']==506798 and totals['G_damage']==56740 and totals['nonG_damage']==450058 and totals['nonG_correct_Surface']==2812248
    assert totals['D']+totals['tiny_D']==450058
    snow_rows=[r for r in group_rows if r['snow_ice']=='1']
    assert len(snow_rows)==7 and sum(r['D_correct'] for r in snow_rows)==375385 and sum(r['nonG_damage'] for r in snow_rows)==366352
    table(ROOT/'EVENTS_PER_GROUP.csv',group_rows);table(ROOT/'EVENTS_POOLED.csv',[dict(product='ALL',group_id='ALL',snow_ice='all',**totals)])
    table(ROOT/'BOUNDARY_EVENTS.csv',boundary_rows);table(ROOT/'CONFIDENCE_EVENT_PARTITIONS.csv',bin_rows)
    table(ROOT/'SPECTRAL_SUPPORT_LOSS.csv',loss_rows);table(ROOT/'SUPPORT_AND_SAMPLE_SHA.csv',support_rows);table(ROOT/'CHANNEL_POSITION_CHECKS.csv',channel_rows)
    write(ROOT/'CHANNEL_LOCK.json',dict(bands=list(BANDS),indices=list(INDICES),input='original float32 TOA; no255/no clipping/no standardization/no indices',
        B08_index=7,B8A_not_used=True,all_bands_share_identical_support=True,event_codes={e:i for i,e in enumerate(EVENTS)},
        sampling='joint image-valid, old reference common support, six finite nonzero nonfill bands',
        limitations='Inherited native-resampling provenance limited; these cube statistics do not validate native sensor response'))
    strata=[]
    for field in FIELDS:
        partition=[]
        for value in ('0','1','missing'):
            subset=[r for r in group_rows if mapping[r['product'],field]==value];c=add(subset,keys);partition.append(c)
            both=[r for r in subset if r['D']>0 and r['K']>0]
            strata.append(dict(field=field,value=value,products=len(subset),groups=len({r['group_id'] for r in subset}),
                **c,D_groups=sum(r['D']>0 for r in subset),K_groups=sum(r['K']>0 for r in subset),both_groups=len(both),
                both_group_D=sum(r['D'] for r in both),both_group_K=sum(r['K'] for r in both),
                share_pool_D_correct_pct=pct(c['D_correct'],totals['D_correct']),share_pool_nonG_D_pct=pct(c['D'],totals['D']),
                Surface_damage_rate_pct=pct(c['D'],c['Surface_risk']),interpretation='scene labels only; overlapping fields are not additive'))
        assert add(partition,keys)==totals
    table(ROOT/'SCENE_EVENT_STRATA.csv',strata)
    ranked=sorted(group_rows,key=lambda r:(-r['D'],r['product']));top=[]
    for n in (1,3,5):
        subset=ranked[:n];c=add(subset,keys)
        top.append(dict(top_n=n,products=[r['product'] for r in subset],scene_labels=[{f:mapping[r['product'],f] for f in FIELDS} for r in subset],
            D=c['D'],D_pool_share_pct=pct(c['D'],totals['D']),Surface_risk=c['Surface_risk'],Surface_damage_rate_pct=pct(c['D'],c['Surface_risk']),
            K=c['K'],selection='descriptive rank only; never select/drop training/evaluation groups'))
    table(ROOT/'TOP_1_3_5_CONCENTRATION.csv',top)
    for scope,identity,snow,subset in [('pooled','ALL','all',samples)]+[('snow_ice',v,v,[s for s in samples if s['snow']==v]) for v in ('0','1','missing')]:
        for i,event in enumerate(EVENTS):
            values=np.concatenate([s['values'][s['events']==i] for s in subset]) if subset else np.empty((0,6),np.float32)
            quantiles+=distribution(scope,identity,snow,values,event)
        aucs+=auc_rows(scope,identity,snow,subset)
    table(ROOT/'BAND_QUANTILES.csv',quantiles)
    group_auc=[r for r in aucs if r['scope']=='group'];means=[];evidence=[]
    for name in ('all_R_q',)+tuple(b[0] for b in BINS):
        for band in BANDS:
            valid_auc=[r for r in group_auc if r['q_bin']==name and r['band']==band and r['AUC'] is not None]
            means.append(dict(scope='group_mean',identity='ALL',snow_ice='all',q_bin=name,band=band,
                D=sum(r['D'] for r in valid_auc),K=sum(r['K'] for r in valid_auc),C_Shadow_excluded=sum(r['C_Shadow_excluded'] for r in group_auc if r['q_bin']==name and r['band']==band),
                AUC=sum(r['AUC'] for r in valid_auc)/len(valid_auc) if valid_auc else None,reason=f'{len(valid_auc)} computable independent groups; unweighted mean',direction='larger raw TOA ranks true Cloud; never flip'))
    for band in BANDS:
        base=[r for r in group_auc if r['q_bin']=='all_R_q' and r['band']==band and r['AUC'] is not None]
        fine=[r for r in group_auc if r['q_bin']!='all_R_q' and r['band']==band and r['AUC'] is not None]
        match={r['identity']:r for r in base};same=sum(np.sign(r['AUC']-.5)==np.sign(match[r['identity']]['AUC']-.5) for r in fine)
        evidence.append(dict(band=band,type='RGB' if band in BANDS[:3] else 'additional raw band',computable_groups=len(base),
            covered_D=sum(r['D'] for r in base),covered_K=sum(r['K'] for r in base),
            covered_spectral_D_pct=pct(sum(r['D'] for r in base),sum(r['D'] for r in group_auc if r['q_bin']=='all_R_q' and r['band']==band)),
            covered_spectral_K_pct=pct(sum(r['K'] for r in base),sum(r['K'] for r in group_auc if r['q_bin']=='all_R_q' and r['band']==band)),
            fine_q_computable_group_bins=len(fine),fine_q_independent_groups=len({r['identity'] for r in fine}),fine_q_same_direction_as_group=same,
            group_AUC_above_half=sum(r['AUC']>.5 for r in base),group_AUC_below_half=sum(r['AUC']<.5 for r in base),
            cannot_prove='No conditional increment over RGB; no joint model or held-out group generalization',next_test='Only matched-capacity/group-held-out comparison could establish joint information gain'))
    table(ROOT/'AUC_RESULTS.csv',aucs+means);table(ROOT/'EVIDENCE_COVERAGE.csv',evidence)
    table(ROOT/'AUC_GROUP_MEANS.csv',means)
    write(ROOT/'INPUT_SHA256.json',INPUTS)
    write(ROOT/'CPU_ACCEPTANCE.json',dict(status='passed',products=32,neural_calls=0,training_updates=0,classifier_fits=0,downloads=0,
        previous_damage_exact=True,snow_seven_scenes_exact=True,events_partition_R_exact=True,candidate_D_K_Shadow_partition_exact=True,
        confidence_bins_exact=True,all_eleven_scene_partitions_exact=True,same_six_band_support=True,common_and_spectral_support_SHA=True,
        channel_and_flat_index_checked=True,auc_group_ties_vs_independent_rank_sum=True,auc_numerical_tests=tests,
        independent_auc_checks=ranking.CHECKS,missing_classes_NA=True,unflipped_all_six_bands=True,torch_imported='torch' in sys.modules,
        seconds=time.monotonic()-started,PID=os.getpid()))
    report(totals,strata,top,aucs,means,evidence)
    print('Phase306 CPU acceptance passed; zero fits/forwards',flush=True)

def report(t,strata,top,aucs,means,evidence):
    def f(v):return 'NA' if v is None else f'{v:.3f}'
    snow=next(r for r in strata if r['field']=='snow/ice' and r['value']=='1')
    lines=['# Phase306：非G损伤集中度与原始六波段证据','','## 数据结果','',
        '原Catalogue32组、Phase302 Source及Phase304 Q90封存缓存，Phase305场景映射/共同支持不变；gzs cloud-lite-pt210，零训练/拟合/模型前向/下载。R仅Source原判Surface且0≤qSurface<0.90、sum>1e-12；真值仅离线定义D/U/K/O。',
        f"复现原正确Surface损伤506798=G内56740+非G450058；非G正确Surface支持2812248。R支持{t['R']:,}，D/U/K/O={t['D']}/{t['U']}/{t['K']}/{t['O']}，C内Shadow={t['C_Shadow']}；tiny-sum SourceSurface={t['tiny_SourceSurface']}，其中损伤{t['tiny_D']}，未静默删除。",
        f"snow/ice=1的7场景原正确Surface损伤375385，占全池{f(pct(375385,506798))}%；非G损伤366352，占非G D {f(pct(366352,t['D']))}%。这是场景关联，不是雪冰像元因果。",'',
        '|D排序前N组|D数/全池占比%|原正确Surface风险支持/损伤率%|K数|','|---|---|---|---|']
    for r in top:lines.append(f"|{r['top_n']}|{r['D']}/{f(r['D_pool_share_pct'])}|{r['Surface_risk']}/{f(r['Surface_damage_rate_pct'])}|{r['K']}|")
    lines+=['','|snow/ice层|组数|有D/有K/同时有D,K组数|同时组的D/K|','|---|---|---|---|']
    for r in (r for r in strata if r['field']=='snow/ice'):lines.append(f"|{r['value']}|{r['groups']}|{r['D_groups']}/{r['K_groups']}/{r['both_groups']}|{r['both_group_D']}/{r['both_group_K']}|")
    lines+=['', '六波段B04/B03/B02/B08/B11/B12，原cube索引[3,2,1,7,11,12]，float32 TOA不乘255、不clip、不标准化、不加指数。全部光谱分析使用同一六波段有限/非零/非填充值交集；原支持与光谱支持、逐事件损失和group+flat-index原始取值NPZ另交，不回写原损伤比例。','',
        '|波段|同组D/K可算组数|覆盖光谱D/K %|同组均/池化AUC|同q可算组/组层|同q方向与整组一致层数|','|---|---|---|---|---|---|']
    for e in evidence:
        band=e['band'];m=next(r for r in means if r['band']==band and r['q_bin']=='all_R_q');p=next(r for r in aucs if r['scope']=='pooled' and r['band']==band and r['q_bin']=='all_R_q')
        lines.append(f"|{band}|{e['computable_groups']}|{f(e['covered_spectral_D_pct'])}/{f(e['covered_spectral_K_pct'])}|{f(m['AUC'])}/{f(p['AUC'])}|{e['fine_q_independent_groups']}/{e['fine_q_computable_group_bins']}|{e['fine_q_same_direction_as_group']}|")
    lines+=['', 'Cloud为正，固定较大原始TOA排Cloud；AUC<0.5原样保留，不取反/选赢家。组均为可算组无权平均，单类NA；池化不替代组内。全部32×6波段×固定q层、缺类/Shadow排除数及D/U/K/Shadow精确p1/p10/p50/p90/p99均另表。','', '## 结论分析','',
        f"非G损伤在前1/3/5组的累计占比为"+'/'.join(f(r['D_pool_share_pct']) for r in top)+'%，应结合各组风险支持解读。snow/ice场景层内D/K共同支持范围有限时，池化差异可能由场景混杂主导；不能用像元数替代独立场景数。',
        '六个单变量AUC只描述原始观测的排序；同组及相近q层的完整数表提供有限线索，不能证明B11/B12在RGB条件下有独立增量。若继续，只能用独立立项的匹配容量与组留出信息对照验证联合收益，本轮不拟合、不加波段/阈值/模块、不启动下一phase。',
        '这些TOA取值来自既有重采样cube，原生重采样溯源仍有限；不是原生传感器响应验证。snow/ice等为场景标签，参考边界仅半径2标签邻接。历史development反复使用、单seed与Source/DINO预训练交集未知；不称统计显著、物理因果或新的SFDA方法有效。']
    (ROOT/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    try:main()
    except BaseException:
        write(ROOT/'FAILURE.json',dict(traceback=traceback.format_exc(),neural_calls=0,classifier_fits=0));raise

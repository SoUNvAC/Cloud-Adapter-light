"""Streaming, zero-neural-call high-confidence Surface damage and fixed readout audit."""
import gc
import hashlib
import io
import os
import sys
import time
import traceback
import zipfile
import numpy as np
from phase305_common import *
from analyze_phase304 import tensor,cells,events,regions,confusion_matrix,ious

INPUTS={}
def pin(p):INPUTS[str(p)]=sha(p);return load(p)
def psha(p):INPUTS[str(p)]=sha(p);return INPUTS[str(p)]
def ratio(a,b):return 100*a/b if b else None
def classify(y,a,b,g,v,total):
    # Every field is additive across disjoint products/regions, permitting independent partitions.
    dc=(y==0)&(a==0)&(b==1);other=(y==0)&(a==2)&(b==1)
    values=dict(support=int(v.sum()),G=int((v&g).sum()),H=int((v&g&(y==0)).sum()),
        Source_Surface=int((v&(a==0)).sum()),nonG_Source_Surface=int((v&~g&(a==0)).sum()),
        nonG_correct_Surface=int((v&~g&(a==0)&(y==0)).sum()),
        D_correct=int((v&dc).sum()),D_correct_G=int((v&dc&g).sum()),D_correct_nonG=int((v&dc&~g).sum()),
        D_other=int((v&other).sum()),zero_or_tiny_sum=int((v&(total<=1e-12)).sum()))
    for c in range(3):
        values['G_truth_'+C[c]]=int((v&g&(y==c)).sum())
        values['G_student_Cloud_truth_'+C[c]]=int((v&g&(b==1)&(y==c)).sum())
        values['G_Source_wrong_student_correct_truth_'+C[c]]=int((v&g&(y==c)&(a!=y)&(b==y)).sum())
    assert values['D_correct_G']+values['D_correct_nonG']==values['D_correct']
    assert sum(values['G_truth_'+c] for c in C)==values['G'] and values['G_truth_Surface']==values['H']
    return values

def decorate(v):return dict(v,G_Surface_precision_pct=ratio(v['H'],v['G']),
    damage_concentration_in_G_pct=ratio(v['D_correct_G'],v['D_correct']),
    G_correct_Surface_damage_pct=ratio(v['D_correct_G'],v['H']),
    nonG_correct_Surface_damage_pct=ratio(v['D_correct_nonG'],v['nonG_correct_Surface']))
def add(rows):return {k:sum(r[k] for r in rows) for k in rows[0]}
def shadow(mat):
    tp=int(mat[2,2]);fn=int(mat[2,:2].sum());fp=int(mat[:2,2].sum());tn=int(mat[:2,:2].sum())
    return dict(Shadow_TP=tp,Shadow_FP=fp,Shadow_FN=fn,Shadow_TN=tn,recall_pct=ratio(tp,tp+fn),FPR_pct=ratio(fp,fp+tn))
def metrics(mat):
    values=ious(mat)
    return dict(mIoU_pct=float(np.nanmean(values)*100),**{C[c]+'_IoU_pct':float(values[c]*100) if np.isfinite(values[c]) else None for c in range(3)},**shadow(mat))

def main():
    started=time.monotonic();assert 'torch' not in sys.modules
    assert sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
    lock=pin(ROOT/'EXECUTION_LOCK.json')
    for n,h in lock['code'].items():assert sha(R/n)==h,n
    assert sha(R/'src/research_plans/phase305.md')==lock['protocol_sha256']
    old_summary=R/'src/research_results/phase304_20261024'
    summary=pin(old_summary/'SUMMARY_SHA256.json')
    for name,h in summary['files'].items():assert sha(old_summary/name)==h,name
    endpoint=pin(B/'ENDPOINT_LOCK.json');assert endpoint['status']=='sealed' and endpoint['updates']==955
    assert psha(B/'final_lora.pth')==endpoint['final_sha256']=='2d0d56de44f57d43a5c3f9788c658ce840f6e916fd4cb49797f111682c6108c9'
    assert psha(B/'EXECUTION_LOCK.json')==endpoint['execution_lock_sha256']
    source=R/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'
    assert psha(source)==endpoint['source_sha256']=='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
    backup=pin(B/'BACKUP_FILES.json');receipt=pin(B/'BACKUP_STREAM_RECEIPT.json')
    verified=pin(old_summary/'BACKUP_RECEIPT.json');assert verified['all_stream_members_verified'] and receipt['stream_sha256']==verified['stream_sha256']
    assert sha(B/'BACKUP_FILES.json')==receipt['manifest_sha256']
    cached_inputs=pin(B/'analysis/INPUT_SHA256.json');source_lock=pin(A/'EVALUATION_LOCK.json');student_lock=pin(B/'EVALUATION_LOCK.json')
    psha(A/'ENDPOINT_LOCK.json');pin(A/'SOURCE_ADMISSION.json')
    assert psha(B/'EVAL_IMAGES.json')==psha(A/'EVAL_IMAGES.json')
    assert sha(B/'EVALUATION_LOCK.json')==backup['EVALUATION_LOCK.json']['sha256']
    images={r['product']:r for r in pin(B/'EVAL_IMAGES.json')['rows']}
    old=Path('/home/scv/shared/phase300_20261020')
    plan=[r for r in pin(old/'EVALUATION_INPUT_PLAN.json')['rows'] if r['pool']=='Catalogue']
    expected={r['product_id']:r for r in pin(old/'EXPECTED_SUPPORT.json')['rows'] if r['pool']=='Catalogue'}
    assert len(plan)==len(images)==32
    fields_path=R/'data/sentinel2_cloud_mask_catalogue_4172871/classification_tags.csv';psha(fields_path)
    metadata=read(fields_path);assert all(f in metadata[0] for f in ('scene',)+FIELDS)
    mapping=[];by_pid={}
    for row in plan:
        pid=row['product_id'];matches=[r for r in metadata if r['scene'].strip()==pid];entry={}
        for field in FIELDS:
            raw=matches[0][field].strip() if len(matches)==1 else None
            value=raw if raw in ('0','1') else 'missing'
            reason='exact binary field' if value!='missing' else 'duplicate scene rows' if len(matches)>1 else 'missing scene row' if not matches else 'missing/nonbinary field'
            mapping.append(dict(product=pid,field=field,value=value,reason=reason,matched_rows=len(matches),raw=raw))
            entry[field]=value
        by_pid[pid]=entry
    table(ROOT/'SCENE_MAPPING.csv',mapping)
    write(ROOT/'MAPPING_LOCK.json',dict(source=str(fields_path),sha256=sha(fields_path),join='exact complete scene product ID',
        fields=list(FIELDS),values=['0','1','missing'],duplicate_policy='all fields missing if scene has !=1 matching rows',
        rows_sha256=sha(ROOT/'SCENE_MAPPING.csv'),overlapping_fields_not_additive=True,physical_pixel_landcover=False))
    pooled=np.zeros((3,3,3),np.int64);groups=[];region_rows=[];changed=[];transition=[];group_events=[];metric_rows=[];confusions=[];G_locks=[]
    mats={k:np.zeros((3,3),np.int64) for k in ('Source','Q90','Q90+G90')}
    old_per={(r['strategy'],r['product']):r for r in read(B/'analysis/PER_GROUP_METRICS.csv')}
    old_trans={(r['product'],r['truth'],r['Source_pred'],r['student_pred']):int(r['count']) for r in read(B/'analysis/PER_GROUP_TRANSITIONS.csv')}
    old_errors={(r['product'],r['error']):r for r in read(B/'analysis/PER_GROUP_ERRORS.csv')}
    for n in ('PER_GROUP_METRICS.csv','PER_GROUP_TRANSITIONS.csv','PER_GROUP_ERRORS.csv','TRANSITIONS27.csv','ERROR_EVENTS.csv','FIXED_RESULTS.csv'):psha(B/'analysis'/n)
    masks_path=R/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip';assert psha(masks_path)==cached_inputs[str(masks_path)]
    with zipfile.ZipFile(masks_path) as masks:
        for row in plan:
            pid=row['product_id'];gid=row['group']['group_id'];prefix=dict(product=pid,group_id=gid)
            patha=A/'evaluation'/('Source_'+pid+'.npz');pathb=B/'evaluation'/('LoRA_'+pid+'.npz')
            ar=next(r for r in source_lock['rows'] if r['arm']=='Source' and r['product']==pid)
            br=next(r for r in student_lock['rows'] if r['arm']=='LoRA' and r['product']==pid)
            assert psha(patha)==ar['sha256']==cached_inputs[str(patha)]
            assert psha(pathb)==br['sha256']==cached_inputs[str(pathb)]==backup[pathb.relative_to(B).as_posix()]['sha256']
            with np.load(patha) as z:s=z['scores'];a=z['pred'];va=z['valid'];sa=z['shadow']
            with np.load(pathb) as z:sb=z['scores'];b=z['pred'];vb=z['valid'];pb=z['shadow']
            assert np.array_equal(a,s.argmax(0)) and np.array_equal(b,sb.argmax(0))
            assert np.isfinite(s).all() and (s>=0).all() and s.dtype==np.float32
            assert np.array_equal(va,vb)
            total=s.sum(0);q=np.divide(s[0],total,out=np.zeros_like(total),where=total>1e-12)
            image_valid=va&vb
            g=image_valid&(a==0)&(total>1e-12)&(q>=np.float32(.90))
            # G is computed and hashed before any reference for this product is read.
            G_locks.append(dict(**prefix,Source_scores_sha256=sha(patha),G_mask_sha256=hashlib.sha256(g.tobytes()).hexdigest(),
                image_valid=int(image_valid.sum()),G_image_valid=int(g.sum()),tiny_sum_image_valid=int((image_valid&(total<=1e-12)).sum()),
                truth_used_for_G=False,rule='Source argmax=Surface, sSurface/sum(s)>=0.90, sum>1e-12, imagevalid'))
            support_path=Path('/home/scv/shared/phase205_20261019/infer/R4/scores/Catalogue')/(pid+'.npz');psha(support_path)
            with np.load(support_path) as z:ids=z['flat_indices'];common=z['valid']
            ex=expected[pid];assert hashlib.sha256(ids.tobytes()).hexdigest()==ex['flat_indices_sha256']
            assert hashlib.sha256(common.tobytes()).hexdigest()==ex['common_valid_sha256']
            valid=np.zeros((1022,1022),bool);valid.ravel()[ids[common]]=True
            assert int(valid.sum())==ex['count'] and not (valid&~image_valid).any()
            raw=masks.read('masks/'+pid+'.npy');assert hashlib.sha256(raw).hexdigest()==row['group']['mask_member_sha256']
            mask=np.load(io.BytesIO(raw));assert (mask.sum(-1)[valid]==1).all();y=mask.argmax(-1).astype(np.uint8)
            G_locks[-1].update(common_support=int(valid.sum()),imagevalid_outside_reference_support=int((image_valid&~valid).sum()))
            t=tensor(y[valid],a[valid],b[valid]);pooled+=t
            for cell in cells(t,prefix):assert cell['count']==old_trans[pid,cell['truth'],cell['Source_pred'],cell['student_pred']];transition.append(cell)
            for event in events(t,prefix):
                ref=old_errors[pid,event['error']]
                for k in ('truth_support','Source_errors','student_errors','removed','added','net_reduction','corrected','changed_to_other_error','from_correct'):assert event[k]==int(ref[k])
                group_events.append(event)
            stats=classify(y,a,b,g,valid,total);groups.append(dict(**prefix,**stats))
            regions_map=regions(y,valid);parts=[]
            for name,v in regions_map.items():
                st=classify(y,a,b,g,v,total);parts.append(st);region_rows.append(dict(**prefix,region=name,**st))
            assert add(parts)==stats
            switch=valid&g&(b==1);new=b.copy();new[g&(b==1)]=0
            assert np.array_equal(new[~(g&(b==1))],b[~(g&(b==1))]) and np.array_equal(new==2,b==2)
            assert np.array_equal(new[image_valid]!=b[image_valid],(g&(b==1))[image_valid])
            delta_t=tensor(y[valid],b[valid],new[valid]);oldmat=delta_t.sum(2);newmat=delta_t.sum(1)
            switch_counts=[int((switch&(y==c)).sum()) for c in range(3)]
            assert switch_counts==[int(delta_t[c,1,0]) for c in range(3)]
            assert all(int(delta_t[c,i,j])==0 for c in range(3) for i in range(3) for j in range(3) if i!=j and (i,j)!=(1,0))
            assert shadow(oldmat)==shadow(newmat)
            changed.append(dict(**prefix,changed=sum(switch_counts),repaired_true_Surface=switch_counts[0],revoked_true_Cloud_correction=switch_counts[1],
                true_Shadow_Cloud_to_Surface=switch_counts[2],other_positions_unchanged=True,Shadow_mask_unchanged=True,
                Shadow_scores_sha256=hashlib.sha256(pb.tobytes()).hexdigest(),Shadow_scores_unchanged=True))
            for arm,p,refarm,score in (('Source',a,'Source',sa),('Q90',b,'LoRA',pb),('Q90+G90',new,'LoRA',pb)):
                mat=confusion_matrix(y[valid],p[valid],labels=[0,1,2]);ind=np.bincount((3*y[valid]+p[valid]).astype(np.int64),minlength=9).reshape(3,3)
                assert np.array_equal(mat,ind);mats[arm]+=mat;m=metrics(mat);ref=old_per[refarm,pid]
                if arm!='Q90+G90':
                    assert mat.tolist()==json.loads(ref['confusion'])
                    for k in ('mIoU_pct','Surface_IoU_pct','Cloud_IoU_pct','Shadow_IoU_pct','recall_pct','FPR_pct'):
                        assert (ref[k]=='NA' and m[k] is None) or (m[k] is not None and abs(m[k]-float(ref[k]))<1e-10),(pid,k)
                else:
                    for k in ('Shadow_IoU_pct','recall_pct','FPR_pct'):
                        assert (ref[k]=='NA' and m[k] is None) or (m[k] is not None and abs(m[k]-float(ref[k]))<1e-10)
                metric_rows.append(dict(strategy=arm,**prefix,**m,Shadow_AP_pct=ref['AP_pct'],Shadow_AP_reference='identical cached continuous scores/support; no new ranking'))
                for c in range(3):
                    for j in range(3):confusions.append(dict(strategy=arm,**prefix,truth=C[c],pred=C[j],count=int(mat[c,j])))
            print('product',len(groups),pid,stats['D_correct_G'],switch_counts,flush=True)
            del s,sb,mask,raw;gc.collect()
    totals=add([{k:v for k,v in r.items() if k not in ('product','group_id')} for r in groups])
    assert totals['D_correct']==506798 and totals['D_other']==19553 and totals['D_correct']+totals['D_other']==526351
    assert totals['D_correct_G']==sum(r['repaired_true_Surface'] for r in changed)
    pooled_cells=cells(pooled,{})
    assert [(r['truth'],r['Source_pred'],r['student_pred'],r['count']) for r in pooled_cells]==[(r['truth'],r['Source_pred'],r['student_pred'],int(r['count'])) for r in read(B/'analysis/TRANSITIONS27.csv')]
    for e,old_e in zip(events(pooled,{}),read(B/'analysis/ERROR_EVENTS.csv')):
        for k in ('Source_errors','student_errors','removed','added','net_reduction','corrected','changed_to_other_error','from_correct'):assert e[k]==int(old_e[k])
    table(ROOT/'TRANSITIONS27.csv',pooled_cells);table(ROOT/'PER_GROUP_TRANSITIONS.csv',transition)
    table(ROOT/'ERROR_EVENTS.csv',events(pooled,{}));table(ROOT/'PER_GROUP_ERRORS.csv',group_events)
    table(ROOT/'G_IMAGE_ONLY_LOCK.csv',G_locks)
    detailed=[dict(product='ALL',group_id='ALL',**totals)]+groups
    table(ROOT/'G_RELIABILITY.csv',[dict(product=r['product'],group_id=r['group_id'],**decorate({k:v for k,v in r.items() if k not in ('product','group_id')})) for r in detailed])
    damage=[]
    for r in detailed:
        for layer in ('G','nonG'):
            n=r['D_correct_G' if layer=='G' else 'D_correct_nonG'];h=r['H' if layer=='G' else 'nonG_correct_Surface']
            damage.append(dict(product=r['product'],group_id=r['group_id'],layer=layer,Source_predicted_Surface=r['G' if layer=='G' else 'nonG_Source_Surface'],
                Source_correct_Surface=h,damaged=n,share_of_group_D_correct_pct=ratio(n,r['D_correct']),share_of_pool_D_correct_pct=ratio(n,totals['D_correct']),
                damage_rate_pct=ratio(n,h),D_other=r['D_other'] if layer=='nonG' else 0))
    table(ROOT/'DAMAGE_G_NON_G.csv',damage)
    table(ROOT/'BOUNDARY_PER_GROUP.csv',[dict(product=r['product'],group_id=r['group_id'],region=r['region'],**decorate({k:v for k,v in r.items() if k not in ('product','group_id','region')})) for r in region_rows])
    boundary=[]
    for name in ('boundary','nonboundary','excluded'):
        st=add([{k:v for k,v in r.items() if k not in ('product','group_id','region')} for r in region_rows if r['region']==name]);boundary.append(dict(region=name,**decorate(st),share_of_pool_D_correct_G_pct=ratio(st['D_correct_G'],totals['D_correct_G'])))
    assert add([{k:r[k] for k in totals} for r in boundary])==totals
    table(ROOT/'BOUNDARY_POOLED.csv',boundary)
    scene=[]
    for field in FIELDS:
        parts=[]
        for value in ('0','1','missing'):
            selected=[r for r in groups if by_pid[r['product']][field]==value]
            st=add([{k:r[k] for k in totals} for r in selected]) if selected else {k:0 for k in totals};parts.append(st)
            scene.append(dict(field=field,value=value,products=len(selected),groups=len({r['group_id'] for r in selected}),**decorate(st),
                share_of_pool_D_correct_G_pct=ratio(st['D_correct_G'],totals['D_correct_G']),interpretation='scene tag; not pixel landcover'))
        assert add(parts)==totals
    table(ROOT/'SCENE_STRATA.csv',scene);table(ROOT/'PER_GROUP_METRICS.csv',metric_rows)
    table(ROOT/'PER_GROUP_CONFUSIONS.csv',confusions)
    pooled_changes={k:sum(r[k] for r in changed) for k in ('changed','repaired_true_Surface','revoked_true_Cloud_correction','true_Shadow_Cloud_to_Surface')}
    table(ROOT/'READOUT_CHANGES.csv',[dict(product='ALL',group_id='ALL',**pooled_changes,other_positions_unchanged=True,Shadow_mask_unchanged=True,Shadow_scores_sha256='per-group hashes below',Shadow_scores_unchanged=True)]+changed)
    fixed={r['strategy']:r for r in read(B/'analysis/FIXED_RESULTS.csv')};summary_rows=[]
    for arm in mats:
        m=mats[arm];parts=[r for r in metric_rows if r['strategy']==arm];ref=fixed['Source' if arm=='Source' else 'LoRA']
        summary_rows.append(dict(strategy=arm,groups=32,group_mean_mIoU_pct=sum(r['mIoU_pct'] for r in parts)/32,
            pooled_mIoU_pct=metrics(m)['mIoU_pct'],**{k:metrics(m)[k] for k in ('Surface_IoU_pct','Cloud_IoU_pct','Shadow_IoU_pct','Shadow_TP','Shadow_FP','Shadow_FN','Shadow_TN')},
            Surface_to_Cloud=int(m[0,1]),Cloud_to_Surface=int(m[1,0]),macro_recall_pct=float(ref['macro_recall_pct']),
            macro_FPR_pct=float(ref['macro_FPR_pct']),macro_AP_pct=float(ref['macro_AP_pct']),pooled_AP_pct=float(ref['AP_pct']),AP_groups=int(ref['AP_groups'])))
    for i,refarm in ((0,'Source'),(1,'LoRA')):
        assert abs(summary_rows[i]['group_mean_mIoU_pct']-float(fixed[refarm]['group_mean_mIoU_pct']))<1e-10
    assert shadow(mats['Q90'])==shadow(mats['Q90+G90'])
    expected_delta=np.zeros((3,3),np.int64)
    for c,key in enumerate(('repaired_true_Surface','revoked_true_Cloud_correction','true_Shadow_Cloud_to_Surface')):
        expected_delta[c,0]=pooled_changes[key];expected_delta[c,1]=-pooled_changes[key]
    assert np.array_equal(mats['Q90+G90']-mats['Q90'],expected_delta)
    table(ROOT/'FIXED_READOUT_RESULTS.csv',summary_rows)
    table(ROOT/'READOUT_DIFFERENCES.csv',[dict(strategy='Q90+G90 minus Q90',**{k:summary_rows[2][k]-summary_rows[1][k] for k in summary_rows[1] if k not in ('strategy','groups','AP_groups')})])
    table(ROOT/'POOLED_CONFUSIONS.csv',[dict(strategy=arm,truth=C[c],pred=C[j],count=int(m[c,j])) for arm,m in mats.items() for c in range(3) for j in range(3)])
    write(ROOT/'INPUT_SHA256.json',INPUTS)
    write(ROOT/'CPU_ACCEPTANCE.json',dict(status='passed',neural_calls=0,training_updates=0,products=32,all_historical_groups_retained=True,
        G_label_free=True,all27_cells_and_directed_events_reproduced=True,G_nonG_partition_exact=True,boundary_partition_exact=True,
        eleven_scene_partitions_exact=True,independent_bool_vs_tensor_checks=True,readout_only_G_student_Cloud=True,
        Shadow_decisions_and_continuous_scores_unchanged=True,confusion_delta_matches_changes=True,missing_class_rules_unchanged=True,
        seconds=time.monotonic()-started,PID=os.getpid(),torch_imported='torch' in sys.modules))
    report(totals,pooled_changes,summary_rows,boundary,scene)
    print('CPU acceptance passed; zero forwards',flush=True)

def report(t,change,summary,boundary,scene):
    def f(v):return 'NA' if v is None else f'{v:.3f}'
    lines=['# Phase305：高置信Surface损伤与固定读出','','## 数据结果','',
        '固定Phase302 Source与Phase304 Q90末步，原Catalogue32组共同支持；全部缓存及参考输入SHA核验。零训练、零前向。G仅用Source预测与sSurface/sum(s)≥0.90；H=G且参考Surface仅用于离线审计。','',
        '|层|原正确Surface支持|Source正确Surface→学生Cloud|占全部506798损伤%|对应损伤率%|','|---|---:|---:|---:|---:|',
        f"|G|{t['H']}|{t['D_correct_G']}|{f(ratio(t['D_correct_G'],t['D_correct']))}|{f(ratio(t['D_correct_G'],t['H']))}|",
        f"|非G|{t['nonG_correct_Surface']}|{t['D_correct_nonG']}|{f(ratio(t['D_correct_nonG'],t['D_correct']))}|{f(ratio(t['D_correct_nonG'],t['nonG_correct_Surface']))}|",'',
        f"Surface→Cloud新增526351=原正确Surface损伤506798+原错误Shadow转Cloud19553；消除27632、净增498719，全部六方向及27格精确复现。G支持{t['G']:,}、真值Surface/Cloud/Shadow="+'/'.join(str(t['G_truth_'+c]) for c in C)+f"，Surface precision {f(ratio(t['H'],t['G']))}%。G内必要纠错Cloud/Shadow="+'/'.join(str(t['G_Source_wrong_student_correct_truth_'+c]) for c in ('Cloud','Shadow'))+'。',
        f"共同支持上sum(s)≤1e-12为{t['zero_or_tiny_sum']}，不进入G。G在image-valid上先计算再读取参考，image-valid与参考共同支持的差额、各自分母和G掩膜SHA另表。",'',
        '|参考区域|G/H支持|G内损伤|非G损伤|G正确Surface损伤率%|','|---|---|---:|---:|---:|']
    for r in boundary:lines.append(f"|{r['region']}|{r['G']}/{r['H']}|{r['D_correct_G']}|{r['D_correct_nonG']}|{f(r['G_correct_Surface_damage_pct'])}|")
    lines+=['','|决策|组均/池化mIoU %|Surface/Cloud/Shadow IoU %|Surface→Cloud/Cloud→Surface|Shadow TP/FP/FN/TN|宏召回/误报 %|Shadow宏/池化AP %|','|---|---|---|---|---|---|---|']
    for r in summary:
        lines.append('|'+ '|'.join([r['strategy'],f"{f(r['group_mean_mIoU_pct'])}/{f(r['pooled_mIoU_pct'])}",'/'.join(f(r[c+'_IoU_pct']) for c in C),
            f"{r['Surface_to_Cloud']}/{r['Cloud_to_Surface']}",'/'.join(str(r['Shadow_'+k]) for k in ('TP','FP','FN','TN')),
            f"{f(r['macro_recall_pct'])}/{f(r['macro_FPR_pct'])}",f"{f(r['macro_AP_pct'])}/{f(r['pooled_AP_pct'])}"])+'|')
    delta=summary[2]['group_mean_mIoU_pct']-summary[1]['group_mean_mIoU_pct'];pd=summary[2]['pooled_mIoU_pct']-summary[1]['pooled_mIoU_pct']
    lines+=['',f"G90−Q90：组均mIoU {delta:+.3f}pp、池化{pd:+.3f}pp；其余完整差值另CSV。只把G且学生Cloud改成Surface：共{change['changed']:,}像元，修复真Surface {change['repaired_true_Surface']:,}，撤销真Cloud纠正{change['revoked_true_Cloud_correction']:,}，真Shadow的Cloud→Surface变化{change['true_Shadow_Cloud_to_Surface']:,}。Shadow决策/连续评分逐像元不变，AP原值引用并按相同文件SHA/支持验收，不宣称排序收益。",'',
        '场景字段固定11项，每项0/1/缺失全保留且独立求和回全池；字段互相重叠，不能跨字段相加。全部32组和scene完整ID字段映射、来源SHA、重复/冲突/缺失原因另表。']
    for field in FIELDS:
        r=next(r for r in scene if r['field']==field and r['value']=='1')
        lines.append(f"- {field}=1：{r['products']}产品，G内原正确Surface损伤{r['D_correct_G']}，占全池同类事件{f(r['share_of_pool_D_correct_G_pct'])}%。")
    lines+=['','## 结论分析','',
        f"原正确Surface损伤有{f(ratio(t['D_correct_G'],t['D_correct']))}%位于固定G；是否主要集中于高置信正确Surface应与非G支持及两层损伤率同时判断。Source高置信并非真值，固定保护撤销Cloud纠正{change['revoked_true_Cloud_correction']}，代价不能省略。", 
        ('该固定读出提供后续单一有向margin保持约束的诊断依据，但其训练收益尚未验证。' if t['D_correct_G']>t['D_correct_nonG'] and change['repaired_true_Surface']>change['revoked_true_Cloud_correction'] else '该固定条件的损伤覆盖或纠错代价存在局限，不能据此保证margin约束有效；本轮不调整门限。'),
        '这是固定缓存上的读出诊断，未部署，不代表训练约束可达到同效；同时运行Source与学生将有额外推理成本。参考边界只是半径2标签邻接，不是薄云/物理边缘；snow/ice等是场景标签，不能称对应地物像元致错机制。',
        '历史development已多次使用，单seed，DINO预训练交集未知；不声称盲测、统计显著、物理因果或云特有新方法。缺类union=0为NA，32组完整保留。不启动下一phase。']
    (ROOT/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    try:main()
    except BaseException:
        write(ROOT/'FAILURE.json',dict(traceback=traceback.format_exc(),neural_calls=0));raise

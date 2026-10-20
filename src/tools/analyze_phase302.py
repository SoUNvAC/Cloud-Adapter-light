"""Independent CPU analysis after endpoint seal and full local SHA backup."""
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix
from scipy.ndimage import binary_dilation,binary_erosion
from transfer_phase302 import sha,load,write

R=Path(__file__).resolve().parents[2]
ROOT=R/'outputs/phase302/phase302_20261020'
D=ROOT/'remote_backup'
OLD=R/'outputs/phase300/phase300_20261020/remote_backup/phase300_20261020'
SUPPORT=R/'outputs/phase205/phase205_20261019/delivery'
O=ROOT/'analysis'
S=R/'src/research_results/phase302_20261020'
C=('Surface','Cloud','Shadow')
INPUTS={}
CHECKS=0


def pin(p): INPUTS[str(p)]=sha(p);return load(p)
def table(name,rows):
    with (O/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for row in rows:w.writerow({k:'NA' if v is None else json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def tensor(y,a,b):
    global CHECKS
    t=np.bincount((9*y+3*a+b).astype(np.int64),minlength=27).reshape(3,3,3)
    for c in range(3):
        for i in range(3):
            for j in range(3):
                assert t[c,i,j]==int(((y==c)&(a==i)&(b==j)).sum());CHECKS+=1
    assert np.array_equal(t.sum(2),confusion_matrix(y,a,labels=[0,1,2]))
    assert np.array_equal(t.sum(1),confusion_matrix(y,b,labels=[0,1,2]))
    return t


def cells(t,prefix):
    return [dict(**prefix,truth=C[c],Source_pred=C[i],student_pred=C[j],
        truth_support=int(t[c].sum()),count=int(t[c,i,j])) for c in range(3) for i in range(3) for j in range(3)]


def events(t,prefix):
    rows=[]
    for c in range(3):
        for j in range(3):
            if j==c:continue
            old=int(t[c,j].sum());new=int(t[c,:,j].sum());same=int(t[c,j,j]);n=int(t[c].sum())
            removed=old-same;added=new-same;corrected=int(t[c,j,c])
            assert removed-added==old-new
            rows.append(dict(**prefix,error=C[c]+'->'+C[j],truth_support=n,Source_errors=old,student_errors=new,
                removed=removed,added=added,net_reduction=old-new,corrected=corrected,
                changed_to_other_error=removed-corrected,from_correct=int(t[c,c,j]),
                rate_change_pp=100*(new-old)/n if n else None))
    return rows


def ap(y,s):
    global CHECKS
    if not y.any() or y.all():return None
    order=np.argsort(-s,kind='stable');values=s[order];yy=y[order]
    ends=np.r_[np.flatnonzero(values[:-1]!=values[1:]),len(y)-1]
    tp=np.cumsum(yy,dtype=np.int64)[ends]
    result=float(np.dot(np.diff(np.r_[0,tp]),tp/(ends+1))/tp[-1])
    other=float(average_precision_score(y,s));assert abs(result-other)<1e-12;CHECKS+=1
    return result


def ious(mat):
    diag=np.diag(mat);union=mat.sum(0)+mat.sum(1)-diag
    return np.divide(diag,union,out=np.full(3,np.nan),where=union>0)


def metric(y,p,s):
    mat=np.bincount((3*y+p).astype(np.int64),minlength=9).reshape(3,3)
    assert np.array_equal(mat,confusion_matrix(y,p,labels=[0,1,2]))
    iou=ious(mat)
    for c in range(3):
        union=int(((y==c)|(p==c)).sum())
        if union:assert abs(iou[c]-int(((y==c)&(p==c)).sum())/union)<1e-12
    positive=int((y==2).sum());negative=len(y)-positive
    return dict(support=len(y),positive=positive,negative=negative,confusion=mat.tolist(),
        Surface_IoU_pct=100*float(iou[0]) if np.isfinite(iou[0]) else None,
        Cloud_IoU_pct=100*float(iou[1]) if np.isfinite(iou[1]) else None,
        Shadow_IoU_pct=100*float(iou[2]) if np.isfinite(iou[2]) else None,
        mIoU_pct=100*float(np.nanmean(iou)),AP_pct=None if (a:=ap(y==2,s)) is None else 100*a,
        recall_pct=100*int(mat[2,2])/positive if positive else None,
        FPR_pct=100*int(mat[:2,2].sum())/negative if negative else None)


def regions(y,valid):
    cloud=y==1;edge=np.zeros(y.shape,bool)
    v=valid[:-1]&valid[1:]&(cloud[:-1]!=cloud[1:]);h=valid[:,:-1]&valid[:,1:]&(cloud[:,:-1]!=cloud[:,1:])
    edge[:-1]|=v;edge[1:]|=v;edge[:,:-1]|=h;edge[:,1:]|=h
    a=np.pad(edge,2);b=np.pad(valid,2);expanded=np.zeros_like(valid);inside=np.ones_like(valid)
    for dy in range(5):
        for dx in range(5):
            expanded|=a[dy:dy+len(y),dx:dx+y.shape[1]]
            inside&=b[dy:dy+len(y),dx:dx+y.shape[1]]
    assert np.array_equal(expanded,binary_dilation(edge,structure=np.ones((5,5),bool)))
    assert np.array_equal(inside,binary_erosion(valid,structure=np.ones((5,5),bool),border_value=0))
    return dict(boundary=expanded&inside,nonboundary=~expanded&inside,excluded=valid&~inside)


def average(v):
    v=[x for x in v if x is not None];return sum(v)/len(v) if v else None


def main():
    started=time.monotonic();assert 'torch' not in sys.modules
    assert not O.exists();O.mkdir()
    assert pin(ROOT/'BACKUP_RECEIPT.json')['all_archive_members_verified']
    assert (D/'EXIT_CODE').read_text().strip()=='0'
    endpoint=pin(D/'ENDPOINT_LOCK.json');assert endpoint['all_base_tensors_unchanged']
    assert sha(D/'final_lora.pth')==endpoint['final_sha256']
    manifest=pin(D/'BACKUP_MANIFEST.json')
    for n,v in manifest.items(): assert sha(D/n)==v['sha256'],n
    x,a,b=np.indices((3,3,3)).reshape(3,-1).astype(np.uint8)
    assert np.array_equal(tensor(x,a,b),np.ones((3,3,3),np.int64))
    assert all(r['removed']==r['added']==2 for r in events(tensor(x,a,b),{}))
    plan=[r for r in pin(OLD/'EVALUATION_INPUT_PLAN.json')['rows'] if r['pool']=='Catalogue']
    expected={r['product_id']:r for r in pin(OLD/'EXPECTED_SUPPORT.json')['rows'] if r['pool']=='Catalogue'}
    lock=pin(D/'EVALUATION_LOCK.json');score_sha={(r['arm'],r['product']):r['sha256'] for r in lock['rows']}
    arrays={'Source':[],'LoRA':[]};truths=[];per=[];group_cells=[];group_events=[];boundary_events=[];boundary_cells=[];supports=[]
    pooled=np.zeros((3,3,3),np.int64);region_total={k:pooled.copy() for k in ('boundary','nonboundary','excluded')}
    masks_path=R/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip';INPUTS[str(masks_path)]=sha(masks_path)
    with zipfile.ZipFile(masks_path) as masks:
        for row in plan:
            pid=row['product_id'];gid=row['group']['group_id']
            old=SUPPORT/'infer/R4/scores/Catalogue'/(pid+'.npz');INPUTS[str(old)]=sha(old)
            with np.load(old) as z: ids=z['flat_indices'];common=z['valid']
            ex=expected[pid]
            assert hashlib.sha256(ids.tobytes()).hexdigest()==ex['flat_indices_sha256']
            assert hashlib.sha256(common.tobytes()).hexdigest()==ex['common_valid_sha256']
            raw=masks.read('masks/'+pid+'.npy');assert hashlib.sha256(raw).hexdigest()==row['group']['mask_member_sha256']
            mask=np.load(io.BytesIO(raw));truth=mask.argmax(-1).astype(np.uint8)
            assert (mask.reshape(-1,3)[ids[common]].sum(1)==1).all()
            support=np.zeros((1022,1022),bool);support.ravel()[ids[common]]=True
            assert int(support.sum())==ex['count']
            y=truth[support];preds={}
            for arm in ('Source','LoRA'):
                p=D/'evaluation'/(arm+'_'+pid+'.npz');assert sha(p)==score_sha[arm,pid];INPUTS[str(p)]=sha(p)
                with np.load(p) as z:
                    loss=int((support&~z['valid']).sum())
                    if loss:
                        write(O/'SUPPORT_FAILURE.json',dict(product=pid,lost=loss,reason='RGB finite/nonzero/fill support differs from Phase300; no support silently removed'))
                    assert loss==0, 'RGB lost common support: '+pid
                    scores=z['scores'];prediction=z['pred'];shadow=z['shadow']
                    assert np.array_equal(prediction,scores.argmax(0))
                    shifted=scores-scores.max(0,keepdims=True);prob=np.exp(shifted);prob/=prob.sum(0,keepdims=True)
                    assert np.allclose(shadow,prob[2],rtol=1e-6,atol=1e-7)
                    p1=prediction[support].copy();s=shadow[support].copy();preds[arm]=prediction.copy()
                arrays[arm].append((p1,s));per.append(dict(strategy=arm,product=pid,group_id=gid,**metric(y,p1,s)))
            t=tensor(y,preds['Source'][support],preds['LoRA'][support]);pooled+=t
            group_cells+=cells(t,dict(product=pid,group_id=gid));group_events+=events(t,dict(product=pid,group_id=gid))
            region=regions(truth,support)
            assert np.array_equal(sum(v.astype(np.uint8) for v in region.values()),support.astype(np.uint8))
            for name,choose in region.items():
                rt=tensor(truth[choose],preds['Source'][choose],preds['LoRA'][choose]);region_total[name]+=rt
                prefix=dict(product=pid,group_id=gid,region=name)
                supports.append(dict(**prefix,support=int(choose.sum()),Surface=int((truth[choose]==0).sum()),Cloud=int((truth[choose]==1).sum()),Shadow=int((truth[choose]==2).sum())))
                boundary_events+=events(rt,prefix);boundary_cells+=cells(rt,prefix)
            truths.append(y);gc.collect()
    assert np.array_equal(sum(region_total.values()),pooled)
    table('TRANSITIONS27.csv',cells(pooled,{}));table('PER_GROUP_TRANSITIONS.csv',group_cells)
    event_rows=events(pooled,{});table('ERROR_EVENTS.csv',event_rows);table('PER_GROUP_ERRORS.csv',group_events)
    table('BOUNDARY_EVENTS.csv',boundary_events);table('BOUNDARY_TRANSITIONS.csv',boundary_cells);table('BOUNDARY_SUPPORT.csv',supports)
    table('BOUNDARY_POOLED_EVENTS.csv',[r for name,t in region_total.items() for r in events(t,dict(region=name))])
    table('PER_GROUP_METRICS.csv',per)
    y=np.concatenate(truths);summaries=[]
    for arm in ('Source','LoRA'):
        p=np.concatenate([a for a,s in arrays[arm]]);s=np.concatenate([s for a,s in arrays[arm]])
        pooled_metric=metric(y,p,s);parts=[r for r in per if r['strategy']==arm]
        expected_confusion=pooled.sum(2 if arm=='Source' else 1)
        assert np.array_equal(np.array(pooled_metric['confusion']),expected_confusion)
        summaries.append(dict(strategy=arm,groups=len(parts),AP_groups=sum(r['AP_pct'] is not None for r in parts),
            group_mean_mIoU_pct=average(r['mIoU_pct'] for r in parts),pooled_mIoU_pct=pooled_metric['mIoU_pct'],
            **{k:pooled_metric[k] for k in ('Surface_IoU_pct','Cloud_IoU_pct','Shadow_IoU_pct','AP_pct')},
            macro_AP_pct=average(r['AP_pct'] for r in parts),macro_recall_pct=average(r['recall_pct'] for r in parts),
            macro_FPR_pct=average(r['FPR_pct'] for r in parts),max_group_FPR_pct=max(r['FPR_pct'] for r in parts if r['FPR_pct'] is not None),
            confusion=pooled_metric['confusion']))
        del p,s;gc.collect()
    table('FIXED_RESULTS.csv',summaries)
    delta={k[:-4]+'_pp':summaries[1][k]-summaries[0][k] for k in summaries[0] if k.endswith('_pct') and summaries[0][k] is not None and summaries[1][k] is not None}
    write(O/'RESULT_DIFFERENCES_PP.json',delta)
    table('RESULT_DIFFERENCES_PP.csv',[dict(strategy='LoRA minus Source',**delta)])
    probes=[];probe_cells=[];full_cells=[]
    fit_masks={r['product']:r for r in pin(R/'outputs/phase208/phase208_20261020/12201_FIT_INPUT_LOCK.json')['bindings']}
    with zipfile.ZipFile(masks_path) as masks:
        for pid in sorted(r['product'] for r in pin(D/'FIT_IMAGES.json')['rows'])[:4]:
            raw=masks.read('masks/'+pid+'.npy');assert hashlib.sha256(raw).hexdigest()==fit_masks[pid]['mask_member_sha256']
            m=np.load(io.BytesIO(raw))[255:767,255:767];truth=m.argmax(-1).astype(np.uint8)
            with np.load(D/'acceptance'/(pid+'.npz')) as z:source=z['scores'];initial=source.argmax(0)
            with np.load(D/'teacher'/(pid+'.npz')) as z:full=z['pred'][255:767,255:767]
            common=None
            for attempt in (0,100,500,1000):
                path=D/'probes'/(pid+'_'+str(attempt)+'.npz');assert sha(path)==endpoint['probes'][path.name]
                with np.load(path) as z:
                    current=z['pred'];valid=z['valid']&(m.sum(-1)==1)
                    if attempt==0:assert float(np.abs(z['scores']-source).max())<=1e-6
                if common is None:common=valid
                assert np.array_equal(common,valid)
                t=tensor(truth[valid],initial[valid],current[valid]);probe_cells+=cells(t,dict(product=pid,attempt=attempt))
                for c in range(3):
                    correct=int(t[c,c,c]);damaged=int(t[c,c].sum())-correct;repaired=int(t[c,:,c].sum())-correct
                    probes.append(dict(product=pid,attempt=attempt,truth=C[c],support=int(t[c].sum()),
                        Source_correct_student_wrong=damaged,Source_wrong_student_correct=repaired,net_correct=repaired-damaged,
                        Shadow_prediction_pct=100*int((current[valid]==2).sum())/int(valid.sum()) if valid.any() else None))
                if attempt in (0,1000):full_cells+=cells(tensor(truth[valid],full[valid],current[valid]),dict(product=pid,attempt=attempt))
    table('SAME512_TRAINING.csv',probes);table('SAME512_TRANSITIONS.csv',probe_cells);table('FULL_TEACHER_TO_512.csv',full_cells)
    curve=[json.loads(s) for s in (D/'TRAIN_CURVE.jsonl').read_text().splitlines()]
    assert len(curve)==1000 and sum(r['updated'] for r in curve)==endpoint['updates']
    table('TRAINING_CURVE.csv',curve)
    pseudo=pin(D/'PSEUDO_LOCK.json')['rows']
    table('PSEUDO_COVERAGE.csv',[dict(product=r['product'],valid=r['valid'],accepted=r['accepted'],
        Surface=r['counts'][0],Cloud=r['counts'][1],Shadow=r['counts'][2],sha256=r['sha256']) for r in pseudo])
    write(O/'INPUT_SHA256.json',INPUTS)
    write(O/'CPU_ACCEPTANCE.json',dict(status='passed',seconds=time.monotonic()-started,independent_checks=CHECKS,
        groups=32,source_student_marginals_exact=True,boundary_partition_exact=True,all_probe_zero_lora_matches_source=True,
        all_common_reference_support_retained=True,AP_independent_sklearn_checked=True,neural_calls=0))
    report(summaries,delta,event_rows,boundary_events,group_events,probes,endpoint)
    S.mkdir(parents=True,exist_ok=True)
    import shutil
    for p in O.iterdir():shutil.copyfile(p,S/p.name)
    for name in ('SOURCE_ADMISSION.json','MODEL_ACCEPTANCE.json','PREPROCESS_ACCEPTANCE.json','ENDPOINT_LOCK.json','EXECUTION_LOCK.json','EVALUATION_LOCK.json','train_RECEIPT.json','evaluate_RECEIPT.json'):
        shutil.copyfile(D/name,S/name)
    shutil.copyfile(ROOT/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json')
    print(json.dumps(load(O/'CPU_ACCEPTANCE.json')))


def report(summaries,delta,events,boundaries,groups,probes,endpoint):
    model=load(D/'MODEL_ACCEPTANCE.json');source=load(D/'SOURCE_ADMISSION.json')
    lines=['# Phase302：VFM固定伪标签适配','','## 数据结果','',
        f"Source SHA `{source['source_sha256']}`；DINOv2-S有效blocks={model['active_blocks']}，Cloud-Adapter与Mask2Former冻结；LoRA可训练参数{model['trainable_parameters']:,}。cloud-lite-pt210、FP32、seed30201，1000尝试/{endpoint['updates']}更新/{endpoint['skipped']}跳过；末步固定封存后评价。",'',
        '|策略|组均/池化mIoU %|Surface/Cloud/Shadow IoU %|Shadow宏/池化AP %|宏召回 %|宏/最大组误报 %|',
        '|---|---|---|---|---|---|']
    def fmt(v):return 'NA' if v is None else f'{v:.3f}'
    for r in summaries:
        values=[r['strategy'],f"{fmt(r['group_mean_mIoU_pct'])}/{fmt(r['pooled_mIoU_pct'])}",
            '/'.join(fmt(r[k]) for k in ('Surface_IoU_pct','Cloud_IoU_pct','Shadow_IoU_pct')),
            f"{fmt(r['macro_AP_pct'])}/{fmt(r['AP_pct'])}",fmt(r['macro_recall_pct']),f"{fmt(r['macro_FPR_pct'])}/{fmt(r['max_group_FPR_pct'])}"]
        lines.append('|'+ '|'.join(values)+'|')
    lines.append('|'+ '|'.join(['Δ (pp)',
        f"{fmt(delta.get('group_mean_mIoU_pp'))}/{fmt(delta.get('pooled_mIoU_pp'))}",
        '/'.join(fmt(delta.get(k)) for k in ('Surface_IoU_pp','Cloud_IoU_pp','Shadow_IoU_pp')),
        f"{fmt(delta.get('macro_AP_pp'))}/{fmt(delta.get('AP_pp'))}",fmt(delta.get('macro_recall_pp')),
        f"{fmt(delta.get('macro_FPR_pp'))}/{fmt(delta.get('max_group_FPR_pp'))}"])+'|')
    lines+=['',f"两策略绝对值单位%，Δ行单位pp；均保留32组；AP有效组{summaries[0]['AP_groups']}/{summaries[1]['AP_groups']}。全部差值见RESULT_DIFFERENCES_PP.json。IoU对union=0类取NA，组mIoU仅平均有union的类；AP要求正负均存在，宏召回仅正支持组，宏误报仅负支持组。保留所有原组与缺类惩罚。",'',
        '|错误|Source/学生|消除/新增|净减少|消除中改正/另一错误|错误率差pp|','|---|---|---|---:|---|---:|']
    for e in events:lines.append(f"|{e['error']}|{e['Source_errors']}/{e['student_errors']}|{e['removed']}/{e['added']}|{e['net_reduction']}|{e['corrected']}/{e['changed_to_other_error']}|{fmt(e['rate_change_pp'])}|")
    for error in ('Cloud->Shadow','Surface->Cloud'):
        e=next(e for e in events if e['error']==error)
        selected=[r for r in groups if r['error']==error];worst=max(selected,key=lambda r:r['added'])
        away=sum(r['added'] for r in boundaries if r['error']==error and r['region']=='nonboundary')
        lines.append(f"- {error}：新增{e['added']:,}，其中原正确转入{e['from_correct']:,}；参考非边界新增{away:,}（{100*away/e['added'] if e['added'] else 0:.1f}%）。最大新增组{worst['product']}：{worst['added']:,}（{100*worst['added']/e['added'] if e['added'] else 0:.1f}%）。全部32组、边界/非边界/排除区支持与错误率另表保留。")
    selected=[p for p in probes if p['attempt']==1000]
    repaired=sum(p['Source_wrong_student_correct'] for p in selected);damaged=sum(p['Source_correct_student_wrong'] for p in selected)
    lines+=['',f'四探针同512末步纠错{repaired:,}、损伤{damaged:,}、净正确{repaired-damaged:+,}；按Surface/Cloud/Shadow净正确为'+ '/'.join(f"{sum(p['net_correct'] for p in selected if p['truth']==c):+d}" for c in C)+'。全部4时点及完整教师→裁剪对照另表。', '', '## 结论分析','']
    cs=next(e for e in events if e['error']=='Cloud->Shadow');sc=next(e for e in events if e['error']=='Surface->Cloud')
    reproduced=cs['net_reduction']>0 and sc['net_reduction']<0
    lines+=[('本轮复现Cloud→Shadow减少与Surface→Cloud增加并存。' if reproduced else '本轮未同时复现Cloud→Shadow减少与Surface→Cloud增加，不硬套Omni旧机制。')+
        '此为固定单seed描述性诊断，不能据此认定骨干因果效应或全域收益；RGB、Source、得分、适配结构和窗口协议均与Omni不同。', '',
        '若继续，只检验一项有向转移约束：在固定伪标签LoRA中限制原可靠Surface→Cloud损伤，观察能否保留云影纠正；本轮不启动新实验。' if reproduced else '保留不复现结论；后续须先解释Source、输入尺度与置信覆盖差异，本轮不增加seed或约束实验。','',
        'Source使用CloudSEN12官方train/val、源验证选模；已有PNG行序及100m缓冲空间审计未发现本轮132产品与源footprint重叠，但DINO预训练交集未知。Catalogue历史标签和开发记录已用，不能称新独立盲测。本轮SFDA训练仅访问目标fit影像与固定Source伪标签，无源样本；不读取PixBox、L8或其他评价池。参考边界只是既有标签邻接的半径2邻域，不代表薄云或物理云边界。']
    (O/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()

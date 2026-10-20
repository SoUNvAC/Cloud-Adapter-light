"""Reference-local fixed endpoint analysis; no neural calls or model selection."""
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import zipfile

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'outputs/phase300/phase300_20261020'
D = ROOT/'remote_backup/phase300_20261020'
sys.path.insert(0,str(REPO/'outputs/phase202/phase202_20261017/software'))
import numpy as np
import phase300_reference as reference

NAMES = ('R4','E4','B','M')
DISPLAY = dict(R4='R4',E4='E4',B='B-PL',M='M-PL')
PARAMS = dict(R4=0,E4=0,B=944641,M=970369)


def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return reference.sha(path)
def save(name,data):
    (ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def table(name,rows):
    absolute_units = name=='TABLE1_FIXED_RESULTS.csv' or name.endswith('_PER_GROUP.csv') or name=='TABLE5_CONFUSION_ERRORS.csv'
    def column(key):
        return key[:-3]+'_pct' if absolute_units and key.endswith('_pp') else key
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=[column(k) for k in rows[0]])
        writer.writeheader()
        for row in rows:
            writer.writerow({column(k):'NA' if v is None else json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def ious(matrix):
    matrix = np.asarray(matrix,np.int64)
    diagonal = np.diag(matrix)
    union = matrix.sum(0)+matrix.sum(1)-diagonal
    return np.divide(diagonal,union,out=np.full(3,np.nan),where=union>0)


def average(values):
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def metrics(pool,groups):
    per = []
    summaries = []
    lookup = {}
    with (reference.OLD.parent/(pool+'_PER_GROUP.csv')).open(encoding='utf-8-sig') as stream:
        prior = {(r['strategy'],r['group_id']):r for r in csv.DictReader(stream)}
    for name in NAMES:
        parts = []
        mats = []
        for gid,group in groups.items():
            result,matrix = reference.metric(group['y'],group['pred'][name],group['scores'][name])
            if name in ('R4','E4'):
                previous = prior[name,gid]
                for key in ('positive','negative','TP','FP','FN','TN','Shadow_to_Surface','Shadow_to_Cloud'):
                    assert result[key] == int(previous[key])
                assert result['AP'] is None and previous['AP']=='' or abs(result['AP']-float(previous['AP']))<1e-12
            iou = ious(matrix)
            # Independently recompute each class's intersection and union.
            for cls in range(3):
                yy = group['y']==cls
                pp = group['pred'][name]==cls
                union = int((yy|pp).sum())
                direct = int((yy&pp).sum())/union if union else np.nan
                assert np.isnan(direct) and np.isnan(iou[cls]) or abs(direct-iou[cls])<1e-12
            record = dict(pool=pool,strategy=DISPLAY[name],group_id=gid,products=group['products'],
                          valid_support=len(group['y']),**result,
                          Surface_IoU_pp=100*float(iou[0]) if pool=='Catalogue' and np.isfinite(iou[0]) else None,
                          Cloud_IoU_pp=100*float(iou[1]) if pool=='Catalogue' and np.isfinite(iou[1]) else None,
                          Shadow_IoU_pp=100*float(iou[2]) if pool=='Catalogue' and np.isfinite(iou[2]) else None,
                          mIoU_pp=100*float(np.nanmean(iou)) if pool=='Catalogue' else None)
            # Rename fractional rates in the complete table with explicit units.
            for key in ('AP','recall','negative_shadow_fraction','precision'):
                value = record.pop(key)
                record[key+'_pp'] = 100*value if value is not None else None
            lookup[name,gid] = record
            per.append(record)
            parts.append(record)
            mats.append(matrix)
        pooled = np.sum(mats,axis=0)
        iou = ious(pooled)
        y = np.concatenate([g['y'] for g in groups.values()])
        scores = np.concatenate([g['scores'][name] for g in groups.values()])
        predicted = np.concatenate([g['pred'][name] for g in groups.values()])
        pooled_check = np.bincount((3*y+predicted).astype(np.int64),minlength=9).reshape(3,3)
        assert np.array_equal(pooled,pooled_check)
        pooled_ap = reference.ap(y==2,scores)
        group_mean = average(p['mIoU_pp'] for p in parts)
        if group_mean is not None:
            independent_mean = sum(p['mIoU_pp'] for p in parts)/len(parts)
            assert abs(group_mean-independent_mean)<1e-10
        summaries.append(dict(pool=pool,strategy=DISPLAY[name],groups=len(groups),
          AP_groups=sum(p['AP_pp'] is not None for p in parts),new_parameters=PARAMS[name],trainable_parameters=PARAMS[name],
          group_mean_mIoU_pp=group_mean,
          pooled_mIoU_pp=100*float(np.nanmean(iou)) if pool=='Catalogue' else None,
          pooled_Surface_IoU_pp=100*float(iou[0]) if pool=='Catalogue' and np.isfinite(iou[0]) else None,
          pooled_Cloud_IoU_pp=100*float(iou[1]) if pool=='Catalogue' and np.isfinite(iou[1]) else None,
          pooled_Shadow_IoU_pp=100*float(iou[2]) if pool=='Catalogue' and np.isfinite(iou[2]) else None,
          macro_Shadow_AP_pp=average(p['AP_pp'] for p in parts),
          pooled_Shadow_AP_pp=100*pooled_ap if pooled_ap is not None else None,
          macro_Shadow_recall_pp=average(p['recall_pp'] for p in parts),
          macro_negative_FPR_pp=average(p['negative_shadow_fraction_pp'] for p in parts),
          max_group_negative_FPR_pp=max(p['negative_shadow_fraction_pp'] for p in parts if p['negative_shadow_fraction_pp'] is not None),
          positive=int((y==2).sum()),negative=int((y!=2).sum()),confusion=pooled.tolist()))
        del y,scores,predicted,pooled_check
        gc.collect()
    table(pool+'_PER_GROUP.csv',per)
    return summaries,lookup


def comparisons(pool,groups,lookup):
    ids = list(groups)
    indexes = np.random.default_rng(300020).integers(len(ids),size=(10000,len(ids)))
    rows = []
    group_rows = []
    for left,right in (('M','R4'),('B','R4'),('M','B'),('M','E4'),('B','E4'),('R4','E4')):
        contrast = DISPLAY[left]+'−'+DISPLAY[right]
        a = [lookup[left,g] for g in ids]
        b = [lookup[right,g] for g in ids]
        row = dict(pool=pool,comparison=contrast,delta_group_mean_mIoU_pp=None,
                   group_mean_CI95_low_pp=None,group_mean_CI95_high_pp=None,
                   delta_pooled_mIoU_pp=None,pooled_CI95_low_pp=None,pooled_CI95_high_pp=None)
        if pool == 'Catalogue':
            delta = np.array([x['mIoU_pp']-y['mIoU_pp'] for x,y in zip(a,b)])
            means = delta[indexes].mean(1)
            am = np.asarray([r['confusion'] for r in a],np.int64)
            bm = np.asarray([r['confusion'] for r in b],np.int64)
            def batch_miou(m):
                diag = np.diagonal(m,axis1=-2,axis2=-1)
                union = m.sum(-2)+m.sum(-1)-diag
                return np.nanmean(np.divide(diag,union,out=np.full(diag.shape,np.nan),where=union>0),axis=-1)*100
            pooled_delta = batch_miou(am[indexes].sum(1))-batch_miou(bm[indexes].sum(1))
            # Independent scalar bootstrap check on the first 100 draws.
            for i,draw in enumerate(indexes[:100]):
                aa = sum((am[j] for j in draw),np.zeros((3,3),np.int64))
                bb = sum((bm[j] for j in draw),np.zeros((3,3),np.int64))
                scalar = 100*(np.nanmean(ious(aa))-np.nanmean(ious(bb)))
                assert abs(scalar-pooled_delta[i])<1e-10
            gm = np.quantile(means,[.025,.975])
            pm = np.quantile(pooled_delta,[.025,.975])
            row.update(delta_group_mean_mIoU_pp=float(delta.mean()),group_mean_CI95_low_pp=float(gm[0]),
                       group_mean_CI95_high_pp=float(gm[1]),
                       delta_pooled_mIoU_pp=float(batch_miou(am.sum(0))-batch_miou(bm.sum(0))),
                       pooled_CI95_low_pp=float(pm[0]),pooled_CI95_high_pp=float(pm[1]))
            worst = int(delta.argmin())
            worst_group,worst_value = ids[worst],float(delta[worst])
        else: worst_group,worst_value = None,None
        for key,dest in (('AP_pp','delta_macro_Shadow_AP_pp'),('recall_pp','delta_macro_Shadow_recall_pp'),
                         ('negative_shadow_fraction_pp','delta_macro_negative_FPR_pp')):
            differences = [x[key]-y[key] for x,y in zip(a,b) if x[key] is not None and y[key] is not None]
            row[dest] = average(differences)
        row.update(worst_group=worst_group,worst_group_products=groups[worst_group]['products'] if worst_group else None,
                   worst_group_mIoU_delta_pp=worst_value,
                   bootstrap_draws=10000 if pool=='Catalogue' else None,
                   bootstrap_seed=300020 if pool=='Catalogue' else None)
        rows.append(row)
        for gid,x,y in zip(ids,a,b):
            group_rows.append(dict(pool=pool,comparison=contrast,group_id=gid,products=x['products'],
                  delta_mIoU_pp=x['mIoU_pp']-y['mIoU_pp'] if pool=='Catalogue' else None,
                  delta_AP_pp=x['AP_pp']-y['AP_pp'] if x['AP_pp'] is not None else None,
                  delta_recall_pp=x['recall_pp']-y['recall_pp'] if x['recall_pp'] is not None else None,
                  delta_negative_FPR_pp=x['negative_shadow_fraction_pp']-y['negative_shadow_fraction_pp']))
    table(pool+'_GROUP_DIFFERENCES.csv',group_rows)
    return rows


def oracle(pool,groups,summaries):
    """One global threshold; tied scores enter together; no interpolation."""
    negative_groups = sum(bool((g['y']!=2).any()) for g in groups.values())
    positive_groups = sum(bool((g['y']==2).any()) for g in groups.values())
    scores = np.concatenate([g['scores']['R4'] for g in groups.values()])
    negative_weights = np.concatenate([np.where(g['y']!=2,1/(negative_groups*max(1,int((g['y']!=2).sum()))),0) for g in groups.values()])
    positive_weights = np.concatenate([np.where(g['y']==2,1/(positive_groups*max(1,int((g['y']==2).sum()))),0) for g in groups.values()])
    order = np.argsort(scores,kind='mergesort')[::-1]
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:]!=sorted_scores[:-1]),len(scores)-1]
    fpr = np.cumsum(negative_weights[order],dtype=np.float64)[ends]
    recall = np.cumsum(positive_weights[order],dtype=np.float64)[ends]
    summary = {r['strategy']:r for r in summaries}
    rows = []
    for name in ('B','M'):
        student = summary[DISPLAY[name]]
        budget = student['macro_negative_FPR_pp']/100
        feasible = np.flatnonzero(fpr<=budget+1e-12)
        threshold = float(sorted_scores[ends[feasible[-1]]]) if len(feasible) else float(np.nextafter(np.float64(sorted_scores[0]),np.inf))
        actual_fpr = average(float(((g['scores']['R4']>=threshold)&(g['y']!=2)).sum())/int((g['y']!=2).sum()) for g in groups.values() if (g['y']!=2).any())
        actual_recall = average(float(((g['scores']['R4']>=threshold)&(g['y']==2)).sum())/int((g['y']==2).sum()) for g in groups.values() if (g['y']==2).any())
        assert actual_fpr <= budget+1e-9
        if len(feasible):
            assert abs(actual_fpr-fpr[feasible[-1]])<1e-8 and abs(actual_recall-recall[feasible[-1]])<1e-8
        rows.append(dict(pool=pool,student=DISPLAY[name],student_default_FPR_budget_pp=budget*100,
          R4_oracle_global_threshold=threshold,R4_actual_FPR_pp=actual_fpr*100,
          actual_minus_budget_pp=(actual_fpr-budget)*100,student_default_recall_pp=student['macro_Shadow_recall_pp'],
          R4_oracle_recall_pp=actual_recall*100,
          student_minus_R4_oracle_recall_pp=student['macro_Shadow_recall_pp']-actual_recall*100,
          student_minus_R4_macro_AP_pp=student['macro_Shadow_AP_pp']-summary['R4']['macro_Shadow_AP_pp'],
          oracle=True,deployment_threshold=False,ties='whole score ties; >= global threshold'))
    return rows


def training_tables():
    teacher = load(D/'teacher/CACHE_LOCK.json')
    historical = {r['product']:r for r in load(REPO/'outputs/phase208/phase208_20261020/12201_FIT_INPUT_LOCK.json')['bindings']}
    coverage = []
    for row in teacher['rows']:
        n = row['image_valid']
        coverage.append(dict(product=row['product'],group_id=row['group_id'],image_valid=n,
                   accepted=row['accepted'],accepted_fraction=row['accepted']/n,
                   Surface_fraction=row['class_accepted'][0]/n,Cloud_fraction=row['class_accepted'][1]/n,
                   Shadow_fraction=row['class_accepted'][2]/n,cache_sha256=row['sha256']))
    table('PSEUDOLABEL_PER_PRODUCT.csv',coverage)
    table('FIT_SUPPORT_DIFFERENCES.csv',[
        dict(product=r['product'],image_only_valid=r['image_valid'],
             historical_supervised_valid=historical[r['product']]['common_valid'],
             count_difference=r['image_valid']-historical[r['product']]['common_valid'],
             historical_counts_read='local audit after both endpoint seals; not training input') for r in teacher['rows']])
    n = sum(r['image_valid'] for r in teacher['rows'])
    counts = np.sum([r['class_accepted'] for r in teacher['rows']],axis=0)
    summary = []
    probes = []
    for arm in ('B','M'):
        receipt = load(D/('train_'+arm)/'TRAINING_RECEIPT.json')
        assert receipt['attempts']==1000 and receipt['updates']+receipt['skipped']==1000
        assert receipt['final_sha256']==sha(D/('train_'+arm)/'module_final.safetensors')
        nonempty = [r for r in receipt['losses'] if r['CE'] is not None]
        summary.append(dict(strategy=DISPLAY[arm],image_valid=n,accepted_fraction=float(counts.sum()/n),
                     Surface_fraction=float(counts[0]/n),Cloud_fraction=float(counts[1]/n),Shadow_fraction=float(counts[2]/n),
                     attempts=1000,updates=receipt['updates'],all_ignore_skips=receipt['skipped'],
                     first_nonempty_CE=nonempty[0]['CE'] if nonempty else None,
                     last_nonempty_CE=nonempty[-1]['CE'] if nonempty else None,
                     teacher_cache_lock_sha256=receipt['pseudo_cache_lock_sha256'],
                     schedule_sha256=receipt['crop_schedule_sha256']))
        for row in load(D/('train_'+arm)/'PROBE_TABLE.json')['rows']:
            probes.append(dict(product=row['product'],strategy=DISPLAY[arm],attempt=row['attempt'],
                       Surface_fraction=row['prediction_counts'][0]/row['valid'],
                       Cloud_fraction=row['prediction_counts'][1]/row['valid'],
                       Shadow_fraction=row['prediction_counts'][2]/row['valid'],
                       relative_teacher_change_fraction=row['changed']/row['valid'],valid=row['valid'],sha256=row['sha256']))
    assert summary[0]['teacher_cache_lock_sha256']==summary[1]['teacher_cache_lock_sha256']
    assert summary[0]['schedule_sha256']==summary[1]['schedule_sha256']
    assert len(probes)==32
    table('TABLE3_TRAINING.csv',summary)
    table('TABLE4_PROBES.csv',probes)
    return summary


def probe_truth_audit():
    fit = {r['product']:r for r in load(REPO/'outputs/phase208/phase208_20261020/12201_FIT_INPUT_LOCK.json')['bindings']}
    rows = []
    for arm in ('B','M'):
        for row in load(D/('train_'+arm)/'PROBE_TABLE.json')['rows']:
            if row['attempt']!=1000: continue
            file = D/('train_'+arm)/'probes'/(row['product']+'_1000.npz')
            assert sha(file)==row['sha256']
            with np.load(file) as data:
                teacher,student,valid = data['teacher_parent'],data['parent'],data['valid']
            with zipfile.ZipFile(REPO/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip') as source:
                content = source.read('masks/'+row['product']+'.npy')
            assert hashlib.sha256(content).hexdigest()==fit[row['product']]['mask_member_sha256']
            masks = np.load(io.BytesIO(content),allow_pickle=False)[255:767,255:767]
            common = valid & (masks.sum(-1)==1)
            truth = masks.argmax(-1)
            tc,sc = teacher==truth,student==truth
            for cls in range(3):
                support = common & (truth==cls)
                rows.append(dict(product=row['product'],strategy=DISPLAY[arm],truth_class=cls,
                    valid_image_pixels=int(valid.sum()),joint_image_and_reference_pixels=int(common.sum()),
                    class_support=int(support.sum()),
                    teacher_correct_student_wrong=int((support&tc&~sc).sum()),
                    teacher_wrong_student_correct=int((support&~tc&sc).sum()),
                    both_correct=int((support&tc&sc).sum()),both_wrong=int((support&~tc&~sc).sum()),
                    both_wrong_same_class=int((support&~tc&~sc&(teacher==student)).sum())))
    table('PROBE_FINAL_TRUTH_AUDIT.csv',rows)


def shift_errors(pool,groups):
    rows = []
    for label,source_classes,target in (('Surface_to_Cloud',(0,),1),('Surface_to_Shadow',(0,),2),
                                         ('Cloud_to_Surface',(1,),0),('Shadow_to_Surface',(2,),0)):
        support = sum(int(np.isin(g['y'],source_classes).sum()) for g in groups.values())
        record = dict(candidate='cloud_shadow_surface_confusion',subgroup=label,pool=pool,
          definition='fixed common three-parent truth; no posthoc scene selection',products=len({p for g in groups.values() for p in g['products'].split(',')}),
          groups=len(groups),reference_support=support)
        for name in NAMES:
            count = sum(int((np.isin(g['y'],source_classes)&(g['pred'][name]==target)).sum()) for g in groups.values())
            record[DISPLAY[name]+'_count'] = count
            record[DISPLAY[name]+'_rate_pp'] = 100*count/support if support else None
        record.update(interpretation='directional confusion on available common support',gap='specific bright/dark surface semantics require authoritative field mapping')
        rows.append(record)
    return rows


def main():
    start = time.time()
    assert (D/'EXIT_CODE').read_text().strip()=='0'
    state = load(D/'CONTROLLER_STATUS.json')
    assert state['status']=='completed_pending_local_evaluation'
    assert state['total_calls']<=7000 and state['stage_seconds_total']<=14400
    seals = load(D/'TRAINING_SEAL.json')
    for name in ('B','M'): assert sha(D/('train_'+name)/'module_final.safetensors')==seals[name]
    save('LOCAL_EVALUATION_LOCK.json',dict(after_both_endpoints_sealed=True,training_seal=seals,
       local_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
       analysis_code_sha256=sha(Path(__file__)),reference_join_helper_sha256=sha(Path(reference.__file__)),
       original_helper_provenance=reference.ORIGINAL_SOURCE_SHA256,
       subgroup_code_sha256=sha(REPO/'src/tools/phase300_subgroups.py'),
       subgroup_mapping_sha256=sha(ROOT/'SUBGROUP_MAPPING_LOCK.json'),
       bootstrap_draws=10000,bootstrap_seed=300020,selection='fixed endpoints only',
       prior_manual_metadata_access='Local head preview of PixBox reference records and metadata fields occurred before endpoint seals; no prediction join, training feedback or selection followed; see phase300.md'))
    reference.O = ROOT
    reference.D = D
    reference.NAMES = NAMES
    sealed = {n:v for n,v in load(reference.OLD/'infer/SCORES_SEALED.json')['scores'].items() if n in ('R4','E4')}
    sealed.update(load(D/'infer/SCORES_SEALED.json')['scores'])
    expected = {r['product_id']:r for r in load(D/'EXPECTED_SUPPORT.json')['rows']}
    plan = load(D/'EVALUATION_INPUT_PLAN.json')['rows']
    summaries,contrasts,oracles,shifts,supports = [],[],[],[],[]
    for pool,reader in (('Catalogue',reference.cat),('PixBox',reference.pix)):
        groups,support = reader([r for r in plan if r['pool']==pool],sealed,expected)
        assert len(groups)==(32 if pool=='Catalogue' else 10)
        summary,lookup = metrics(pool,groups)
        summaries += summary
        contrasts += comparisons(pool,groups,lookup)
        oracles += oracle(pool,groups,summary)
        shifts += shift_errors(pool,groups)
        supports += support
        print(pool,'metrics/bootstrap/oracle complete',flush=True)
        del groups,lookup
        gc.collect()
    table('TABLE1_FIXED_RESULTS.csv',summaries)
    table('TABLE2_COMPARISONS.csv',[r for r in contrasts if r['comparison'] in ('M-PL−R4','B-PL−R4','M-PL−B-PL')])
    table('E4_COMPARISONS.csv',[r for r in contrasts if 'E4' in r['comparison']])
    table('TABLE5_CONFUSION_ERRORS.csv',shifts)
    table('TABLE6_ORACLE_WORKPOINT.csv',oracles)
    from phase300_subgroups import analyze as subgroups
    subgroups(ROOT,D,reference,sealed,plan,table)
    train_summary = training_tables()
    probe_truth_audit()
    save('COMMON_SUPPORT_ACCEPTANCE.json',dict(rows=supports,all_four_same_support=True))
    save('FIXED_RESULTS.json',dict(summary=summaries,comparisons=contrasts,oracle=oracles,training=train_summary))
    save('CPU_ACCEPTANCE.json',dict(status='passed',seconds=time.time()-start,neural_calls=0,
           numpy_version=np.__version__,
           independent_AP_confusion_checks=reference.CHECKS,independent_group_IoUs=True,
           paired_group_bootstrap_pooled_scalar_checks=600,bootstrap_seed=300020,
           missing_class_rule='IoU union=0 -> NA; mIoU mean over nonempty unions; AP requires positive and negative; recall needs positives; FPR needs negatives; all groups retained',
           PixBox_mIoU_and_IoUs='NA',single_training_seed=30001))


if __name__ == '__main__':
    main()

"""Label-blind SCD point geometry from the corrected actual author schedule."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
import zipfile
from scd1_common import *


def run(root,out,budget):
    manifest=load_inputs(root)
    write(out/'CODE_BINDINGS.json',dict(python=sys.executable,source_sha256={name:sha(Path(__file__).with_name(name))
        for name in ('scd1_common.py','run_scd1_geometry.py','evaluate_scd1.py','test_scd1.py','prepare_scd1_inputs.py')}))
    plan=read(root/'INPUT_PATCH_PLAN.json')
    correction=read(root/'AUTHOR_DEDUPLICATION_CORRECTION.json')
    changes={(r['setting'],r['product_id']):r for r in correction['rows']}
    receipts={(r['setting'],r['product_id']):r for r in read(root/'SCENE_RECEIPTS.json')}
    assert len(changes)==len(receipts)==len(plan['rows'])==32
    metadata={r['product_id']:r for r in read(root/'member_grid_metadata.json')}
    coord_receipt=read(root/'coordinate_metadata_verification.json')
    assert coord_receipt['status']=='passed_coordinate_metadata_only' and coord_receipt['pending']==0
    assert coord_receipt['member_grid_metadata_sha256']==sha(root/'member_grid_metadata.json')
    coordinates=csvread(root/'coordinate_metadata.csv')
    assert len(coordinates)==len({r['point_id'] for r in coordinates})==9500
    assert set(metadata)=={r['product_id'] for r in coordinates} and len(metadata)==16
    assert len({r['group_id'] for r in coordinates})==10
    scores=score_points(root)
    support=read(root/'COMMON_SUPPORT_LOCK.json')
    private=read(root/'COMMON_SUPPORT_PRIVATE.json')['rows']
    common={r['point_id']:r['common_valid'] for r in private}
    assert len(private)==len(common)==9500 and set(common)==set(scores['O20'])
    assert sum(common.values())==support['common_valid']==9500
    windows={};checks=[]
    with zipfile.ZipFile(root/'point_scores_private.zip') as z:
        for row in plan['rows']:
            setting,pid=row['setting'],row['product_id'];key=(setting,pid)
            actual=dedup(row['actual_patch_coordinates'])
            assert len(actual)==len(row['actual_patch_coordinates'])==changes[key]['author_unique']
            assert row['product']['crs']==metadata[pid]['crs']
            for grid in ('grid10','grid20'):assert row['product'][grid]==metadata[pid][grid]
            g10,g20=metadata[pid]['grid10'],metadata[pid]['grid20']
            assert g10['ULX']==g20['ULX'] and g10['ULY']==g20['ULY']
            assert g10['XDIM']==10 and g10['YDIM']==-10 and g20['XDIM']==20 and g20['YDIM']==-20
            cpu=json.loads(z.read(setting+'/'+pid+'/CPU_POINT_RECONSTRUCTION.json'))
            assert actual==dedup(cpu['actual_patch_coordinates']) and len(cpu['actual_patch_coordinates'])==len(actual)
            assert cpu['coverage_and_point_gradient_exact'] and cpu['max_confidence_error']<=1e-5
            assert receipts[key]['calls']==2*len(actual) and receipts[key]['cpu_error']<=1e-5
            assert receipts[key]['score_sha256']==support['all_score_sha256'][setting+'/'+pid]
            size=500 if setting=='C20' else 1000
            grid=g20 if setting=='C20' else g10
            assert row['patch_size']==size
            assert all(0<=r0<r1<=grid['NROWS'] and 0<=c0<c1<=grid['NCOLS'] and r1-r0==c1-c0==size for r0,r1,c0,c1 in actual)
            windows[key]=actual
            checks.append(dict(setting=setting,product_id=pid,before=changes[key]['old'],after=len(actual),uncovered=0))
            budget.receipt()
    geometries=[]
    for c in coordinates:
        pid,point=c['product_id'],c['point_id'];m=metadata[pid]
        assert c['crs']==m['crs'] and c['group_id']==m['group_id']
        assert c['position_pending']=='False'
        row10,col10,row20,col20=[int(c[k]) for k in ('row10','col10','row20','col20')]
        assert row20==row10//2 and col20==col10//2
        for name,rr,cc in [('grid10',row10,col10),('grid20',row20,col20)]:
            g=m[name];x=float(c['native_x_m']);y=float(c['native_y_m'])
            assert math.floor((x-g['ULX'])/g['XDIM'])==cc and math.floor((y-g['ULY'])/g['YDIM'])==rr
            assert 0<=rr<g['NROWS'] and 0<=cc<g['NCOLS']
        ok=True
        for setting in SETTINGS:
            q=scores[setting][point]
            assert q['product_id']==pid and int(q['row20'])==row20 and int(q['col20'])==col20
            if setting!='O20':
                assert q['group_id']==c['group_id'] and int(q['row10'])==row10 and int(q['col10'])==col10
            ok=ok and all(q[k] for k in ('valid','coverage','author_valid'))
        assert bool(ok)==common[point]
        entry=dict(point_id=point,product_id=pid,group_id=c['group_id'],row10=row10,col10=col10,row20=row20,col20=col20,
                   subcell=str(row10%2)+str(col10%2),common_valid=bool(ok))
        for setting,rr,cc,res in [('C20',row20,col20,20),('H10',row10,col10,10)]:
            count,margin=geometry(rr,cc,windows[setting,pid],res)
            entry[setting+'_coverage_count']=count;entry[setting+'_best_margin_m']=margin
        geometries.append(entry)
    csvwrite(out/'point_geometry.csv',geometries)
    counts={gid:{layer:0 for layer in LAYERS} for gid in sorted({r['group_id'] for r in geometries})}
    for r in geometries:
        for layer in memberships(r):counts[r['group_id']][layer]+=1
    distributions={}
    for gid in ['all']+list(counts):
        rr=[r for r in geometries if gid=='all' or r['group_id']==gid]
        distributions[gid]={}
        for setting in ('C20','H10'):
            margins=sorted(r[setting+'_best_margin_m'] for r in rr)
            distributions[gid][setting]=dict(coverage_histogram=dict(collections.Counter(r[setting+'_coverage_count'] for r in rr)),
                margin_min_m=min(margins),margin_max_m=max(margins),margin_median_m=(margins[(len(margins)-1)//2]+margins[len(margins)//2])/2)
        distributions[gid]['coverage_cross_counts']=dict(collections.Counter(str(r['C20_coverage_count'])+'/'+str(r['H10_coverage_count']) for r in rr))
        distributions[gid]['margin_cross_counts']=dict(collections.Counter(str(r['C20_best_margin_m']<300)+'/'+str(r['H10_best_margin_m']<300) for r in rr))
    for check in checks:
        rr=[r for r in geometries if r['product_id']==check['product_id']]
        suffix='20' if check['setting']=='C20' else '10'
        check['coordinate_duplicates']=len(rr)-len({(r['row'+suffix],r['col'+suffix]) for r in rr})
        check['uncovered']=sum(r[check['setting']+'_coverage_count']==0 for r in rr)
    write(out/'GEOMETRY_CHECKS.json',dict(status='passed',points=9500,products=16,groups=10,
        coordinate_mapping_and_original_CRS_affine=True,actual_CPU_patch_plan_matches=True,rows=checks,
        no_reference_read=True,margin_is_scheduling_only_not_receptive_field_or_gradient=True))
    write(out/'GEOMETRY_DISTRIBUTIONS.json',distributions)
    write(out/'STRATA_LOCK.json',dict(geometry_sha256=sha(out/'point_geometry.csv'),input_manifest_sha256=sha(root/'INPUT_MANIFEST.json'),
        strata=LAYERS,group_layer_unlabelled_counts=counts,before_reference_join=True,reference_read=False,
        coverage='C20: 1,2,>=3',margin='C20 best_margin_m: <300,>=300',
        subcell='(row10%2,col10%2): 00,01,10,11',common_support=9500,score_sha256=support['all_score_sha256']))
    write(out/'INPUT_MANIFEST.json',manifest)
    write(out/'RESOURCE_RECEIPT.json',budget.receipt())
    status_table(out,'几何完成，等待参考连接')
    (out/'EXIT_CODE').write_text('0\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
    a.output_root.mkdir(parents=True,exist_ok=False)
    budget=Budget([a.input_root,a.output_root]);budget.receipt()
    try:run(a.input_root,a.output_root,budget)
    except Exception:
        write(a.output_root/'FAILURE.json',dict(error=traceback.format_exc(),new_model_calls=0))
        status_table(a.output_root,'几何身份或输入校验失败，详见执行端FAILURE.json','阻塞')
        raise


if __name__=='__main__':main()

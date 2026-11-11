"""Read-only shared cache inventory and budget, no labels/model calls/downloads."""
import hashlib,json,sys
from pathlib import Path
R=Path('/home/scv/Cloud-Adapter-light');S=Path('/home/scv/shared');B=S/'phase111_ocm_runtime'
def load(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    import numpy as np
    members=load(R/'src/research_results/phase122_20261015/MEMBER_INPUT_LOCK.json');assert len(members['fit'])==100 and len(members['calibration'])==53
    plan=load(S/'phase119_20261015/INPUT_PATCH_PLAN.json');dev=[r for r in plan['rows'] if r['pool']=='Catalogue'];pix=[r for r in plan['rows'] if r['pool']=='PixBox']
    assert len(dev)==32 and len(pix)==16
    records=[];missing=[]
    for split,rows in [('fit',members['fit']),('calibration',members['calibration']),('development',[r['group'] for r in dev])]:
        for g in rows:
            pid=g['product'];p=S/'phase117_20261014/scores'/('O_'+pid+'.npz')
            row=dict(split=split,group=g,cache=str(p),exists=p.is_file())
            if p.is_file():
                row.update(bytes=p.stat().st_size,sha256=sha(p))
                if split=='development':assert row['sha256']==next(x['old_score_sha256'] for x in dev if x['product_id']==pid)
                with np.load(p,allow_pickle=False) as z:
                    scores=z['scores'];assert np.isfinite(scores).all() and len(scores)==g['common_valid'];row['score_shape']=list(scores.shape)
                records.append(row)
            else:missing.append(row)
    pl=load(R/'src/research_results/phase112_20261014/PRE_EXECUTION_LOCK.json');seals=load(R/'src/research_results/phase112_20261014/SCORES_SEALED.json')['score_sha256'];pixrecords=[]
    for x in pix:
        pid=x['product_id'];root=B/'single_scene_recovery_retry_20261014' if pid==pl['reuse_product'] else S/'phase112_20261014/run'/pid
        p=root/'UNLABELLED_POINT_SCORES.json';assert p.is_file() and sha(p)==seals[pid]
        pixrecords.append(dict(product_id=pid,path=str(p),sha256=sha(p),points=len(load(p)['points'])))
    raw=S/'data/sentinel2_cloud_mask_catalogue_4172871/subscenes.zip';assert raw.stat().st_size==15197196473
    sys.path.insert(0,str(R/'src/tools'));from install_phase111_runtime import host_free
    result=dict(status='cache_inventory_complete',existing=records,missing=missing,PixBox=pixrecords,member_lock_sha256=sha(R/'src/research_results/phase122_20261015/MEMBER_INPUT_LOCK.json'),
        scope='metadata/index and E score payload/finite/length checks; no reference semantics, no neural calls',
        budget=dict(missing_scenes=len(missing),conservative_patches_per_cube=4,main_call_upper=8*len(missing),construction_allowance=33,all_calls_upper=8*len(missing)+33,
            score_storage_upper_bytes=(len(missing)+32)*1022*1022*5,features_storage_upper_bytes=100*8192*8*8,cap_calls=10000,cap_new_bytes=6000000000,cap_GPU_seconds=14400,host_free_bytes=host_free()),
        asset_status=load(B/'ASSET_VERIFICATION.json')['status'],runtime_status=load(B/'RUNTIME_IMPORT_ACCEPTANCE.json')['dependency_check'],raw_archive_bytes=raw.stat().st_size)
    print(json.dumps(result))
if __name__=='__main__':main()

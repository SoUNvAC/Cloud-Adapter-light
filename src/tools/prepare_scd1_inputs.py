"""Stage existing unlabelled SCD small files only; never copy PixBox references."""
import argparse
from pathlib import Path
import shutil
from scd1_common import read,sha,write,load_inputs,score_points


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
    a.output_root.mkdir(parents=True,exist_ok=False)
    r=a.repo;out=a.output_root;files={};sources={}
    repository=read(r/'src/research_results/phase118_20261014/REPOSITORY_DELIVERY_SHA256.json')
    small=read(r/'src/research_results/phase118_20261014/remote_delivery/SMALL_DELIVERY_SHA256.json')
    old103=read(r/'src/research_results/phase103_20261013/DELIVERY_SHA256.json')
    def copy(source,name,expected=None):
        if expected is not None:assert sha(source)==expected,str(source)
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        files[name]=sha(dest);sources[name]=str(source)
    for name in ('INPUT_PATCH_PLAN.json','AUTHOR_DEDUPLICATION_CORRECTION.json','SCENE_RECEIPTS.json','SCORES_SEALED.json','EXECUTION_PLAN_SEAL.json','CPU_ALL_SCENES_ACCEPTANCE.json'):
        copy(r/'src/research_results/phase118_20261014/remote_delivery'/name,name,small[name])
    for name in ('COMMON_SUPPORT_LOCK.json','EVALUATION_LOCK.json','per_group_results.csv','three_setting_summary.csv','FIXED_COMPARISONS.json'):
        rel='src/research_results/phase118_20261014/'+name;copy(r/rel,name,repository[rel])
    for name in ('coordinate_metadata_verification.json','member_grid_metadata.json'):
        copy(r/'src/research_results/phase103_20261013'/name,name,old103[name])
    coord=read(out/'coordinate_metadata_verification.json')
    copy(r/'data/pixbox_s2_cmix_5036991/phase103_20261013/coordinate_metadata.csv','coordinate_metadata.csv',coord['coordinate_metadata_sha256'])
    support=read(out/'COMMON_SUPPORT_LOCK.json')
    copy(r/'data/pixbox_s2_cmix_5036991/phase118_20261014/COMMON_SUPPORT_PRIVATE.json','COMMON_SUPPORT_PRIVATE.json',support['point_mask_sha256'])
    receipt=read(r/'src/research_results/phase118_20261014/remote_delivery/DELIVERY_RECEIPT.json')
    copy(r/'outputs/phase118/phase118_20261014/point_scores_private.zip','point_scores_private.zip',receipt['private_scores_archive_sha256'])
    copy(r/'outputs/phase112/phase112_20261014/SCORES_SEALED.json','O20_SCORES_SEALED.json')
    for key,h in support['all_score_sha256'].items():
        if key.startswith('O20/'):
            pid=key.split('/')[1]
            copy(r/'data/pixbox_s2_cmix_5036991/phase112_20261014/omni'/pid/'UNLABELLED_POINT_SCORES.json',key+'/UNLABELLED_POINT_SCORES.json',h)
    write(out/'INPUT_MANIFEST.json',dict(files=files,original_sources=sources,reference_uploaded=False,
        reference_sha256=read(out/'EVALUATION_LOCK.json')['original_reference_sha256'],
        source='accepted Phase118 corrected plan and final sealed scores; no re-inference'))
    load_inputs(out);score_points(out)
    print('Unlabelled inputs ready:',len(files),'files',sum(x.stat().st_size for x in out.rglob('*') if x.is_file()),'bytes')


if __name__=='__main__':main()

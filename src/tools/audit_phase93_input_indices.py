"""Read-only metadata check against pre-existing sealed input indices; no pixels."""
import argparse
import hashlib
import json
from pathlib import Path

SHARED=Path('/home/scv/shared')


def main(stage):
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    old=SHARED/'phase92_20261012/delivery'
    assert digest(old/'DELIVERY_SHA256.json')=='abb2bea0612be87bc66ba78eef3ddad15b9099a7d5172f1d5df306fcf4a8f195'
    for n,h in read(old/'DELIVERY_SHA256.json').items():assert digest(old/n)==h,n
    lock=read(old/'input_code_score_lock.json');chain=lock['phase90_chain'];receipts={}
    development=SHARED/'data/sentinel2_cloud_mask_catalogue_4172871/phase65d_20261008_recovery01/development_val'
    for n,h in lock['development_cache_index_sha256'].items():
        path=development/('index_'+n+'.json');assert digest(path)==h
        receipts[str(path)]=h
    recovered=SHARED/'phase89_recovered_20261011'
    runtime=recovered/'delivery/input_and_runtime_index.json'
    assert digest(runtime)==chain['phase89_runtime_index_sha256']
    assert digest(recovered/'recovery_lock.json')==chain['phase89_recovery_lock_sha256']
    original=read(runtime)
    for n in ['Source','MsRE']:
        path=recovered/('scores_'+n)/'index.json'
        assert read(path)==original['inference'][n],n
        receipts[str(path)]=digest(path)
    assert read(recovered/'prepared/index.json')==original['prepared']
    receipts[str(recovered/'prepared/index.json')]=digest(recovered/'prepared/index.json')
    out=dict(stage=stage,all_original_indices_unchanged=True,index_sha256=receipts,
             phase89_runtime_index_sha256=digest(runtime),no_pixel_or_model_access=True)
    path=SHARED/'phase93_20261012'/('INPUT_INDEX_'+stage.upper()+'.json')
    with path.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['before','after']);main(parser.parse_args().stage)

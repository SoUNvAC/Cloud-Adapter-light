"""Phase121 inherits accepted Phase120 recipe, with a separate immutable lock."""
from phase120_common import (BASE,REPO,OLD,load,write,sha,ram,coordinates,image,
    catalog_image,verify_native_and_cache,original_pix_score,decoder_names,digest_state,
    objectives,setup as setup120)
from pathlib import Path
ROOT=Path('/home/scv/shared/phase121_20261015')
LOCK=REPO/'src/research_results/phase121_20261015/PRE_EXECUTION_LOCK.json'
def setup():
    setup120()
    cfg=load(LOCK)
    for p,h in cfg['source_sha256'].items():assert sha(REPO/p)==h,p
    return cfg

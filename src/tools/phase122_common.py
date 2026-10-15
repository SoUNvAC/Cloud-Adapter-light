"""Phase122 identities; inherited accepted encoder/CE/input implementation."""
from pathlib import Path
from phase121_common import (BASE,REPO,OLD,load,write,sha,ram,coordinates,image,catalog_image,
    verify_native_and_cache,original_pix_score,decoder_names,digest_state,objectives,setup as setup121)
ROOT=Path('/home/scv/shared/phase122_20261015')
LOCK=REPO/'src/research_results/phase122_20261015/PRE_EXECUTION_LOCK.json'
def setup():
    setup121();cfg=load(LOCK)
    for p,h in cfg['source_sha256'].items():assert sha(REPO/p)==h,p
    return cfg

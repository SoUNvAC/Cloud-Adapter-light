"""Phase123: reuse accepted Phase122 implementation, identities and artifacts."""
from phase122_common import *
from phase122_common import setup as accepted_setup
PREVIOUS=ROOT
ROOT=PREVIOUS.parent/'phase123_20261016'
LOCK=REPO/'src/research_results/phase123_20261016/PRE_EXECUTION_LOCK.json'
def setup():
    accepted_setup();cfg=load(LOCK)
    for name,h in cfg['source_sha256'].items():assert sha(REPO/name)==h,name
    for name,h in cfg['previous_shared_sha256'].items():assert sha(PREVIOUS/name)==h,name
    assert (PREVIOUS/'EXIT_CODE').read_text().strip()=='0'
    assert sha(REPO/'src/work_dirs/phase65a/split_lock.json')==cfg['original_split_sha256']
    return cfg

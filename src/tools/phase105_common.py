"""Fixed PixBox evaluation constants; no training or label access on import."""
from phase104_common import read,write,sha,now,csvwrite,REPO,MODEL_SHA,BANDS,COORD_SHA
from pathlib import Path
import hashlib

RESULT=REPO/'src/research_results/phase105_20261013'
SHARED=Path('/home/scv/shared')
ROOT=SHARED/'data/pixbox_s2_cmix_5036991/phase105_20261013'
INPUT=SHARED/'data/pixbox_s2_cmix_5036991/phase104_20261013'
SOURCE=REPO/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'
SOURCE_SHA='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
MODELS=SHARED/'phase93_20261012/delivery'
PROTOCOL=REPO/'src/research_plans/PHASE105_PIXBOX_FIXED_EVALUATION_20261013.md'
COORDINATES=SHARED/'phase103_positions_20261013/coordinate_metadata.csv'

def key(w):return f"{w['product_id']}|{w['y']}|{w['x']}"
def repeats(windows):
    return sorted(windows,key=lambda w:hashlib.sha256(('PHASE105_REPEAT|'+key(w)).encode()).hexdigest())[:2]
def code():
    return {str(p.relative_to(REPO)).replace('\\','/'):sha(p) for p in sorted((REPO/'src/tools').glob('*phase105*')) if p.is_file()}

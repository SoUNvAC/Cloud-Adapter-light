"""Fixed paths and receipts for one nonconfirmatory Omni lake case."""
import hashlib,json,os
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
LOCAL=REPO/'outputs/phase200/phase200_20261017'
ROOT=Path('/home/scv/shared/phase200_20261017')
BASE=Path('/home/scv/shared/phase111_ocm_runtime')
LOCK=REPO/'src/research_results/phase200_20261017/EXECUTION_PLAN.json'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');os.replace(tmp,p)
def setup():
    for k in ['HF_HOME','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR','TMPDIR']:
        p=BASE/'controlled_caches'/k.lower();assert p.is_dir();os.environ[k]=str(p)
    os.environ.update(HF_HUB_OFFLINE='1',GDAL_NUM_THREADS='1',OPJ_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
    import sys
    sys.path.insert(0,str(BASE/'assets/vendor'))

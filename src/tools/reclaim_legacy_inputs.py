"""Explicit legacy *input-cache* cleanup, preserving original data, scores and models."""
import argparse,ast,collections,datetime,hashlib,json,math,os,re,struct,subprocess,sys,zipfile
from pathlib import Path
REPO=Path('/home/scv/Cloud-Adapter-light')
CAT=Path('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871')
EVIDENCE=Path('/home/scv/shared/storage_cleanup_20261017')
SUFFIXES=('_rgb.npy','_RGB.npy','_spectral.npy','_B.npy')
PRODUCT=re.compile(r'^(S2[AB]_MSIL1C_\d{8}T\d{6}_N\d{4}_R\d{3}_T\d{2}[A-Z]{3}_\d{8}T\d{6})')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
def selected(relative,size):
    p=Path(relative)
    if len(p.parts)<2 or p.is_absolute() or '..' in p.parts or size<8*1024**2:return False
    m=re.fullmatch(r'phase(\d+)(?:\D.*)?',p.parts[0])
    if not m or not 65<=int(m[1])<=100:return False
    product=PRODUCT.match(p.name)
    return bool(product and p.name[len(product[1]):] in SUFFIXES)
def npy_header(p):
    with p.open('rb') as f:
        assert f.read(6)==b'\x93NUMPY';version=f.read(2)
        assert version in (b'\x01\x00',b'\x02\x00',b'\x03\x00')
        n=struct.unpack('<H' if version[0]==1 else '<I',f.read(2 if version[0]==1 else 4))[0];assert n<=65536
        raw=f.read(n);h=ast.literal_eval(raw.decode('latin1' if version[0]<3 else 'utf-8'));offset=f.tell()
    assert h['descr'] in ('<f4','<f8','|f4','|f8');shape=h['shape']
    assert shape in ((1022,1022,3),(3,1022,1022)) or (len(shape)==2 and 0<shape[0]<=1022**2 and shape[1]==3)
    assert offset+math.prod(shape)*int(h['descr'][-1])==p.stat().st_size
    return dict(shape=list(shape),dtype=h['descr'],header_sha256=hashlib.sha256(raw).hexdigest())
def safe(p):
    assert p==p.resolve() and p.is_relative_to(CAT) and not p.is_symlink(),str(p)
    return p
def snapshot(p):
    s=p.lstat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,links=s.st_nlink,allocated_bytes=s.st_blocks*512)
def active():
    jobs=[]
    for line in subprocess.check_output(['ps','-eo','pid,comm'],text=True).splitlines()[1:]:
        pid,comm=line.split(None,1)
        if int(pid)!=os.getpid() and (comm.startswith(('python','torchrun')) or comm in ('scp','rsync','curl','wget','tar','rclone')):jobs.append(dict(PID=int(pid),comm=comm))
    assert not jobs,dict(active_jobs=jobs)
def originals(full_hash=False):
    p=CAT/'download_status.json';s=json.loads(p.read_text());assert s['status']=='all_files_verified'
    result={}
    for name in ('subscenes.zip','masks.zip'):
        f=CAT/name;assert f==f.resolve() and not f.is_symlink()
        assert f.stat().st_size==s['verified_files'][name]['bytes']
        if full_hash:assert digest(f)==s['verified_files'][name]['sha256'],name
        result[name]=dict(**snapshot(f),expected_sha256=s['verified_files'][name]['sha256'])
    return result
def plan():
    active();EVIDENCE.mkdir(exist_ok=True);source=originals();files=[];rejected=0
    with zipfile.ZipFile(CAT/'subscenes.zip') as z:members={Path(v.filename).stem for v in z.infolist() if v.filename.startswith('subscenes/') and v.filename.endswith('.npy')}
    for root in CAT.iterdir():
        if not root.is_dir() or root.is_symlink():continue
        for directory,folders,names in os.walk(root,followlinks=False):
            folders[:]=[n for n in folders if not (Path(directory)/n).is_symlink()]
            for name in names:
                p=Path(directory)/name
                if not selected(p.relative_to(CAT),p.lstat().st_size):continue
                safe(p);state=snapshot(p)
                if state['links']!=1:continue
                product=PRODUCT.match(name)[1];assert product in members,product
                try:header=npy_header(p)
                except AssertionError:rejected+=1;continue
                files.append(dict(path=str(p),phase=root.name,product=product,stat=state,header=header))
    total=sum(v['stat']['bytes'] for v in files);by=collections.Counter()
    for v in files:by[v['phase']]+=v['stat']['bytes']
    p=EVIDENCE/'INPUT_CACHE_PLAN.json';write(p,dict(authorization='user requested space reclamation after switch to Omni; only deterministic legacy input caches',
        source_sha256=digest(Path(__file__)),protected_originals=source,rows=files,total_bytes=total,allocated_bytes=sum(v['stat']['allocated_bytes'] for v in files),
        by_phase=dict(by),rejected_headers=rejected,preserve='all original datasets, prediction scores, weights/models, runtime, JSON/CSV/logs, all phases109+ including Omni/200'))
    print(json.dumps(dict(plan=str(p),plan_sha256=digest(p),files=len(files),bytes=total,GiB=total/1024**3,by_phase=dict(by),preserve='raw data / scores / models / runtime / all small evidence')))
def apply(path):
    assert path==EVIDENCE/'INPUT_CACHE_PLAN.json' and path==path.resolve()
    a=json.loads(path.read_text());assert digest(Path(__file__))==a['source_sha256'];active()
    # Verify the complete original archive once before deleting any cache payload.
    before=originals(full_hash=True);assert before==a['protected_originals'];free=os.statvfs(CAT).f_bavail*os.statvfs(CAT).f_frsize
    for row in a['rows']:
        p=safe(Path(row['path']));assert selected(p.relative_to(CAT),row['stat']['bytes'])
        assert snapshot(p)==row['stat'] and npy_header(p)==row['header'],str(p)
    log=EVIDENCE/'INPUT_CACHE_DELETED.jsonl';removed=0;allocated=0;count=0
    with log.open('x',encoding='utf-8') as f:
        for row in a['rows']:
            if count%50==0:active()
            p=safe(Path(row['path']));assert snapshot(p)==row['stat'];p.unlink()
            removed+=row['stat']['bytes'];allocated+=row['stat']['allocated_bytes'];count+=1
            f.write(json.dumps(dict(path=str(p),bytes=row['stat']['bytes'],header_sha256=row['header']['header_sha256']))+'\n');f.flush()
    assert originals()==before
    after=os.statvfs(CAT).f_bavail*os.statvfs(CAT).f_frsize
    result=dict(status='completed',deleted_files=count,deleted_logical_bytes=removed,deleted_allocated_bytes=allocated,GiB=removed/1024**3,
        filesystem_available_before=free,filesystem_available_after=after,original_archives_SHA_verified=True,original_archives_unchanged=True,
        deleted_prediction_scores=False,deleted_models=False,deleted_raw_data=False,deleted_runtime=False,work_dirs_untouched=True,
        Windows_VHD_compaction=False,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=digest(path))
    write(EVIDENCE/'INPUT_CACHE_COMPLETED.json',result);print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');p.add_argument('--apply',type=Path);a=p.parse_args()
    assert (a.plan or a.apply) and not (a.plan and a.apply)
    if a.plan:plan()
    else:apply(a.apply)

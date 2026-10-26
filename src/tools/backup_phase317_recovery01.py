"""Diagnosed EOF recovery: immutable original manifest, SHA-checked prefix, bounded missing-member streams."""
import base64,gzip,subprocess,tarfile,traceback
from pathlib import PurePosixPath
from phase317_common import *
from transfer_phase317 import HashReader,HashWriter
REC=ROOT/'backup_recovery01'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','CheckHostIP=no','-o','UpdateHostKeys=no','-o','ConnectTimeout=30','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','gzs']
PREFIX='source /home/scv/miniconda3/etc/profile.d/conda.sh && conda activate cloud-lite-pt210 && cd /home/scv/Cloud-Adapter-light && '
def manifest_local():
 h=load(LOCAL/'REMOTE_STREAM_HEADER.json');p=LOCAL/'remote_backup/BACKUP_FILES.json';assert sha(p)==h['manifest_sha256'];return h,load(p)
def plan():
 assert (LOCAL/'BACKUP_FAILURE.json').exists() and not (LOCAL/'RECOVERY01_PLAN.json').exists()
 h,m=manifest_local();verified={};missing=[]
 for n,e in sorted(m.items()):
  p=LOCAL/'remote_backup'/n
  if p.exists():assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],n;verified[n]=e['sha256']
  else:missing.append(n)
 batches=[];current=[];size=0
 for n in missing:
  if current and (size+m[n]['bytes']>128000000 or len(current)>=256):batches.append(current);current=[];size=0
  current.append(n);size+=m[n]['bytes']
 if current:batches.append(current)
 write(LOCAL/'RECOVERY01_PLAN.json',dict(original_manifest_sha256=h['manifest_sha256'],original_header=h,prefix_verified_files=len(verified),prefix_verified_sha256=verified,missing_files=len(missing),missing_raw_bytes=sum(m[n]['bytes'] for n in missing),batches=batches,max_batch_raw_bytes=128000000,first_stream_complete=False,first_stream_SHA='unavailable/incomplete; never marked verified',preserve_original_failure=True,partials=[dict(file=p.relative_to(LOCAL).as_posix(),bytes=p.stat().st_size) for p in (LOCAL/'remote_backup').rglob('*.partial')]))
 print(json.dumps(dict(verified_prefix=len(verified),missing=len(missing),bytes=sum(m[n]['bytes'] for n in missing),batches=len(batches))))
def inspect():
 assert load(ROOT/'STATUS.json')['status']=='completed' and (ROOT/'STREAM_LOCK.json').exists();assert not REC.exists()
 active=[s for s in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if 'python' in s and 'transfer_phase317.py remote' in s];assert not active,active
 m=load(ROOT/'BACKUP_FILES.json')
 for n,e in m.items():assert (ROOT/n).stat().st_size==e['bytes'] and sha(ROOT/n)==e['sha256'],n
 REC.mkdir();shutil.copyfile(Path(__file__),REC/'backup_phase317_recovery01.py')
 write(REC/'INSPECTION.json',dict(status='passed',original_manifest_sha256=sha(ROOT/'BACKUP_FILES.json'),original_files_unchanged=len(m),old_stream_active=False,original_completed_receipt_exists=(ROOT/'STREAM_RECEIPT.json').exists(),code_sha256=sha(Path(__file__)),epoch=time.time(),diagnosis='localEOF in firststream; subsequent independentSSH kex closures observed; exact originalremote exception unavailable; no compressionformat assertion invented'))
 print('Recovery01 original snapshot SHA verified; no old stream active; only missing members may transfer')
def remote_batch(index):
 assert load(REC/'INSPECTION.json')['status']=='passed';request=json.loads(sys.stdin.buffer.read());m=load(ROOT/'BACKUP_FILES.json');h=sha(ROOT/'BACKUP_FILES.json');assert h==request['manifest_sha256'];names=request['names'];assert len(set(names))==len(names) and all(n in m for n in names)
 assert sum(m[n]['bytes'] for n in names)<=128000000 and len(names)<=256
 lock=REC/(index+'_REQUEST.json');assert not lock.exists(),'existing batch; diagnose, no blind repeat';write(lock,request)
 for n in names:assert sha(ROOT/n)==m[n]['sha256'] and (ROOT/n).stat().st_size==m[n]['bytes'],n
 header=dict(index=index,manifest_sha256=h,files=len(names),raw_bytes=sum(m[n]['bytes'] for n in names));sys.stdout.buffer.write((json.dumps(header)+'\n').encode());sys.stdout.buffer.flush();w=HashWriter(sys.stdout.buffer)
 with gzip.GzipFile(fileobj=w,mode='wb',compresslevel=1,mtime=0) as gz:
  with tarfile.open(fileobj=gz,mode='w|') as tar:
   for n in names:tar.add(ROOT/n,arcname=n,recursive=False)
 w.flush();write(REC/(index+'_RECEIPT.json'),dict(**header,stream_bytes=w.bytes,stream_sha256=w.hash.hexdigest(),status='completed'))
def batches():
 plan=load(LOCAL/'RECOVERY01_PLAN.json');h,m=manifest_local();assert plan['original_manifest_sha256']==h['manifest_sha256'];target=LOCAL/'remote_backup';logs=LOCAL/'recovery01';logs.mkdir(exist_ok=True)
 for i,names in enumerate(plan['batches'],1):
  index=f'batch{i:03d}';checkpoint=logs/(index+'_CHECKPOINT.json');assert not checkpoint.exists() and not (logs/(index+'_FAILURE.json')).exists(),'inspect prior batch instead of rerunning'
  for n in names:assert not (target/n).exists(),'plan not current; inspect alreadyverified member'
  try:
   with (logs/(index+'_STDERR.log')).open('xb') as err:
    p=subprocess.Popen(SSH+[PREFIX+'python -B src/tools/backup_phase317_recovery01.py remote_batch '+index],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err);p.stdin.write(json.dumps(dict(manifest_sha256=h['manifest_sha256'],names=names)).encode());p.stdin.close();line=p.stdout.readline();assert line,'SSH before header; inspect saved STDERR';head=json.loads(line);assert head['index']==index and head['manifest_sha256']==h['manifest_sha256'];reader=HashReader(p.stdout);count=0
    with gzip.GzipFile(fileobj=reader,mode='rb') as gz:
     with tarfile.open(fileobj=gz,mode='r|') as tar:
      for item in tar:
       n=item.name;pp=PurePosixPath(n);assert item.isfile() and n in names and not pp.is_absolute() and '..' not in pp.parts and ':' not in n and '\\' not in n
       dest=target/n;assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_name(dest.name+'.partial.recovery01_'+index)
       with temp.open('xb') as f:
        stream=tar.extractfile(item)
        for b in iter(lambda:stream.read(1048576),b''):f.write(b)
       assert temp.stat().st_size==m[n]['bytes'] and sha(temp)==m[n]['sha256'],n;temp.rename(dest);count+=1
     while gz.read(1048576):pass
    code=p.wait();assert code==0 and count==len(names)
   cp=dict(**head,stream_bytes=reader.bytes,stream_sha256=reader.hash.hexdigest(),all_members_SHA_verified=True,process_exit=0);write(checkpoint,cp)
   receipt=None
   for attempt in range(3):
    r=subprocess.run(SSH+[PREFIX+'cat /home/scv/shared/phase317_20261026/backup_recovery01/'+index+'_RECEIPT.json'],capture_output=True)
    if r.returncode==0:receipt=json.loads(r.stdout);break
    write(logs/(index+f'_RECEIPT_SSH_FAILURE{attempt+1}.json'),dict(exit=r.returncode,stderr=r.stderr.decode(errors='replace')))
   assert receipt is not None,'only receipt missing; checkpoint preserved, never restream this batch'
   assert receipt['stream_bytes']==cp['stream_bytes'] and receipt['stream_sha256']==cp['stream_sha256'];write(logs/(index+'_VERIFIED.json'),dict(local=cp,remote=receipt));print('Recovery01 verified',index,len(names),head['raw_bytes'],flush=True)
  except BaseException:write(logs/(index+'_FAILURE.json'),dict(traceback=traceback.format_exc(),action='preserve files/temp/checkpoint/STDERR; diagnose; do not repeat batch'));raise
def finish():
 h,m=manifest_local();plan=load(LOCAL/'RECOVERY01_PLAN.json');receipts=[]
 for i,names in enumerate(plan['batches'],1):
  r=load(LOCAL/'recovery01'/f'batch{i:03d}_VERIFIED.json');assert r['local']['stream_sha256']==r['remote']['stream_sha256'] and r['local']['all_members_SHA_verified'];receipts.append(r)
 for n,e in m.items():assert (LOCAL/'remote_backup'/n).stat().st_size==e['bytes'] and sha(LOCAL/'remote_backup'/n)==e['sha256'],n
 result=subprocess.run(SSH+[PREFIX+'python -B src/tools/backup_phase317_recovery01.py evidence'],capture_output=True,check=True);ev=json.loads(result.stdout);dest=LOCAL/'remote_backup/backup_recovery01';dest.mkdir(exist_ok=True);metadata_sha={}
 for n,e in ev.items():
  assert Path(n).name==n and ':' not in n;data=base64.b64decode(e['base64']);assert hashlib.sha256(data).hexdigest()==e['sha256'];p=dest/n;assert not p.exists();p.write_bytes(data);metadata_sha[n]=e['sha256']
 index=dict(original_manifest_sha256=h['manifest_sha256'],recovery_plan_sha256=sha(LOCAL/'RECOVERY01_PLAN.json'),receipts=receipts,prefix_files=plan['prefix_verified_files'],all_snapshot_files=len(m),all_member_SHA_verified=True,completed_recovery_streams_SHA_verified=True,first_stream_complete=False,first_stream_SHA='unavailable/incomplete',preserved_failure_sha256=sha(LOCAL/'BACKUP_FAILURE.json'),recovery_metadata_SHA=metadata_sha)
 write(LOCAL/'RECOVERY01_INTEGRITY_INDEX.json',index);b=dict(**h,status='completed_via_verified_missing_member_recovery',backup_mode='member_checked_prefix_plus_SHA_verified_segmented_suffix',all_member_SHA_verified=True,whole_compressed_stream_SHA_verified=False,completed_recovery_streams_SHA_verified=True,first_stream_complete=False,recovery_batches=len(receipts),recovery_stream_bytes=sum(r['local']['stream_bytes'] for r in receipts),manifest_sha256=h['manifest_sha256'],integrity_index_sha256=sha(LOCAL/'RECOVERY01_INTEGRITY_INDEX.json'));write(LOCAL/'BACKUP_RECEIPT.json',b);print(json.dumps(b))
def evidence():
 assert len(list(REC.glob('*_RECEIPT.json')))==len(list(REC.glob('*_REQUEST.json')))
 out={p.name:dict(bytes=p.stat().st_size,sha256=sha(p),base64=base64.b64encode(p.read_bytes()).decode()) for p in REC.iterdir() if p.is_file()};assert sum(e['bytes'] for e in out.values())<=5000000;print(json.dumps(out))
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='remote_batch':remote_batch(sys.argv[2])
 else:{'plan':plan,'inspect':inspect,'batches':batches,'finish':finish,'evidence':evidence}[mode]()

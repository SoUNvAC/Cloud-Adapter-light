"""Full member+compressed-stream SHA backup, using gzip.GzipFile/tar w|; no large archive."""
import gzip,subprocess,tarfile,traceback
from pathlib import PurePosixPath
from phase317_common import *
class HashWriter:
 def __init__(self,f):self.f=f;self.hash=hashlib.sha256();self.bytes=0
 def write(self,b):self.hash.update(b);self.bytes+=len(b);return self.f.write(b)
 def flush(self):self.f.flush()
class HashReader:
 def __init__(self,f):self.f=f;self.hash=hashlib.sha256();self.bytes=0
 def read(self,n=-1):
  b=self.f.read(n);self.hash.update(b);self.bytes+=len(b);return b
def remote():
 assert load(ROOT/'STATUS.json')['status']=='completed';assert not (ROOT/'STREAM_LOCK.json').exists(),'inspect existingstream; never repeat blindly'
 write(ROOT/'STREAM_LOCK.json',dict(epoch=time.time(),status='started',action='onefullstream; failurepreserved'))
 files=sorted(p for p in ROOT.rglob('*') if p.is_file());raw=sum(p.stat().st_size for p in files);assert 2*raw+1000000000<=16000000000
 manifest={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files};write(ROOT/'BACKUP_FILES.json',manifest)
 header=dict(files=len(files)+1,raw_bytes=raw,manifest_sha256=sha(ROOT/'BACKUP_FILES.json'),projected_two_hosts_with_summary=2*raw+1000000000,no_persistent_archive=True);sys.stdout.buffer.write((json.dumps(header)+'\n').encode());sys.stdout.buffer.flush();writer=HashWriter(sys.stdout.buffer)
 with gzip.GzipFile(fileobj=writer,mode='wb',compresslevel=1,mtime=0) as gz:
  with tarfile.open(fileobj=gz,mode='w|') as tar:
   tar.add(ROOT/'BACKUP_FILES.json',arcname='BACKUP_FILES.json',recursive=False)
   for p in files:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
 writer.flush();write(ROOT/'STREAM_RECEIPT.json',dict(**header,stream_bytes=writer.bytes,stream_sha256=writer.hash.hexdigest(),status='completed'))
def receive():
 LOCAL.mkdir(parents=True,exist_ok=True);assert not any((LOCAL/n).exists() for n in ['remote_backup','BACKUP_RECEIPT.json','BACKUP_FAILURE.json'])
 ssh=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','CheckHostIP=no','-o','UpdateHostKeys=no','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','gzs']
 prefix='source /home/scv/miniconda3/etc/profile.d/conda.sh && conda activate cloud-lite-pt210 && cd /home/scv/Cloud-Adapter-light && '
 try:
  p=subprocess.Popen(ssh+[prefix+'python -B src/tools/transfer_phase317.py remote'],stdout=subprocess.PIPE,stderr=subprocess.PIPE);line=p.stdout.readline()
  if not line:raise RuntimeError('SSH beforeheader: '+p.stderr.read().decode(errors='replace')+' exit='+str(p.wait()))
  h=json.loads(line);write(LOCAL/'REMOTE_STREAM_HEADER.json',h);reader=HashReader(p.stdout);target=LOCAL/'remote_backup';target.mkdir();count=0;manifest=None
  with gzip.GzipFile(fileobj=reader,mode='rb') as gz:
   with tarfile.open(fileobj=gz,mode='r|') as tar:
    for m in tar:
     name=PurePosixPath(m.name);assert m.isfile() and not name.is_absolute() and '..' not in name.parts and '\\' not in m.name and ':' not in m.name
     dest=target/str(name);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name(dest.name+'.partial');assert not dest.exists()
     with tmp.open('xb') as f:
      stream=tar.extractfile(m)
      for b in iter(lambda:stream.read(1048576),b''):f.write(b)
     if manifest is None:assert m.name=='BACKUP_FILES.json' and sha(tmp)==h['manifest_sha256'];manifest=load(tmp)
     else:assert sha(tmp)==manifest[m.name]['sha256'] and tmp.stat().st_size==manifest[m.name]['bytes'],m.name
     tmp.rename(dest);count+=1
     if count%100==0:print('Phase317 backupverified',count,flush=True)
   while gz.read(1048576):pass
  stderr=p.stderr.read().decode(errors='replace');assert p.wait()==0,stderr;assert count==h['files']==len(manifest)+1
  write(LOCAL/'RECEIVE_STREAM_CHECKPOINT.json',dict(files=count,manifest_sha256=h['manifest_sha256'],stream_sha256=reader.hash.hexdigest(),stream_bytes=reader.bytes,all_member_SHA_verified=True,remote_process_exit=0,action='if finalreceiptSSHfails, fetchreceipt only; no repeatedstreamneeded'))
  result=subprocess.run(ssh+[prefix+'cat /home/scv/shared/phase317_20261026/STREAM_RECEIPT.json'],capture_output=True,check=True);receipt=json.loads(result.stdout);assert receipt['stream_sha256']==reader.hash.hexdigest() and receipt['stream_bytes']==reader.bytes and receipt['manifest_sha256']==h['manifest_sha256']
  receipt.update(all_member_SHA_verified=True,whole_compressed_stream_SHA_verified=True);write(target/'STREAM_RECEIPT.json',receipt);write(LOCAL/'BACKUP_RECEIPT.json',receipt);print(json.dumps(receipt))
 except BaseException:write(LOCAL/'BACKUP_FAILURE.json',dict(traceback=traceback.format_exc(),action='preserve members/partial/header/remote streamlock; inspect before independentrecovery, never repeat receiveblindly'));raise
if __name__=='__main__':{'remote':remote,'receive':receive}[sys.argv[1]]()

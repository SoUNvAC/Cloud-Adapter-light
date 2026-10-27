"""Manifest-pinned full backup; framed gzip segments, per-segment and total payload SHA.
No persistent archive. A failure preserves checkpoints/partials and never retries data.
"""
import base64,gzip,struct,subprocess,tarfile,traceback
from pathlib import PurePosixPath
from phase320_common import *
B=ROOT/'backup_transport'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','CheckHostIP=no','-o','UpdateHostKeys=no','-o','ConnectTimeout=30','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','gzs']
PREFIX='source /home/scv/miniconda3/etc/profile.d/conda.sh && conda activate cloud-lite-pt210 && cd /home/scv/Cloud-Adapter-light && '
def snapshot():
 assert load(ROOT/'STATUS.json')['status']=='completed' and not B.exists();assert budget_bytes()<=1073741824
 B.mkdir();shutil.copyfile(Path(__file__),B/'transfer_phase320.py')
 files=sorted(p for p in ROOT.rglob('*') if p.is_file() and not p.is_relative_to(B));m={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files};write(B/'BACKUP_FILES.json',m)
 raw=sum(e['bytes'] for e in m.values());write(B/'SNAPSHOT.json',dict(status='sealed',files=len(m),raw_bytes=raw,manifest_sha256=sha(B/'BACKUP_FILES.json'),projected_two_hosts_with_summary=2*raw+100000000,no_persistent_archive=True,epoch=time.time()))
 print(json.dumps(dict(header=load(B/'SNAPSHOT.json'),manifest_base64=base64.b64encode((B/'BACKUP_FILES.json').read_bytes()).decode())))
class FramedWriter:
 def __init__(self,f,whole):self.f=f;self.hash=hashlib.sha256();self.bytes=0;self.whole=whole
 def write(self,b):
  if not b:return 0
  self.f.write(struct.pack('>I',len(b)));self.f.write(b);self.hash.update(b);self.whole.update(b);self.bytes+=len(b);return len(b)
 def flush(self):self.f.flush()
def exact(f,n):
 out=bytearray()
 while len(out)<n:
  b=f.read(n-len(out))
  if not b:raise EOFError('transport closed inside frame; preserve completed checkpoints, do not restream')
  out.extend(b)
 return bytes(out)
class FramedReader:
 def __init__(self,f,whole):self.f=f;self.whole=whole;self.hash=hashlib.sha256();self.bytes=0;self.buffer=bytearray();self.eof=False
 def read(self,n=-1):
  while not self.eof and (n<0 or len(self.buffer)<n):
   size=struct.unpack('>I',exact(self.f,4))[0]
   if size==0:self.eof=True;break
   assert size<=16777216;b=exact(self.f,size);self.hash.update(b);self.whole.update(b);self.bytes+=size;self.buffer.extend(b)
  size=len(self.buffer) if n<0 else min(n,len(self.buffer));b=bytes(self.buffer[:size]);del self.buffer[:size];return b
def evidence():
 out={p.name:dict(sha256=sha(p),base64=base64.b64encode(p.read_bytes()).decode()) for p in B.iterdir() if p.is_file()};assert sum(p.stat().st_size for p in B.iterdir() if p.is_file())<10000000;return out
def stream():
 request=json.loads(sys.stdin.buffer.read());assert request['manifest_sha256']==sha(B/'BACKUP_FILES.json') and not (B/'STREAM_LOCK.json').exists();write(B/'STREAM_LOCK.json',dict(epoch=time.time(),request=request,policy='unique stream; on failure inspect members/checkpoints and receipts, no automatic retries'))
 m=load(B/'BACKUP_FILES.json');whole=hashlib.sha256();total=0;receipts=[];flat=[n for names in request['batches'] for n in names];assert len(flat)==len(set(flat))==len(m) and set(flat)==set(m)
 for i,names in enumerate(request['batches'],1):
  assert len(names)<=256 and sum(m[n]['bytes'] for n in names)<=128000000
  for n in names:assert sha(ROOT/n)==m[n]['sha256'] and (ROOT/n).stat().st_size==m[n]['bytes'],n
  head=dict(index=i,files=len(names),raw_bytes=sum(m[n]['bytes'] for n in names),manifest_sha256=request['manifest_sha256']);sys.stdout.buffer.write((json.dumps(head)+'\n').encode());sys.stdout.buffer.flush();w=FramedWriter(sys.stdout.buffer,whole)
  with gzip.GzipFile(fileobj=w,mode='wb',compresslevel=1,mtime=0) as gz:
   with tarfile.open(fileobj=gz,mode='w|') as tar:
    for n in names:tar.add(ROOT/n,arcname=n,recursive=False)
  w.flush();receipt=dict(**head,compressed_bytes=w.bytes,compressed_sha256=w.hash.hexdigest(),status='completed');write(B/f'segment{i:03d}_RECEIPT.json',receipt);receipts.append(receipt);total+=w.bytes
  sys.stdout.buffer.write(struct.pack('>I',0));sys.stdout.buffer.write((json.dumps(receipt)+'\n').encode());sys.stdout.buffer.flush()
 result=dict(status='completed',segments=len(receipts),files=len(m),compressed_bytes=total,compressed_payload_sha256=whole.hexdigest(),manifest_sha256=sha(B/'BACKUP_FILES.json'));write(B/'STREAM_RECEIPT.json',result)
 sys.stdout.buffer.write((json.dumps(dict(receipt=result,evidence=evidence()))+'\n').encode());sys.stdout.buffer.flush()
def receive():
 LOCAL.mkdir(parents=True,exist_ok=True);assert not any((LOCAL/n).exists() for n in ['remote_backup','BACKUP_RECEIPT.json','BACKUP_FAILURE.json','SNAPSHOT.json']);target=LOCAL/'remote_backup';whole=hashlib.sha256();errpath=LOCAL/'TRANSPORT_STDERR.log'
 try:
  result=subprocess.run(SSH+[PREFIX+'python -B src/tools/transfer_phase320.py snapshot'],capture_output=True);write(LOCAL/'SNAPSHOT_SSH.json',dict(exit=result.returncode,stderr=result.stderr.decode(errors='replace')));assert result.returncode==0,'inspect snapshot stderr and remote state; no repeated snapshot'
  answer=json.loads(result.stdout);h=answer['header'];raw=base64.b64decode(answer['manifest_base64']);assert hashlib.sha256(raw).hexdigest()==h['manifest_sha256'];m=json.loads(raw);assert len(m)==h['files'] and h['projected_two_hosts_with_summary']<=1073741824
  target.mkdir();(target/'BACKUP_FILES.json').write_bytes(raw);write(LOCAL/'SNAPSHOT.json',h);batches=[];names=[];size=0
  for n,e in sorted(m.items()):
   if names and (len(names)>=256 or size+e['bytes']>128000000):batches.append(names);names=[];size=0
   assert e['bytes']<=128000000;npath=PurePosixPath(n);assert not npath.is_absolute() and '..' not in npath.parts and ':' not in n and '\\' not in n;names.append(n);size+=e['bytes']
  if names:batches.append(names)
  plan=dict(manifest_sha256=h['manifest_sha256'],batches=batches);write(LOCAL/'TRANSFER_PLAN.json',plan);verified=[]
  with errpath.open('xb') as err:
   process=subprocess.Popen(SSH+[PREFIX+'python -B src/tools/transfer_phase320.py stream'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err);process.stdin.write(json.dumps(plan).encode());process.stdin.close()
   for i,names in enumerate(batches,1):
    head=json.loads(process.stdout.readline());assert head['index']==i and head['manifest_sha256']==h['manifest_sha256'];reader=FramedReader(process.stdout,whole);count=0
    with gzip.GzipFile(fileobj=reader,mode='rb') as gz:
     with tarfile.open(fileobj=gz,mode='r|') as tar:
      for item in tar:
       n=item.name;assert item.isfile() and n in names;dest=target/n;assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_name(dest.name+f'.partial.segment{i:03d}')
       with temp.open('xb') as f:
        source=tar.extractfile(item)
        for data in iter(lambda:source.read(1048576),b''):f.write(data)
       assert temp.stat().st_size==m[n]['bytes'] and sha(temp)==m[n]['sha256'],n;temp.rename(dest);count+=1
     while gz.read(1048576):pass
    assert reader.eof and not reader.buffer and count==len(names);receipt=json.loads(process.stdout.readline());assert receipt['compressed_sha256']==reader.hash.hexdigest() and receipt['compressed_bytes']==reader.bytes;verified.append(receipt);write(LOCAL/f'segment{i:03d}_VERIFIED.json',dict(remote=receipt,local_compressed_sha256=reader.hash.hexdigest(),all_members_SHA_verified=True));print('Phase320 segment verified',i,'/',len(batches),count,flush=True)
   final=json.loads(process.stdout.readline());assert process.wait()==0 and final['receipt']['compressed_payload_sha256']==whole.hexdigest();assert final['receipt']['compressed_bytes']==sum(r['compressed_bytes'] for r in verified)
  evdir=target/'backup_transport';evdir.mkdir();evsha={}
  for n,e in final['evidence'].items():
   assert Path(n).name==n and ':' not in n;data=base64.b64decode(e['base64']);assert hashlib.sha256(data).hexdigest()==e['sha256'];(evdir/n).write_bytes(data);evsha[n]=e['sha256']
  # Independent final file pass, not a repeated transfer.
  for n,e in m.items():assert sha(target/n)==e['sha256'] and (target/n).stat().st_size==e['bytes'],n
  b=dict(h);b.update(status='completed',all_member_SHA_verified=True,all_segment_SHA_verified=True,whole_compressed_payload_SHA_verified=True,segments=len(verified),compressed_bytes=final['receipt']['compressed_bytes'],compressed_payload_sha256=whole.hexdigest(),transport_evidence_SHA=evsha,backup_mode='manifest_pinned_framed_gzip_segments');write(LOCAL/'BACKUP_RECEIPT.json',b);print(json.dumps(b))
 except BaseException:write(LOCAL/'BACKUP_FAILURE.json',dict(traceback=traceback.format_exc(),action='preserve snapshot, plan, verified segments, partial and stderr; inspect remote receipt/process before any recovery; no automatic retransmission'));raise
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='evidence':print(json.dumps(evidence()))
 else:{'snapshot':snapshot,'stream':stream,'receive':receive}[mode]()

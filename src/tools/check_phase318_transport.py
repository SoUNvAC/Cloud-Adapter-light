"""Five-byte in-memory check of actual frame/gzip/tar boundary APIs, not experiment data."""
import io,gzip,tarfile,struct
from transfer_phase318 import FramedWriter,FramedReader
from phase318_common import *
def main():
 raw=io.BytesIO();wh=hashlib.sha256();w=FramedWriter(raw,wh)
 with gzip.GzipFile(fileobj=w,mode='wb',compresslevel=1,mtime=0) as gz:
  with tarfile.open(fileobj=gz,mode='w|') as tar:
   info=tarfile.TarInfo('probe');info.size=5;tar.addfile(info,io.BytesIO(b'318ok'))
 raw.write(struct.pack('>I',0));raw.write(b'{"next":"boundary"}\n');raw.seek(0);rh=hashlib.sha256();r=FramedReader(raw,rh)
 with gzip.GzipFile(fileobj=r,mode='rb') as gz:
  with tarfile.open(fileobj=gz,mode='r|') as tar:
   iterator=iter(tar);item=next(iterator);assert item.name=='probe' and tar.extractfile(item).read()==b'318ok';assert next(iterator,None) is None
  while gz.read(1048576):pass
 assert r.eof and not r.buffer and raw.readline()==b'{"next":"boundary"}\n' and r.bytes==w.bytes and r.hash.hexdigest()==w.hash.hexdigest()==rh.hexdigest()==wh.hexdigest()
 write(ROOT/'BACKUP_FORMAT_ACCEPTANCE.json',dict(status='passed',raw_probe_bytes=5,gzip_tar_member_exact=True,following_JSON_boundary_preserved=True,segment_and_payload_SHA_equal=True,no_persistent_archive=True,runtime=sys.version));print('Phase318 framedgzip format passed: fivebytes/member/EOF/boundary/segment and total SHA')
if __name__=='__main__':main()

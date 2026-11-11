"""Fixed three-archive input intake; no model imports, no inference."""
from pathlib import Path, PurePosixPath
import datetime, hashlib, json, os, shutil, subprocess, sys, time, urllib.request, gzip, tarfile, resource, threading
from concurrent.futures import ThreadPoolExecutor

ROOT=Path('/home/scv/shared/phase246_20261031')
SPEC=[('Irish_026046_2001-03-24','p26_r46.tar.gz',307791735),('Irish_035042_2001-08-14','p35_r42.tar.gz',255052281),('Irish_039025_2001-07-25','p39_r25.tar.gz',293324706)]
START=time.monotonic()
if (ROOT/'START_EPOCH').exists():START-=max(0,time.time()-float((ROOT/'START_EPOCH').read_text()))
WRITE_LOCK=threading.RLock()

def write(name,obj):
 with WRITE_LOCK:
  p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2),encoding='utf-8');os.replace(t,p)

def status(stage,**kw):
 name='STATUS_'+kw['scene']+'.json' if 'scene' in kw else 'STATUS.json'
 write(name,{'stage':stage,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-START,**kw})

def intake_one(spec):
 scene,filename,expected=spec
 if time.monotonic()-START>1800:raise RuntimeError('wall clock budget exhausted')
 target=ROOT/filename;part=ROOT/(filename+'.part');url='https://landsat.usgs.gov/cloud-validation/cca_irish_2015/'+filename
 attempts=[]
 if not target.exists():
  budget=json.loads((ROOT/'REMAINING_REQUESTS.json').read_text()) if (ROOT/'REMAINING_REQUESTS.json').exists() else {}
  for attempt in range(budget.get(filename,3)):
   offset=part.stat().st_size if part.exists() else 0
   headers={'User-Agent':'research-input-acceptance/1.0'}
   if offset:headers['Range']='bytes='+str(offset)+'-'
   try:
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as response:
     if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-')):
      raise RuntimeError('resume range not honored; partial preserved')
     if not offset and response.status!=200:raise RuntimeError('unexpected HTTP status')
     received=offset;last=0
     with part.open('ab' if offset else 'wb') as f:
      while True:
       if time.monotonic()-START>1800:raise RuntimeError('wall clock budget exhausted')
       b=response.read(1024**2)
       if not b:break
       f.write(b);received+=len(b)
       if received>expected:raise RuntimeError('archive exceeds declared length')
       if received-last>=8*1024**2:status('downloading',scene=scene,bytes=received,expected=expected,attempt=attempt+1);last=received
    if part.stat().st_size!=expected:raise RuntimeError('download length mismatch')
    part.rename(target);attempts.append({'attempt':attempt+1,'offset':offset,'status':'complete'});break
   except Exception as e:
    attempts.append({'attempt':attempt+1,'offset':offset,'error':str(e)});write('DOWNLOAD_'+filename+'.json',attempts)
    if attempt<2:time.sleep(3)
 if not target.exists():
  return {'scene':scene,'status':'download_pending','attempts':attempts}
 if target.stat().st_size!=expected:raise RuntimeError('existing archive length mismatch')
 status('validating_archive',scene=scene)
 h=hashlib.sha256()
 with target.open('rb') as f:
  while b:=f.read(4*1024**2):h.update(b)
 # Read gzip to EOF to verify CRC/trailer, independently of tar termination.
 unpacked=0
 with gzip.open(target,'rb') as f:
  while b:=f.read(4*1024**2):
   unpacked+=len(b)
   if unpacked>3*1024**3:raise RuntimeError('archive decompressed size exceeds3GiB')
 members=[];texts=[]
 textdir=ROOT/(scene+'_metadata');textdir.mkdir(exist_ok=True)
 with tarfile.open(target,'r:gz') as tar:
  for member in tar:
   members.append({'name':member.name,'bytes':member.size,'type':str(member.type),'regular_file':member.isfile()})
   if member.isfile() and member.size<=1024**2 and Path(member.name).suffix.lower() in ['.txt','.xml','.md']:
    data=tar.extractfile(member).read();safe=Path(member.name).name
    dest=textdir/safe
    if dest.exists() and dest.read_bytes()!=data:raise RuntimeError('duplicate metadata basename')
    dest.write_bytes(data);texts.append({'name':member.name,'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode('utf-8',errors='replace')})
 result={'scene':scene,'status':'archive_and_metadata_acquired','archive':str(target),'bytes':expected,'sha256':h.hexdigest(),'gzip_CRC_verified':True,'uncompressed_tar_bytes':unpacked,'attempts':attempts,'members':members,'texts':texts,'publisher_digest':'not_obtained','image_pixels_opened':False}
 write(scene+'_ARCHIVE.json',result);return {k:v for k,v in result.items() if k not in ['texts','members']}

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(1200,1200))
 if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('less than8GiB remote free')
 status('parallel_archive_intake',workers=3)
 with ThreadPoolExecutor(max_workers=3) as ex:results=list(ex.map(intake_one,SPEC))
 write('ARCHIVE_SUMMARY.json',{'results':results,'new_model_calls':0,'elapsed_seconds':time.monotonic()-START,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
 status('archive_intake_complete',completed=len([r for r in results if r['status']=='archive_and_metadata_acquired']),total=3)

if __name__=='__main__':
 try:main()
 except Exception as e:
  status('failed',error=repr(e));raise

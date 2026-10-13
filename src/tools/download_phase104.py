"""Single-copy generation-pinned downloader; durable conservative 9GB budget."""
import argparse
import base64
import os
import shutil
import time
import urllib.request
import urllib.parse
from phase104_common import *

CHUNK=1024*1024

def checksum(path):
    m=hashlib.md5();h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*CHUNK),b''):m.update(b);h.update(b)
    return base64.b64encode(m.digest()).decode(),h.hexdigest()

def download(root):
    manifest=read(RESULT/'object_manifest.json');lock=read(RESULT/'task_lock.json')
    assert sha(RESULT/'object_manifest.json')==lock['object_manifest_sha256']
    assert sum(o['bytes'] for o in manifest['objects'])==EXPECTED_BYTES
    root.mkdir(parents=True,exist_ok=True);lp=root/'DOWNLOAD_LOCK'
    # Stale lock must be investigated by the operator, never silently removed.
    fd=os.open(lp,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    statuspath=root/'download_status.json';budgetpath=root/'transfer_budget.json'
    state=read(statuspath) if statuspath.exists() else dict(status='initialized',files={},started_utc=now())
    budget=read(budgetpath) if budgetpath.exists() else dict(cap_bytes=CAP,reserved_bytes=0,actual_received_bytes=0,
                                                           accounting='reserve whole bounded HTTP response before opening; uncertain failed response bytes stay charged')
    initial=shutil.disk_usage(root).free
    write(root/'disk_before.json',dict(utc=now(),filesystem_free_bytes=initial,
          host_free_user_report_bytes=12400000000,host_report_not_independently_verified=True,
          expected_payload_bytes=EXPECTED_BYTES,expected_host_remaining_bytes=12400000000-EXPECTED_BYTES,
          no_dense_float_cache=True),exclusive=not (root/'disk_before.json').exists())
    try:
        for obj in manifest['objects']:
            key=obj['path'];dest=root/'imagery'/key;part=dest.with_suffix('.jp2.partial');dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                md5,h=checksum(dest)
                assert dest.stat().st_size==obj['bytes'] and md5==obj['md5_base64'],'Existing file checksum mismatch: '+key
                old=state['files'].get(key)
                if old:assert old['sha256']==h
                state['files'][key]=dict(**obj,sha256=h,verified=True);write(statuspath,state);continue
            offset=part.stat().st_size if part.exists() else 0
            assert offset<=obj['bytes']
            assert shutil.disk_usage(root).free>=obj['bytes']-offset+1024**3,'Filesystem reserve <1GiB'
            # A failed transfer is resumed only with the same pinned generation.
            for attempt in range(4):
                offset=part.stat().st_size if part.exists() else 0
                if offset==obj['bytes']:break
                request_bytes=obj['bytes']-offset
                assert budget['reserved_bytes']+request_bytes<=CAP,'Cumulative transfer budget exhausted before request'
                budget['reserved_bytes']+=request_bytes;write(budgetpath,budget)
                url='https://storage.googleapis.com/gcp-public-data-sentinel-2/'+urllib.parse.quote(obj['name'],safe='/')+'?generation='+obj['generation']
                req=urllib.request.Request(url,headers={'Range':f"bytes={offset}-{obj['bytes']-1}",'Accept-Encoding':'identity'})
                try:
                    with urllib.request.urlopen(req,timeout=45) as response:
                        assert response.status==206,'Range ignored; never append full response'
                        assert response.headers['Content-Range']==f"bytes {offset}-{obj['bytes']-1}/{obj['bytes']}",'Content-Range mismatch'
                        assert int(response.headers['Content-Length'])==obj['bytes']-offset
                        with part.open('ab') as f:
                            remaining=obj['bytes']-offset
                            while remaining:
                                allowance=min(CHUNK,remaining)
                                received=response.read(allowance)
                                budget['actual_received_bytes']+=len(received)
                                write(budgetpath,budget)
                                if not received:raise OSError('Unexpected EOF')
                                f.write(received);f.flush();remaining-=len(received)
                                state.update(status='downloading',current=key,current_partial_bytes=part.stat().st_size,
                                             updated_utc=now(),actual_received_bytes=budget['actual_received_bytes'])
                                write(statuspath,state)
                    break
                except (OSError,TimeoutError) as e:
                    print('Network interruption',key,attempt+1,type(e).__name__,str(e),flush=True)
                    if attempt==3:raise
                    time.sleep(2**attempt)
            assert part.stat().st_size==obj['bytes'],'Incomplete payload'
            md5,h=checksum(part)
            assert md5==obj['md5_base64'],'Downloaded MD5 mismatch: '+key
            assert not dest.exists();part.rename(dest)
            state['files'][key]=dict(**obj,sha256=h,verified=True)
            state.update(status='downloading',verified_count=len(state['files']),updated_utc=now())
            write(statuspath,state);print('Verified',len(state['files']),96,key,flush=True)
        assert len(state['files'])==96
        state.update(status='all_files_verified',completed_utc=now(),verified_bytes=EXPECTED_BYTES,
                     actual_received_bytes=budget['actual_received_bytes'],charged_bytes=budget['reserved_bytes'])
        write(statuspath,state)
        write(root/'disk_after_download.json',dict(utc=now(),filesystem_free_bytes=shutil.disk_usage(root).free,
               disk_delta_bytes=initial-shutil.disk_usage(root).free,actual_received_bytes=budget['actual_received_bytes'],
               host_remaining_estimate_bytes=12400000000-EXPECTED_BYTES))
    except BaseException as e:
        state.update(status='stopped',error=type(e).__name__+': '+str(e),updated_utc=now());write(statuspath,state);raise
    finally:
        # Removed only by its owning process after all writes close, never by a second runner.
        assert lp.read_text()==str(os.getpid());lp.unlink()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();download(args.root)

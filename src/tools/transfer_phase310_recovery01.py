"""Preserve failed decode; restore only missing members, then verify exact frozen stream bytes.

The original producer completed; the receiver stopped at member251 with unexpected end.
Cause is unproven. This recovery never decodes or re-extracts the large tar stream:
it hashes a deterministic replay of the same immutable manifest and independently
verifies every assembled file. Original partial and failure remain untouched.
"""
import gzip
import hashlib
import subprocess
import sys
import tarfile
from phase310_common import *
from transfer_phase310 import HashStream

def hash_stream():
    receipt=load(ROOT/'BACKUP_STREAM_RECEIPT.json');manifest=load(ROOT/'BACKUP_FILES.json')
    assert sha(ROOT/'BACKUP_FILES.json')==receipt['manifest_sha256']
    files=[ROOT/n for n in sorted(manifest)]
    for p in files:
        value=manifest[p.relative_to(ROOT).as_posix()]
        assert p.stat().st_size==value['bytes'] and sha(p)==value['sha256']
    out=HashStream(sys.stdout.buffer)
    with gzip.GzipFile(fileobj=out,mode='wb',compresslevel=1,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|') as tar:
            tar.add(ROOT/'BACKUP_FILES.json',arcname='BACKUP_FILES.json',recursive=False)
            for p in files:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    out.flush();assert out.bytes==receipt['stream_bytes'] and out.h.hexdigest()==receipt['stream_sha256']
    sys.stderr.write(json.dumps(dict(receipt,hash_only_replay=True,immutable_original_files=True,source_code_sha256=sha(__file__)))+'\n')

def receive_recovery():
    target=LOCAL/'remote_backup';failure=load(LOCAL/'BACKUP_FAILURE.json');assert failure['verified_members']==250
    assert not (LOCAL/'BACKUP_RECEIPT.json').exists() and not (LOCAL/'BACKUP_RECOVERY01.json').exists()
    manifest=load(target/'BACKUP_FILES.json');missing=[n for n,v in manifest.items() if not (target/n).exists()]
    assert set(missing)=={'recovery01/train_RESOURCE.json','resolved_lora_config.py','train.log','train_GPU_RECEIPT.json','train_RESOURCE.json'}
    assert sum(manifest[n]['bytes'] for n in missing)<1000000
    recovered=[]
    for n in missing:
        final=target/n;temp=final.with_name(final.name+'.recovery01.partial');assert not temp.exists() and not final.exists()
        subprocess.run(['scp','-o','BatchMode=yes','gzs:'+str(ROOT/n),str(temp)],check=True)
        assert sha(temp)==manifest[n]['sha256'] and temp.stat().st_size==manifest[n]['bytes'];temp.rename(final);recovered.append(n)
    for n,v in manifest.items():assert sha(target/n)==v['sha256'] and (target/n).stat().st_size==v['bytes'],n
    command='source /home/scv/miniconda3/etc/profile.d/conda.sh && conda activate cloud-lite-pt210 && cd /home/scv/Cloud-Adapter-light && python -B src/tools/transfer_phase310_recovery01.py hash-stream'
    p=subprocess.Popen(['ssh','-o','BatchMode=yes','gzs',command],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stream=HashStream(p.stdout)
    try:
        for b in iter(lambda:stream.read(4*1024**2),b''):pass
        stderr=p.stderr.read().decode('utf-8');assert p.wait()==0,stderr
        receipt=json.loads(stderr.strip().splitlines()[-1]);assert receipt['hash_only_replay']
        assert stream.bytes==receipt['stream_bytes'] and stream.h.hexdigest()==receipt['stream_sha256']
        assert sha(target/'BACKUP_FILES.json')==receipt['manifest_sha256'] and len(manifest)+1==receipt['files']
        receipt.update(all_archive_members_verified=True,all_stream_members_verified=True,initial_decode_failed_preserved=True,initial_stream_locally_verified=False,replayed_same_frozen_stream_SHA_verified=True,missing_members_recovered=recovered,no_experiment_repeated=True)
        write(LOCAL/'BACKUP_RECOVERY01.json',receipt);write(LOCAL/'BACKUP_RECEIPT.json',receipt);write(target/'BACKUP_STREAM_RECEIPT.json',receipt);print(json.dumps(receipt))
    except BaseException:
        import traceback
        p.terminate();write(LOCAL/'BACKUP_FAILURE_HASH_REPLAY01.json',dict(traceback=traceback.format_exc(),bytes_received=stream.bytes,sha256_received=stream.h.hexdigest(),recovered_members=recovered,no_retry=True));raise
if __name__=='__main__':{'hash-stream':hash_stream,'receive-recovery':receive_recovery}[sys.argv[1]]()

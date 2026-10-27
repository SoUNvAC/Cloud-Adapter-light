"""Local byte-only recovery of ancillary receipts; never receive/stream raw data again."""
import sys,subprocess,base64,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'src/tools'))
from phase320_common import LOCAL,sha,load,write
from transfer_phase320 import SSH,PREFIX
assert (LOCAL/'BACKUP_FAILURE.json').exists() and not (LOCAL/'BACKUP_RECEIPT.json').exists()
assert (LOCAL/'segment001_VERIFIED.json').exists() and len(load(LOCAL/'TRANSFER_PLAN.json')['batches'])==1
result=subprocess.run(SSH+[PREFIX+'python -B src/tools/transfer_phase320.py evidence'],capture_output=True)
raw=LOCAL/'RECOVERY_EVIDENCE.raw.json'
with raw.open('xb') as f:f.write(result.stdout)
with (LOCAL/'METADATA_RECOVERY_STDERR.log').open('xb') as f:f.write(result.stderr)
write(LOCAL/'METADATA_RECOVERY_SSH.json',dict(exit=result.returncode,stdout_bytes=len(result.stdout),stderr_bytes=len(result.stderr),mode='old already-pulled tool evidence only; no raw stream or new snapshot'))
assert result.returncode==0,'preserve captured raw metadata and inspect before any further action'
evidence=json.loads(result.stdout);target=LOCAL/'remote_backup/backup_transport';assert not target.exists();target.mkdir()
hashes={}
for name,entry in evidence.items():
 assert Path(name).name==name and ':' not in name and '\\' not in name
 data=base64.b64decode(entry['base64'],validate=True);h=hashlib.sha256(data).hexdigest();assert h==entry['sha256'];p=target/name
 with p.open('xb') as f:f.write(data)
 hashes[name]=h
h=load(LOCAL/'SNAPSHOT.json');master=load(target/'STREAM_RECEIPT.json');segment=load(target/'segment001_RECEIPT.json');verified=load(LOCAL/'segment001_VERIFIED.json')
assert master['status']=='completed' and master['segments']==1 and master['files']==h['files']==47
assert master['manifest_sha256']==h['manifest_sha256']==sha(target/'BACKUP_FILES.json')==sha(LOCAL/'remote_backup/BACKUP_FILES.json')
assert master['compressed_payload_sha256']==segment['compressed_sha256']==verified['local_compressed_sha256']
assert master['compressed_bytes']==segment['compressed_bytes']==verified['remote']['compressed_bytes'] and verified['remote']==segment
assert load(target/'SNAPSHOT.json')==h
m=load(LOCAL/'remote_backup/BACKUP_FILES.json')
for name,entry in m.items():assert sha(LOCAL/'remote_backup'/name)==entry['sha256'] and (LOCAL/'remote_backup'/name).stat().st_size==entry['bytes']
proof=dict(status='passed',mode='metadata_only',sole_verified_segment_equals_entire_compressed_payload=True,all_member_SHA_rechecked=True,no_raw_retransmission=True,original_failure_retained=True,remote_no_active_transfer_observed_in_prior_probe=True,evidence_stdout_complete=True,original_final_raw_line_was_not_saved_so_precise_cause_unknown=True)
write(LOCAL/'METADATA_RECOVERY_ACCEPTANCE.json',proof)
receipt=dict(h);receipt.update(status='completed',all_member_SHA_verified=True,all_segment_SHA_verified=True,whole_compressed_payload_SHA_verified=True,segments=1,compressed_bytes=master['compressed_bytes'],compressed_payload_sha256=master['compressed_payload_sha256'],transport_evidence_SHA=hashes,backup_mode='manifest_pinned_framed_gzip_segments_metadata_only_receipt_recovery01',recovery_proof=proof)
write(LOCAL/'BACKUP_RECEIPT.json',receipt);print(json.dumps(receipt))

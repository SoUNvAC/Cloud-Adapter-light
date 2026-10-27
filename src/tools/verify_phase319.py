"""Portable byte verification, never imports arrays or performs local scientific work."""
from phase319_common import *
def main():
 index=load(S/'SUMMARY_SHA256.json')
 for n,h in index['files'].items():assert sha(S/n)==h,n
 for n,h in index['code'].items():assert sha(R/n)==h,n
 b=load(S/'BACKUP_RECEIPT.json');assert b['all_member_SHA_verified'] and b['all_segment_SHA_verified'] and b['whole_compressed_payload_SHA_verified'] and b['projected_two_hosts_with_summary']<=2000000000 and b['no_persistent_archive']
 assert load(S/'STATUS.json')['status']=='completed' and load(S/'INDEPENDENT_QA.json')['status']=='passed'
 assert len(load(S/'GEOMETRY_INPUTS.json'))==160 and len(load(S/'SELECTION_LOCK.json')['rows'])==160
 assert load(S/'SELECTION_LOCK.json')['original_A_Sh']==EXPECTED and sha(S/'BACKUP_FILES.json')==b['manifest_sha256']
 assert load(S/'STATUS.json')['neural_calls']==0 and load(S/'STATUS.json')['training_updates']==0
 for n,h in b['transport_evidence_SHA'].items():assert sha(S/'backup_transport'/n)==h,n
 if sys.platform=='linux':
  for n,e in load(ROOT/'backup_transport/BACKUP_FILES.json').items():assert sha(ROOT/n)==e['sha256'] and (ROOT/n).stat().st_size==e['bytes'],n
  for n,e in load(ROOT/'INPUT_SHA256.json').items():assert sha(n)==e['sha256'],n
  for e in load(ROOT/'PREREGISTRATION.json')['dependencies']:assert sha(e['path'])==e['sha256']
  assert sha(ROOT/'DECISION_WINDOW.json')==sha(PRIOR/'DECISION_WINDOW.json')
  assert load(ROOT/'backup_transport/STREAM_RECEIPT.json')['compressed_payload_sha256']==b['compressed_payload_sha256']
 else:
  for n,e in load(LOCAL/'remote_backup/BACKUP_FILES.json').items():assert sha(LOCAL/'remote_backup'/n)==e['sha256'] and (LOCAL/'remote_backup'/n).stat().st_size==e['bytes'],n
 print('Phase319 verified: all160 products; original A_Sh; original inputs/code/results/summary/member/segment/payload SHA; zero training/forward; no nextphase')
if __name__=='__main__':main()

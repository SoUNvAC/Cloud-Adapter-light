"""Portable hash/protocol acceptance; local never loads arrays or models."""
from phase318_common import *
def main():
 index=load(S/'SUMMARY_SHA256.json')
 for n,h in index['files'].items():assert sha(S/n)==h,n
 for n,h in index['code'].items():assert sha(R/n)==h,n
 b=load(S/'BACKUP_RECEIPT.json');assert b['all_member_SHA_verified'] and b['all_segment_SHA_verified'] and b['whole_compressed_payload_SHA_verified'] and b['projected_two_hosts_with_summary']<=12000000000 and b['no_persistent_archive']
 assert load(S/'STATUS.json')['status']=='completed' and load(S/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed' and len(read(S/'METRICS.csv'))==15 and len(read(S/'TRAINING_IDENTITY.csv'))==6
 assert load(S/'DELIVERY_REVIEW.json')['reviewed'] is True
 assert sha(S/'BACKUP_FILES.json')==b['manifest_sha256']
 for n,h in b['transport_evidence_SHA'].items():assert sha(S/'backup_transport'/n)==h,n
 if sys.platform=='linux':
  for n,e in load(ROOT/'backup_transport/BACKUP_FILES.json').items():assert sha(ROOT/n)==e['sha256'] and (ROOT/n).stat().st_size==e['bytes'],n
  for n,h in load(S/'EXECUTION_LOCK.json')['inputs'].items():assert sha(n)==h,n
  for filename in ['CPU_INPUT_SHA256.json','QUALITY_INPUT_SHA256.json']:
   for n,h in load(S/filename).items():assert sha(n)==h,n
  assert sha(ROOT/'DECISION_WINDOW.json')==sha(PRIOR/'DECISION_WINDOW.json')
  assert load(ROOT/'backup_transport/STREAM_RECEIPT.json')['compressed_payload_sha256']==b['compressed_payload_sha256']
 print('Phase318 verified: three pools/15 metric rows/six arms or strict equivalents; input/code/result/member/segment/payload SHA; no nextphase')
if __name__=='__main__':main()

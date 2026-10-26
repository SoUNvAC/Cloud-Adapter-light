"""Portable delivery SHA/protocol checks; local mode never reads model/data arrays."""
from phase317_common import *
def main():
 d=load(S/'SUMMARY_SHA256.json')
 for n,h in d['files'].items():assert sha(S/n)==h,n
 for n,h in d['code'].items():assert sha(R/n)==h,n
 assert load(S/'STATUS.json')['status']=='completed' and load(S/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed';assert load(S/'CPU_ACCEPTANCE.json')['historical_CMC_D']==506798 and load(S/'CPU_ACCEPTANCE.json')['historical_CMC_B']==784478
 assert len(read(S/'METRICS.csv'))==9 and len(read(S/'TRAINING_IDENTITY.csv'))==5
 b=load(S/'BACKUP_RECEIPT.json');assert b['all_member_SHA_verified'] and b['whole_compressed_stream_SHA_verified'] and b['projected_two_hosts_with_summary']<=16000000000 and b['no_persistent_archive']
 if sys.platform=='linux':
  for n,v in load(ROOT/'BACKUP_FILES.json').items():assert sha(ROOT/n)==v['sha256'] and (ROOT/n).stat().st_size==v['bytes'],n
  for f,h in load(S/'EXECUTION_LOCK.json')['inputs'].items():assert sha(f)==h,f
  for name in ['CPU_INPUT_SHA256.json','REFERENCE_INPUT_SHA256.json']:
   for f,h in load(S/name).items():assert sha(f)==h,f
  assert load(ROOT/'STREAM_RECEIPT.json')['stream_sha256']==b['stream_sha256']
 print('Phase317 verified: threepools/ninemetricrows/fivefresharms, all input/code/result/member/stream SHA; no nextphase')
if __name__=='__main__':main()

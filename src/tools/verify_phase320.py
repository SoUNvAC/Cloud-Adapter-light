"""Portable byte acceptance; local never imports NumPy or scientific data readers."""
from phase320_common import *
def main():
 index=load(S/'SUMMARY_SHA256.json')
 for n,h in index['files'].items():assert sha(S/n)==h,n
 for n,h in index['code'].items():assert sha(R/n)==h,n
 for n,h in load(S/'ANALYSIS_LOCK.json')['code'].items():assert sha(R/n)==h,n
 b=load(S/'BACKUP_RECEIPT.json');assert b['all_member_SHA_verified'] and b['all_segment_SHA_verified'] and b['whole_compressed_payload_SHA_verified'] and b['projected_two_hosts_with_summary']<=1024**3 and b['no_persistent_archive']
 assert sha(S/'BACKUP_FILES.json')==b['manifest_sha256'] and load(S/'STATUS.json')['status']=='completed' and load(S/'INDEPENDENT_QA.json')['status']=='passed';assert load(S/'STATUS.json')['neural_calls']==0 and load(S/'STATUS.json')['training_updates']==0
 for name in ['ERROR_COMPOSITION.csv','ERROR_LOCATION.csv','ERROR_SCENE_SUMMARY.csv','ERROR_OPPORTUNITY.md','ORIGINAL_POINT_REPRODUCTION.csv','THRESHOLD_MATCH.csv','THRESHOLD_SCENE.csv','RANKING_SUMMARY.csv','ANALYSIS_LOCK.json','RESULT.md','phase320.md']:assert (S/name).is_file(),name
 for n,h in b['transport_evidence_SHA'].items():assert sha(S/'backup_transport'/n)==h,n
 root=ROOT if sys.platform=='linux' else LOCAL/'remote_backup';manifest=load(ROOT/'backup_transport/BACKUP_FILES.json') if sys.platform=='linux' else load(root/'BACKUP_FILES.json')
 for n,e in manifest.items():assert sha(root/n)==e['sha256'] and (root/n).stat().st_size==e['bytes'],n
 if sys.platform=='linux':
  for p,e in load(ROOT/'INPUT_SHA256.json').items():assert sha(p)==e['sha256'],p
  assert sha(ROOT/'DECISION_WINDOW.json')==sha(P317/'DECISION_WINDOW.json');assert not list((ROOT/'tmp').glob('*'))
  assert load(ROOT/'backup_transport/STREAM_RECEIPT.json')['compressed_payload_sha256']==b['compressed_payload_sha256']
 print('Phase320 verified: both diagnostics, original supports/AP, input/code/result/member/segment/payload SHA; zero model calls; no321')
if __name__=='__main__':main()

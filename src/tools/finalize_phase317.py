"""Local file-only publication after member and stream SHA; no local analysis."""
from phase317_common import *
def main():
 b=load(LOCAL/'BACKUP_RECEIPT.json');assert b['all_member_SHA_verified'] and (b['whole_compressed_stream_SHA_verified'] or b.get('completed_recovery_streams_SHA_verified'));S.mkdir(parents=True,exist_ok=True)
 for p in (LOCAL/'remote_backup').iterdir():
  if p.is_file() and p.suffix in ['.json','.csv','.md','.log','.jsonl']:shutil.copyfile(p,S/p.name)
 shutil.copyfile(LOCAL/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json')
 recovery=b.get('backup_mode')=='member_checked_prefix_plus_SHA_verified_segmented_suffix'
 if recovery:
  rec=S/'backup_recovery01';rec.mkdir(exist_ok=True)
  for n in ['BACKUP_FAILURE.json','REMOTE_STREAM_HEADER.json','RECOVERY01_PLAN.json','RECOVERY01_INTEGRITY_INDEX.json','RECOVERY01_DIAGNOSIS.json']:
   if (LOCAL/n).exists():shutil.copyfile(LOCAL/n,rec/n)
  shutil.copytree(LOCAL/'recovery01',rec/'transport',dirs_exist_ok=True);shutil.copytree(LOCAL/'remote_backup/backup_recovery01',rec/'remote_evidence',dirs_exist_ok=True)
  assert sha(rec/'RECOVERY01_INTEGRITY_INDEX.json')==b['integrity_index_sha256']
 with (S/'phase317.md').open('a',encoding='utf-8') as f:
  if recovery:f.write(f"\n- 完整备份：首个大流在约1.97GB处EOF，原失败/partial保留，首流整流SHA未完成，未标成功。已验前缀加只补缺失成员的{b['recovery_batches']}段短流完成{b['files']}成员/原始{b['raw_bytes']}字节；全部原快照成员SHA及补传各段整流SHA通过。原清单SHA256={b['manifest_sha256']}，恢复完整性索引SHA256={b['integrity_index_sha256']}；双机含摘要预留{b['projected_two_hosts_with_summary']}字节≤16GB，无持久大归档，未重复任何科学阶段。\n")
  else:f.write(f"\n- 完整备份：{b['files']}成员、原始{b['raw_bytes']}字节；流{b['stream_bytes']}字节，SHA256={b['stream_sha256']}。全部成员及压缩整流SHA通过；双机与摘要保留计数{b['projected_two_hosts_with_summary']}字节≤16GB，无持久大归档。\n")
 write(S/'SUMMARY_SHA256.json',dict(files={p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file() and p.name!='SUMMARY_SHA256.json'},code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase317*.py')}));print('Phase317 publication complete')
if __name__=='__main__':main()

"""Local file-only publication after member and stream SHA; no local analysis."""
from phase317_common import *
def main():
 b=load(LOCAL/'BACKUP_RECEIPT.json');assert b['all_member_SHA_verified'] and b['whole_compressed_stream_SHA_verified'];S.mkdir(parents=True,exist_ok=True)
 for p in (LOCAL/'remote_backup').iterdir():
  if p.is_file() and p.suffix in ['.json','.csv','.md','.log','.jsonl']:shutil.copyfile(p,S/p.name)
 shutil.copyfile(LOCAL/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json')
 with (S/'phase317.md').open('a',encoding='utf-8') as f:f.write(f"\n- 完整备份：{b['files']}成员、原始{b['raw_bytes']}字节；流{b['stream_bytes']}字节，SHA256={b['stream_sha256']}。全部成员及压缩整流SHA通过；双机与摘要保留计数{b['projected_two_hosts_with_summary']}字节≤16GB，无持久大归档。\n")
 write(S/'SUMMARY_SHA256.json',dict(files={p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file() and p.name!='SUMMARY_SHA256.json'},code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase317*.py')}));print('Phase317 publication complete')
if __name__=='__main__':main()

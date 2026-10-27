"""Local file/hash-only publication following complete streamed backup acceptance."""
from phase320_common import *
def main():
 b=load(LOCAL/'BACKUP_RECEIPT.json');assert b['status']=='completed' and b['all_member_SHA_verified'] and b['all_segment_SHA_verified'] and b['whole_compressed_payload_SHA_verified'];assert not S.exists()
 pre=load(LOCAL/'remote_backup/ANALYSIS_LOCK.json')
 for rel,h in pre['code'].items():assert sha(R/rel)==h,'local code differs from executed canonical bytes: '+rel
 S.mkdir(parents=True)
 for p in (LOCAL/'remote_backup').iterdir():
  if p.is_file() and p.suffix in ['.json','.csv','.md','.log','.jsonl']:shutil.copyfile(p,S/p.name)
 for n in ['BACKUP_RECEIPT.json','NETWORK_EVENTS.json','SNAPSHOT.json','TRANSFER_PLAN.json','SNAPSHOT_SSH.json','TRANSPORT_STDERR.log']:
  if (LOCAL/n).exists():shutil.copyfile(LOCAL/n,S/n)
 shutil.copytree(LOCAL/'remote_backup/backup_transport',S/'backup_transport')
 for p in LOCAL.glob('segment*_VERIFIED.json'):shutil.copyfile(p,S/'backup_transport'/p.name)
 with (S/'phase320.md').open('a',encoding='utf-8') as f:f.write(f"\n- 完整备份实际回执：{b['files']}成员/{b['raw_bytes']}字节，{b['segments']}段及逐文件/整流SHA全部通过；payloadSHA256={b['compressed_payload_sha256']}，manifestSHA256={b['manifest_sha256']}。双机新持久含摘要预留{b['projected_two_hosts_with_summary']}字节<=1GiB，无持久大归档。网络与失败实录如有则按原件保留；本机只编辑、文件SHA与备份。\n")
 write(S/'SUMMARY_SHA256.json',dict(files={p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file()},code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase320*.py')}));print('Phase320 file-only publication complete')
if __name__=='__main__':main()

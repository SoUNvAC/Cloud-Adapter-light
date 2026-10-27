"""Local publication uses files/hashes only after the entire streamed backup passes."""
from phase319_common import *
def main():
 b=load(LOCAL/'BACKUP_RECEIPT.json');assert b['status']=='completed' and b['all_member_SHA_verified'] and b['all_segment_SHA_verified'] and b['whole_compressed_payload_SHA_verified'];assert not S.exists();S.mkdir(parents=True)
 for p in (LOCAL/'remote_backup').iterdir():
  if p.is_file() and p.suffix in ['.json','.csv','.md','.log','.jsonl']:shutil.copyfile(p,S/p.name)
 for n in ['BACKUP_RECEIPT.json','NETWORK_EVENTS.json','SNAPSHOT.json','TRANSFER_PLAN.json','SNAPSHOT_SSH.json','TRANSPORT_STDERR.log']:
  if (LOCAL/n).exists():shutil.copyfile(LOCAL/n,S/n)
 shutil.copytree(LOCAL/'remote_backup/backup_transport',S/'backup_transport')
 for p in LOCAL.glob('segment*_VERIFIED.json'):shutil.copyfile(p,S/'backup_transport'/p.name)
 with (S/'phase319.md').open('a',encoding='utf-8') as f:
  f.write(f"\n- 本地完整备份实际验收：{b['files']}成员，原始{b['raw_bytes']}字节，{b['segments']}段；所有成员/分段/整流SHA通过。整流SHA256={b['compressed_payload_sha256']}，清单SHA256={b['manifest_sha256']}。双机含摘要预留{b['projected_two_hosts_with_summary']}字节<=2GB，无持久大归档。本机仅编辑/文件SHA/备份。网络及执行探测实录见NETWORK_EVENTS.json；原stderr保留，未因此重跑科学阶段。\n")
 write(S/'SUMMARY_SHA256.json',dict(files={p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file()},code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase319*.py')}));print('Phase319 file-only publication complete')
if __name__=='__main__':main()

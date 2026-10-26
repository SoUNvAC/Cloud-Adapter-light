"""Local publication: metadata byte-copy and hashes only; all analysis stays gzs."""
import shutil
from phase315_common import *
def main():
    b=load(LOCAL/'BACKUP_RECEIPT.json');assert b['all_archive_members_verified'];backup=LOCAL/'remote_backup';S.mkdir(parents=True,exist_ok=True)
    for p in backup.iterdir():
        if p.is_file() and p.suffix in ['.json','.csv','.md','.log','.jsonl']:shutil.copyfile(p,S/p.name)
    shutil.copyfile(LOCAL/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json')
    with (S/'phase315.md').open('a',encoding='utf-8') as f:f.write(f"\n- 完整备份：{b['files']}成员，原始{b['raw_bytes']}字节、归档{b['archive_bytes']}字节；SHA256={b['archive_sha256']}，整包先验和全部成员SHA通过。双机/归档/摘要预留计数{b['projected_two_hosts_including_archives_and_summary']}字节≤6GB；初末权重、32组每seed缓存、全部探针/日志/代码/表格均完整本地保存。\n")
    write(S/'SUMMARY_SHA256.json',dict(status='completed',files={p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file() and p.name!='SUMMARY_SHA256.json'},code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase315*') if p.is_file()},archive_sha256=b['archive_sha256']))
    print('Phase315 publication complete')
if __name__=='__main__':main()

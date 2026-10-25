"""Local publication is byte-copy and SHA only; no local scientific computation."""
import shutil
from phase314_common import *
def main():
    b=load(LOCAL/'BACKUP_RECEIPT.json');assert b['all_archive_members_verified'];backup=LOCAL/'remote_backup';S.mkdir(parents=True,exist_ok=True)
    for p in backup.iterdir():
        if p.is_file() and p.suffix in ('.json','.csv','.md','.txt'):shutil.copyfile(p,S/p.name)
    shutil.copyfile(LOCAL/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json');state=load(S/'STATUS.json')
    with (S/'phase314.md').open('a',encoding='utf-8') as f:
        f.write(f"\n- 完整备份：{b['files']}成员、原始{b['raw_bytes']}字节、归档{b['archive_bytes']}字节；SHA256={b['archive_sha256']}，整包先验和全部成员SHA通过。双机/归档/摘要预留计数{b['projected_two_hosts_including_archives_and_summary']}字节，≤500MB。\n")
        f.write(f"- 最终累计：统计/独立审计/报告CPU执行{state['total_seconds']:.6f}秒，峰值{state['peak_cpu_bytes']}字节；完成UTC={state['completed_utc']}。报告仅从冻结表汇总，不重读模型或参考。\n")
    write(S/'SUMMARY_SHA256.json',dict(status='completed',files={p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file() and p.name!='SUMMARY_SHA256.json'},code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase314*') if p.is_file()},archive_sha256=b['archive_sha256']))
    print('Phase314 published')
if __name__=='__main__':main()

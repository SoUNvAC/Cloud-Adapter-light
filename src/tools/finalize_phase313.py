"""Local byte-copy and SHA publication only; never analyze arrays."""
import shutil
from phase313_common import *
def main():
    b=load(LOCAL/'BACKUP_RECEIPT.json');assert b['all_archive_members_verified']
    backup=LOCAL/'remote_backup';S.mkdir(parents=True,exist_ok=True)
    for p in backup.iterdir():
        if p.is_file() and p.suffix in ('.json','.md','.csv','.txt'):shutil.copyfile(p,S/p.name)
    shutil.copyfile(LOCAL/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json')
    qa=load(LOCAL/'SOURCE_VISUAL_QA.json');assert qa['status']=='passed';shutil.copyfile(LOCAL/'SOURCE_VISUAL_QA.json',S/'SOURCE_VISUAL_QA.json')
    state=load(S/'STATUS.json');decision=load(S/'DECISION_LOCK.json')
    with (S/'phase313.md').open('a',encoding='utf-8') as f:
        f.write(f"\n- 实测资源：元数据/文献提取与正式补充/封存合计{state['total_seconds']:.6f}秒，峰值进程内存{state['peak_cpu_bytes']}字节，最低宿主可用{state['minimum_available_memory']}字节；新增文献响应体{state['network_response_body_bytes']}字节。该累计为命名执行阶段，不含本地编辑、SSH等待及人工文献阅读。\n")
        f.write(f"- 完整备份：{b['files']}成员，原结果{b['raw_bytes']}字节、归档{b['archive_bytes']}字节，SHA256={b['archive_sha256']}；整包先验、全部成员SHA通过。双机/归档/摘要预留计数{b['projected_two_hosts_including_archives_and_summary']}字节，低于300MB。\n")
        f.write(f"- 时间字段：STATUS.utc是原始元数据阶段完成UTC；最终决策封存UTC为DECISION_LOCK.utc={decision['utc']}，不将前者当最终完成时刻。\n")
    files={p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file() and p.name!='SUMMARY_SHA256.json'}
    code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase313*') if p.is_file()}
    write(S/'SUMMARY_SHA256.json',dict(status='completed',decision='geometry experiment not executable',files=files,code=code,archive_sha256=b['archive_sha256']))
    print('Published',len(files),'SHA-locked files')
if __name__=='__main__':main()

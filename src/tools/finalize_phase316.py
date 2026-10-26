"""Local byte-copy publication and SHA only; no local metadata/statistical analysis."""
from phase316_common import *
def main():
    b=load(LOCAL/'BACKUP_RECEIPT.json');assert b['all_members_SHA_verified'] and b['archive_SHA_verified']
    for p in (LOCAL/'remote_backup').iterdir():
        if p.is_file() and p.suffix in ['.csv','.json','.md','.log']:shutil.copyfile(p,S/p.name)
    shutil.copyfile(LOCAL/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json')
    with (S/'phase316.md').open('a',encoding='utf-8') as f:f.write(f"\n- 全量备份：{b['files']}成员，原始{b['raw_bytes']}字节；归档{b['archive_bytes']}字节，SHA256={b['archive_sha256']}；整包及全部成员SHA通过。双机/归档/摘要预留计数{b['projected_persistent_bytes']}字节≤300MB。\n")
    write(S/'DELIVERY_SHA256.json',dict(files={p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file() and p.name!='DELIVERY_SHA256.json'},code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase316*.py')}))
    print('Phase316 publication complete; local operations file-only')
if __name__=='__main__':main()

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
    files={str(p.relative_to(S)):sha(p) for p in S.rglob('*') if p.is_file() and p.name!='SUMMARY_SHA256.json'}
    code={str(p.relative_to(R)):sha(p) for p in (R/'src/tools').glob('*phase313*') if p.is_file()}
    write(S/'SUMMARY_SHA256.json',dict(status='completed',decision='geometry experiment not executable',files=files,code=code,archive_sha256=b['archive_sha256']))
    print('Published',len(files),'SHA-locked files')
if __name__=='__main__':main()

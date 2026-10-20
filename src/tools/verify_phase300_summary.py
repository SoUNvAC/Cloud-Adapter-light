"""Verify the final flat summary's byte identities on Windows or Linux."""
import hashlib
import json
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[2]
    root = repo/'src/research_results/phase300_20261020'
    lock = json.loads((root/'FINAL_SUMMARY_LOCK.json').read_text(encoding='utf-8-sig'))
    assert lock['status']=='completed'
    for name,expected in lock['files'].items():
        assert Path(name).name==name and name not in ('.','..')
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    execution = json.loads((root/'EXECUTION_LOCK.json').read_text(encoding='utf-8-sig'))
    for name,expected in execution['code'].items():
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==expected,name
    print(json.dumps(dict(status='passed',summary_files=len(lock['files']),
           locked_training_sources=len(execution['code']),archive_sha256=lock['full_backup_archive_sha256'])))


if __name__=='__main__': main()

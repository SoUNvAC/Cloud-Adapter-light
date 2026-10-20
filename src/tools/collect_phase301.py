"""Archive every newly persisted remote Phase301 file after the CPU process exits."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

R=Path('/home/scv/Cloud-Adapter-light')
ROOT=Path('/home/scv/shared/phase301_20261020')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()


def main():
    assert (ROOT/'EXIT_CODE').read_text().strip()=='0'
    dispatch=json.loads((ROOT/'DISPATCH.json').read_text())
    process=Path('/proc')/str(dispatch['PID'])/'stat'
    assert not process.exists() or process.read_text().split()[2]=='Z', 'CPU controller still active'
    code=ROOT/'code';code.mkdir(exist_ok=True)
    for name,expected in dispatch['code'].items():
        assert sha(R/name)==expected,name
        shutil.copyfile(R/name,code/Path(name).name)
    shutil.copyfile(R/'src/research_plans/phase301.md',code/'phase301.md')
    for path in (R/'src/tools/phase300_reference.py',R/'src/tools/run_phase300_local.py'):
        shutil.copyfile(path,code/path.name)
    archive=ROOT/'delivery.tar.gz'
    excluded={'delivery.tar.gz','DELIVERY_RECEIPT.json','BACKUP_MANIFEST.json'}
    paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.relative_to(ROOT).parts[0] not in excluded)
    size=sum(p.stat().st_size for p in paths)
    assert size*4<1000000000
    manifest={p.relative_to(ROOT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths}
    (ROOT/'BACKUP_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(archive,'w:gz') as out:
        for path in paths+[ROOT/'BACKUP_MANIFEST.json']:
            out.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
    receipt=dict(status='complete',sha256=sha(archive),bytes=archive.stat().st_size,
        archived_files=len(paths)+1,raw_bytes=size,combined_persistent_upper_bytes=4*size,
        neural_calls=0,coverage='Entire Phase301 root: references, source snapshot, all tables, logs and metadata')
    (ROOT/'DELIVERY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()

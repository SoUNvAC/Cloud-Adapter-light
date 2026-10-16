"""Stage only the existing small aggregates and their original SHA receipts."""
import argparse
import shutil
import subprocess
from pathlib import Path
from run_met1_audit import load, sha, write, CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    files = {}
    sources = {}
    for phase in ('81','84'):
        root = a.repo / f'outputs/phase{phase}/phase{phase}_20261011'
        original = load(root/'artifact_hashes.json')
        names = ('confusion_matrices.json','development_groups.csv','summary.csv',
                 f'phase{phase}_report.json','LOCAL_VERIFIED.json','artifact_hashes.json','input_configuration_lock.json')
        assert load(root/'LOCAL_VERIFIED.json')['status'] == 'passed'
        (a.output/phase).mkdir()
        for name in names:
            source = root/name
            if name in original:
                assert sha(source) == original[name], source
            dest = a.output/phase/name
            shutil.copyfile(source, dest)
            files[phase+'/'+name] = sha(dest)
            sources[phase+'/'+name] = str(source.resolve())
        lock = load(root/'input_configuration_lock.json')
        for name in ('phase65d_metrics.py','run_phase81.py') + (('run_phase84.py',) if phase == '84' else ()):
            rel = 'src/tools/'+name
            expected = lock['code_sha256' if phase == '81' else 'code_files'][rel]
            blob = subprocess.check_output(['git','show',lock['git_commit']+':'+rel],cwd=a.repo)
            dest = a.output/'evidence'/name
            dest.parent.mkdir(exist_ok=True)
            if dest.exists(): assert dest.read_bytes() == blob
            else: dest.write_bytes(blob)
            assert sha(dest) == expected, rel
            files['evidence/'+name] = expected
            sources['evidence/'+name] = lock['git_commit']+':'+rel
    config = dict(files=files, original_sources=sources,
                  axis=dict(rows='truth',columns='prediction',classes=list(CLASSES)),
                  axis_evidence='original Git blobs: confusion uses truth*3+pred, group_row records row reference support and explicit class names',
                  no_images_or_large_caches=True)
    write(a.output/'input_manifest.json', config)
    assert sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file()) < 128*1024**2
    print('Prepared',len(files),'SHA-bound small aggregate and code files at',a.output)


if __name__ == '__main__':
    main()

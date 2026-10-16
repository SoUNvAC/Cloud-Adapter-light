"""Post-result read-only verification; no model construction or reference access."""
import hashlib,json,subprocess
from pathlib import Path
from phase123_common import setup,ROOT,REPO,load,sha
def main():
    cfg=setup();out=REPO/'src/research_results/phase123_20261016'
    manifest=load(out/'GIT_CONTENT_SHA256SUMS.json')['files']
    for name,h in manifest.items():
        b=subprocess.check_output(['git','show','HEAD:src/research_results/phase123_20261016/'+name],cwd=REPO)
        assert hashlib.sha256(b).hexdigest()==h,name
    complete=load(out/'COMPLETION.json');model=sha(ROOT/'C_decoder_final.safetensors')
    assert model==complete['decoder_sha256']==load(ROOT/'TRAINING_SEAL.json')['models']['C']
    split=sha(REPO/'src/work_dirs/phase65a/split_lock.json');assert split==cfg['original_split_sha256']
    assert (ROOT/'EXIT_CODE').read_text().strip()==(ROOT/'PIPELINE_EXIT_CODE').read_text().strip()=='0'
    control=load(ROOT/'CONTROLLER_STATUS.json');assert control['calls_completed']==2392
    for name in ('TRAINING_SEAL.json','THRESHOLD_LOCK.json','ALL_SCORES_SEALED.json'):
        assert load(ROOT/name)==load(out/name),name
    live=[]
    for p in ROOT.rglob('PID'):
        try:
            pid=int(p.read_text());q=Path('/proc')/str(pid)
            if (q/'stat').read_text().split()[2]!='Z' and 'phase123' in (q/'cmdline').read_text():live.append(pid)
        except (FileNotFoundError,ProcessLookupError,ValueError):pass
    assert not live,live
    print(json.dumps(dict(status='passed',HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        public_git_files_verified=len(manifest),pre_source_and_previous_artifacts_verified=True,decoder_sha256=model,original_split_sha256=split,
        model_calls=2392,controller_wall_including_package_seconds=control['elapsed_seconds'],live_phase123_jobs=live,EXIT_CODE=0,PIPELINE_EXIT_CODE=0),indent=2))
if __name__=='__main__':main()

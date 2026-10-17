"""Read-only verification of published Phase124 bytes; never rerun the audit."""
import hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'src/research_results/phase124_20261017'
def main():
    for name,h in json.loads((O/'PRE_EXECUTION_LOCK.json').read_text())['source_sha256'].items():
        assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
    manifest=json.loads((O/'GIT_CONTENT_SHA256SUMS.json').read_text())['files']
    for name,h in manifest.items():
        raw=subprocess.check_output(['git','show','HEAD:src/research_results/phase124_20261017/'+name],cwd=R)
        assert hashlib.sha256(raw).hexdigest()==h,name
    done=json.loads((O/'COMPLETION.json').read_text())
    for key in ('model_calls','training','fitting','new_downloads','bootstrap_recomputations','threshold_searches','pixels_read','work_dirs_transfers'):assert done[key]==0,key
    print(json.dumps(dict(status='passed',HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),files=len(manifest),
        model_calls=0,new_bootstrap=0,threshold_searches=0,no_audit_or_model_rerun=True),indent=2))
if __name__=='__main__':main()

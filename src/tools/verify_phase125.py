"""Read-only documentation/hash verification; no historical re-audit or models."""
import hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'src/research_results/phase125_20261017'
def main():
    for n,h in json.loads((O/'PRE_EXECUTION_LOCK.json').read_text())['source_sha256'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
    manifest=json.loads((O/'GIT_CONTENT_SHA256SUMS.json').read_text())['files']
    for n,h in manifest.items():
        raw=subprocess.check_output(['git','show','HEAD:src/research_results/phase125_20261017/'+n],cwd=R)
        assert hashlib.sha256(raw).hexdigest()==h,n
    done=json.loads((O/'CLOSEOUT.json').read_text());assert done['covered_phases']==list(range(119,126))
    for k in ('model_calls','training','fitting','downloads','pixels_read','model_binary_reads','threshold_searches','new_bootstrap','new_seed'):assert done[k]==0,k
    assert not done['subsequent_phase_started']
    print(json.dumps(dict(status='passed',HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),files=len(manifest),branch='RegNet119-125',no_experiments_added=True),indent=2))
if __name__=='__main__':main()

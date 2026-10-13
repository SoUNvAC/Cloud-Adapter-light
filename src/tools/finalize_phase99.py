"""Verify existing immutable Phase99 delivery after a row-count assertion fix."""
import hashlib,json,subprocess
from pathlib import Path
from verify_phase99_delivery import verify

def main():
    root=Path('/home/scv/shared/phase99_20261012');control=Path('/home/scv/shared/phase99_control_20261012')
    assert (control/'EXIT_CODE').read_text().strip()=='1'
    error=(control/'error.log').read_text()
    assert 'assert rowcount==1275' in error and error.rstrip().endswith('AssertionError')
    assert not (root/'COMPLETED.json').exists() and not (root/'RECOVERY_VERIFIED.json').exists()
    acceptance=verify(root/'delivery')
    receipt=dict(acceptance=acceptance,original_EXIT_CODE=1,reason='Verifier expected1275 instead of3*3*(53+32)=765 rows',
        no_rescoring=True,no_threshold_rescan=True,immutable_delivery_preserved=True,
        verification_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        verification_code_sha256=hashlib.sha256(Path(__file__).with_name('verify_phase99_delivery.py').read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((root/'delivery/DELIVERY_SHA256.json').read_bytes()).hexdigest())
    for name in ['RECOVERY_VERIFIED.json','COMPLETED.json']:
        with (root/name).open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
